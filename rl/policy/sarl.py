"""Self-Attention RL (SARL) policy integration for CrowdNavigationMPC."""

import logging
import os
from typing import Optional

import torch
import torch.nn as nn

from crowd_sim_plus.envs.utils.state_plus import JointState
from sicnav.policy.cadrl import mlp
from sicnav.policy.multi_human_rl import MultiHumanRL
from sicnav.utils.static_line_to_obs import lines_to_observable_states



class ValueNetwork(nn.Module):
    def __init__(self, input_dim, self_state_dim, mlp1_dims, mlp2_dims, mlp3_dims, attention_dims, with_global_state,
                 cell_size, cell_num):
        super().__init__()
        self.self_state_dim = self_state_dim
        self.global_state_dim = mlp1_dims[-1]
        self.mlp1 = mlp(input_dim, mlp1_dims, last_relu=True)
        self.mlp2 = mlp(mlp1_dims[-1], mlp2_dims)
        self.with_global_state = with_global_state
        if with_global_state:
            self.attention = mlp(mlp1_dims[-1] * 2, attention_dims)
        else:
            self.attention = mlp(mlp1_dims[-1], attention_dims)
        self.cell_size = cell_size
        self.cell_num = cell_num
        mlp3_input_dim = mlp2_dims[-1] + self.self_state_dim
        self.mlp3 = mlp(mlp3_input_dim, mlp3_dims)
        self.attention_weights = None

    def forward(self, state):
        """
        First transform the world coordinates to self-centric coordinates and then do forward computation

        :param state: tensor of shape (batch_size, # of humans, length of a rotated state)
        :return:
        """
        size = state.shape
        self_state = state[:, 0, :self.self_state_dim]
        mlp1_output = self.mlp1(state.view((-1, size[2])))
        mlp2_output = self.mlp2(mlp1_output)

        if self.with_global_state:
            # compute attention scores
            global_state = torch.mean(mlp1_output.view(size[0], size[1], -1), 1, keepdim=True)
            global_state = global_state.expand((size[0], size[1], self.global_state_dim)).\
                contiguous().view(-1, self.global_state_dim)
            attention_input = torch.cat([mlp1_output, global_state], dim=1)
        else:
            attention_input = mlp1_output
        scores = self.attention(attention_input).view(size[0], size[1], 1).squeeze(dim=2)

        # masked softmax
        # weights = softmax(scores, dim=1).unsqueeze(2)
        scores_exp = torch.exp(scores) * (scores != 0).float()
        weights = (scores_exp / torch.sum(scores_exp, dim=1, keepdim=True)).unsqueeze(2)
        self.attention_weights = weights[0, :, 0].data.cpu().numpy()

        # output feature is a linear combination of input features
        features = mlp2_output.view(size[0], size[1], -1)
        # for converting to onnx
        # expanded_weights = torch.cat([torch.zeros(weights.size()).copy_(weights) for _ in range(50)], dim=2)
        weighted_feature = torch.sum(torch.mul(weights, features), dim=1)

        # concatenate agent's state with global weighted humans' state
        joint_state = torch.cat([self_state, weighted_feature], dim=1)
        value = self.mlp3(joint_state)
        return value


class SARL(MultiHumanRL):
    def __init__(self):
        super().__init__()
        self.name = 'SARL'

    def configure(self, config):
        self.set_common_parameters(config)
        mlp1_dims = [int(x) for x in config.get('sarl', 'mlp1_dims').split(', ')]
        mlp2_dims = [int(x) for x in config.get('sarl', 'mlp2_dims').split(', ')]
        mlp3_dims = [int(x) for x in config.get('sarl', 'mlp3_dims').split(', ')]
        attention_dims = [int(x) for x in config.get('sarl', 'attention_dims').split(', ')]
        self.with_om = config.getboolean('sarl', 'with_om')
        with_global_state = config.getboolean('sarl', 'with_global_state')
        self.model = ValueNetwork(self.input_dim(), self.self_state_dim, mlp1_dims, mlp2_dims, mlp3_dims,
                                  attention_dims, with_global_state, self.cell_size, self.cell_num)
        self.multiagent_training = config.getboolean('sarl', 'multiagent_training')
        if self.with_om:
            self.name = 'OM-SARL'
        logging.info('Policy: {} {} global state'.format(self.name, 'w/' if with_global_state else 'w/o'))

    def get_attention_weights(self):
        return self.model.attention_weights
    
class SARLNav(SARL):
    """CrowdNavigation wrapper around the SARL policy with static obstacle support."""

    def __init__(self):
        super().__init__()
        self.name = 'SARLNav'
        self.include_static_obstacles = True
        self.static_obstacle_spacing = 0.4
        self.static_obstacle_radius: Optional[float] = None
        self.default_checkpoint = os.path.join(
            os.path.dirname(__file__), 'sarl', 'rl_model.pth'
        )
        self._weights_loaded = False

    def configure(self, config):
        super().configure(config)
        nav_section = 'sarl_nav'
        if config.has_section(nav_section):
            if config.has_option(nav_section, 'include_static'):
                self.include_static_obstacles = config.getboolean(nav_section, 'include_static')
            if config.has_option(nav_section, 'static_spacing'):
                spacing = max(config.getfloat(nav_section, 'static_spacing'), 1e-3)
                self.static_obstacle_spacing = spacing
            if config.has_option(nav_section, 'static_radius'):
                self.static_obstacle_radius = config.getfloat(nav_section, 'static_radius')

        checkpoint = self._resolve_checkpoint_path(config)
        self._load_checkpoint(checkpoint)

    def compute_cost(self, state):
        if any(value is None for value in [self.gc, self.gc_resolution, self.gc_width, self.gc_ox, self.gc_oy]):
            return 0.0
        return super().compute_cost(state)

    def predict(self, state):
        augmented_state = self._augment_with_static_obstacles(state)
        return super().predict(augmented_state)

    def _augment_with_static_obstacles(self, state):
        if not self.include_static_obstacles:
            return state
        static_segments = getattr(state, 'static_obs', None)
        if not static_segments:
            return state
        radius = self.static_obstacle_radius
        if radius is None and hasattr(state.self_state, 'radius'):
            radius = float(state.self_state.radius)
        pseudo_obstacles = lines_to_observable_states(
            static_segments,
            spacing=self.static_obstacle_spacing,
            radius=radius or 0.3,
        )
        if not pseudo_obstacles:
            return state
        augmented_humans = list(state.human_states) #+ pseudo_obstacles
        return JointState(state.self_state, augmented_humans, state.static_obs)

    def _resolve_checkpoint_path(self, config):
        search_order = [
            ('sarl_nav', 'checkpoint'),
            ('sarl_nav', 'weights'),
            ('sarl', 'checkpoint'),
            ('sarl', 'weights'),
        ]
        candidate: Optional[str] = None
        for section, option in search_order:
            if config.has_section(section) and config.has_option(section, option):
                value = config.get(section, option).strip()
                if value:
                    candidate = value
                    break
        if not candidate:
            candidate = self.default_checkpoint
        if candidate and not os.path.isabs(candidate):
            candidate = os.path.join(os.path.dirname(__file__), candidate)
        return candidate

    def _load_checkpoint(self, checkpoint_path):
        if not checkpoint_path:
            logging.warning('No SARL checkpoint path provided; using random initialization.')
            return
        if not os.path.exists(checkpoint_path):
            logging.warning('SARL checkpoint not found at %s; using random initialization.', checkpoint_path)
            return
        map_location = self.device if self.device is not None else torch.device('cpu')
        try:
            state_dict = torch.load(checkpoint_path, map_location=map_location)
            self.model.load_state_dict(state_dict)
            self._weights_loaded = True
            logging.info('Loaded SARL checkpoint from %s', checkpoint_path)
        except Exception as exc:
            logging.warning('Failed to load SARL checkpoint %s: %s', checkpoint_path, exc)