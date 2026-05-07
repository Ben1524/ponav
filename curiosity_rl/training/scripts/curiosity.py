import pdb
import geoopt.manifolds.stereographic.math as gmath
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
import wandb
from geoopt.optim import RiemannianAdam

def make_mlp(num_units):
    layers = []
    for n in num_units:
        layers.append(nn.LazyLinear(n))
        layers.append(nn.LeakyReLU())
        layers.append(nn.LayerNorm(n))
    return nn.Sequential(*layers)

    
class NavRLModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lidar_net = nn.Sequential(
            nn.LazyConv2d(out_channels=4, kernel_size=[5, 3], padding=[2, 1]), nn.ELU(), 
            nn.LazyConv2d(out_channels=16, kernel_size=[5, 3], stride=[2, 1], padding=[2, 1]), nn.ELU(),
            nn.LazyConv2d(out_channels=16, kernel_size=[5, 3], stride=[2, 2], padding=[2, 1]), nn.ELU(),
            nn.Flatten(),
            nn.LazyLinear(128), nn.LayerNorm(128),
        )
        
        self.dyn_obs_net = nn.Sequential(
            nn.Flatten(),
            make_mlp([128, 64])
        )

        self.feature_net = make_mlp([256, 256])
    
    def forward(self, state):
        # state: (robot_state, lidar, dynamic_obstacle)
        # robot_state: (N, 8) or (N, 1, 8)
        # lidar: (N, 1, 36, 4)
        # dynamic_obstacle: (N, 1, 5, 10) or (N, 1, 5, 10)
        
        robot_state = state[0]
        lidar = state[1]
        dyn_obs = state[2]
        
        # Squeeze agent dimension if present
        if robot_state.dim() == 3:
            robot_state = robot_state.squeeze(1)
        if dyn_obs.dim() == 4:
            dyn_obs = dyn_obs.squeeze(1)
        
        lidar_feat = self.lidar_net(lidar)
        dyn_feat = self.dyn_obs_net(dyn_obs)
        
        # Concatenate: lidar_feat (128) + robot_state (8) + dyn_feat (64)
        combined = torch.cat([lidar_feat, robot_state, dyn_feat], dim=1)
        return self.feature_net(combined)

class ProjectToPoincare(nn.Module):
    def __init__(self, radius=0.99):
        super().__init__()
        self.radius = radius
    def forward(self, x):
        norm = torch.norm(x, dim=-1, keepdim=True)
        target_norm = torch.clamp(norm, max=self.radius)
        return x * (target_norm / (norm + 1e-6))

class HICM():
    """Implementation of:
    [1] Curiosity-driven Exploration by Self-supervised Prediction
    Pathak, Agrawal, Efros, and Darrell - UC Berkeley - ICML 2017.
    https://arxiv.org/pdf/1705.05363.pdf with an hyperbolic flavour

    Learns a simplified model of the environment based on three networks:
    1) Embedding observations into latent space ("feature" network).
    2) Predicting the action, given two consecutive embedded observations
    ("inverse" network).
    3) Predicting the next embedded obs, given an obs and action
    ("forward" network).

    The less the agent is able to predict the actually observed next feature
    vector, given obs and action (through the forwards network), the larger the
    "intrinsic reward", which will be added to the extrinsic reward.
    Therefore, if a state transition was unexpected, the agent becomes
    "curious" and will further explore this transition leading to better
    exploration in sparse rewards environments.
    """

    def __init__(self, config, robot_state_dim, dyn_obs_shape, static_obs_shape, device, lr, scaling_factor, policy="tree_search_rl", continuous_action=False, action_dim=None):
        self.scaling_factor = scaling_factor
        self.continuous_action = continuous_action
        
        if self.continuous_action:
            self.action_dim = action_dim if action_dim is not None else 2
            self.mse_loss = nn.MSELoss()
        else:
            self.action_num = config.action_space.speed_samples * config.action_space.rotation_samples + 1
            self.cross_entropy_loss = torch.nn.CrossEntropyLoss()
            
        self.device = device
        self.lr = lr
        self.beta = 0.2
        self.name = "HyperICM"
        self.embedding_dimension = 128
        if policy == "ponav":
            self._curiosity_feature_net = nn.Sequential(
                NavRLModel(),
                nn.Linear(256, self.embedding_dimension),
                ProjectToPoincare()
            ).to(device)
            # Dummy forward pass to initialize Lazy modules
            dummy_robot = torch.zeros(1, robot_state_dim).to(device)
            dummy_lidar = torch.zeros(1, 1, *static_obs_shape).to(device)
            dummy_dyn = torch.zeros(1, 1, *dyn_obs_shape).to(device)
            self._curiosity_feature_net((dummy_robot, dummy_lidar, dummy_dyn))
        else:
            raise ValueError("Curiosity not implemented for this method")

        if self.continuous_action:
            # Inverse: [embed, embed] -> action
            self._curiosity_inverse_fcnet = nn.Sequential(
                nn.Linear(2*self.embedding_dimension, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, self.action_dim)
            ).to(device)
            # Forward: [embed, action] -> embed
            self._curiosity_forward_fcnet = nn.Sequential(
                nn.Linear(self.embedding_dimension+self.action_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                ProjectToPoincare()
            ).to(device)
        else:
            self._curiosity_inverse_fcnet = nn.Sequential(
                nn.Linear(2*self.embedding_dimension, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, self.action_num)
            ).to(device)

            self._curiosity_forward_fcnet = nn.Sequential(
                nn.Linear(128+self.action_num, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                ProjectToPoincare()
            ).to(device)
        self.get_exploration_optimizer()

    def get_exploration_optimizer(self):
        # Create, but don't add Adam for curiosity NN updating to the policy.
        # If we added and returned it here, it would be used in the policy's
        # update loop, which we don't want (curiosity updating happens inside
        # `postprocess_trajectory`).
        feature_params = list(self._curiosity_feature_net.parameters())
        inverse_params = list(self._curiosity_inverse_fcnet.parameters())
        forward_params = list(self._curiosity_forward_fcnet.parameters())

        # Now that the Policy's own optimizer(s) have been created (from
        # the Model parameters (IMPORTANT: w/o(!) the curiosity params),
        # we can add our curiosity sub-modules to the Policy's Model.
        self._curiosity_feature_net = self._curiosity_feature_net.to(
            self.device
        )
        self._curiosity_inverse_fcnet = self._curiosity_inverse_fcnet.to(
            self.device
        )
        self._curiosity_forward_fcnet = self._curiosity_forward_fcnet.to(
            self.device
        )
        self._optimizer = torch.optim.Adam(
            forward_params + inverse_params + feature_params, lr=self.lr
        )

    def compute_poincare_distance(self, x, y):
        # Project to unit ball (numerical safety)
        def _project_ball(t, radius=0.95):
            norm = torch.linalg.norm(t, dim=-1, keepdim=True).clamp(min=1e-6)
            scale = torch.clamp(radius / norm, max=1.0)
            return t * scale

        x = _project_ball(x)
        y = _project_ball(y)

        sqdist = torch.sum((x - y) ** 2, dim=-1)
        squnorm = torch.sum(x ** 2, dim=-1).clamp(max=0.95)
        sqvnorm = torch.sum(y ** 2, dim=-1).clamp(max=0.95)
        denom = torch.clamp((1 - squnorm) * (1 - sqvnorm), min=1e-6)
        x_temp = 1 + 2 * sqdist / denom
        x_temp = torch.clamp(x_temp, min=1 + 1e-6)
        dist = torch.acosh(x_temp)
        dist = torch.nan_to_num(dist, nan=0.0, posinf=10.0, neginf=0.0)
        return dist
    
    def compute_intrinsic_reward(self, state, next_state, action):
        # When the reward is stored in memory and when new rewards are created in the tree search
        phi = self._curiosity_feature_net(state)
        
        next_phi = self._curiosity_feature_net(next_state)
        
        if self.continuous_action:
            # action: (1, action_dim)
            action_input = action.float()
            if action_input.dim() == 1:
                action_input = action_input.unsqueeze(0)
        else:
            action_input = F.one_hot(torch.Tensor([action]).to(torch.int64).to(phi.device), self.action_num).float()

        predicted_next_phi = self._curiosity_forward_fcnet(
            torch.cat((phi, action_input), 1)
        )

        #poincarè distance
        poincare_dist = self.compute_poincare_distance(next_phi, predicted_next_phi)

        intrinsic_reward = self.scaling_factor * poincare_dist # is that dist a scalar?
        return intrinsic_reward.data.cpu().numpy()[0]

    def compute_intrinsic_reward_batch(self, state, next_state, action):
        # When the reward is stored in memory and when new rewards are created in the tree search
        phi = self._curiosity_feature_net(state)

        next_phi = self._curiosity_feature_net(next_state)
        
        if self.continuous_action:
            action_input = action.float()
        else:
            action_input = F.one_hot(action.to(torch.int64), self.action_num).float()

        predicted_next_phi = self._curiosity_forward_fcnet(
            torch.cat((phi, action_input), 1)
        )
        
        poincare_dist = self.compute_poincare_distance(next_phi, predicted_next_phi)
        intrinsic_reward = self.scaling_factor * poincare_dist
        return intrinsic_reward.unsqueeze(1)
    
    def optimize(self, state, next_state, action):
        phi = self._curiosity_feature_net(state)

        next_phi = self._curiosity_feature_net(next_state)
        
        if self.continuous_action:
            real_actions = action.float()
        else:
            real_actions = F.one_hot(action.to(torch.int64), self.action_num).float()

        predicted_next_phi = self._curiosity_forward_fcnet(
            torch.cat((phi, real_actions), 1)
        )
        
        poincare_dist = self.compute_poincare_distance(next_phi, predicted_next_phi)

        forward_loss = torch.mean(poincare_dist)
        pred_actions = self._curiosity_inverse_fcnet(torch.cat((phi, next_phi), -1))
        
        if self.continuous_action:
            inverse_loss = self.mse_loss(pred_actions, real_actions)
        else:
            inverse_loss = self.cross_entropy_loss(pred_actions, action)

        loss = (1.0 - self.beta) * inverse_loss + self.beta * forward_loss
        # wandb.log({"curiosity/inverse_loss": inverse_loss.item(), "curiosity/forward_loss": forward_loss.item()})
        # print("\n\nLOGGED CURIOSITY METRICS\n\n")
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()


        