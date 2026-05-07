import numpy as np
import torch
import torch.nn as nn
from tensordict.nn import TensorDictModuleBase, TensorDictSequential, TensorDictModule
from einops.layers.torch import Rearrange
from torchrl.modules import ProbabilisticActor
from torchrl.envs.transforms import CatTensors
import torch
import os
from torchrl.data import CompositeSpec, UnboundedContinuousTensorSpec
from tensordict.tensordict import TensorDict
from torchrl.envs.utils import ExplorationType, set_exploration_type

from dataclasses import dataclass, field

@dataclass
class FeatureExtractorConfig:
    learning_rate: float = 5e-4
    dyn_obs_num: int = 5

@dataclass
class ActorConfig:
    learning_rate: float = 5e-4
    clip_ratio: float = 0.1
    action_limit: float = 2.0  # m/s

@dataclass
class CriticConfig:
    learning_rate: float = 5e-4
    clip_ratio: float = 0.1

@dataclass
class AlgoConfig:
    feature_extractor: FeatureExtractorConfig = field(default_factory=FeatureExtractorConfig)
    actor: ActorConfig = field(default_factory=ActorConfig)
    critic: CriticConfig = field(default_factory=CriticConfig)
    
    entropy_loss_coefficient: float = 1e-3
    training_frame_num: int = 32
    training_epoch_num: int = 4
    num_minibatches: int = 16

cfg = AlgoConfig()


import torch
import torch.nn as nn
from typing import Iterable, Union
from tensordict.tensordict import TensorDict
import numpy as np

class ValueNorm(nn.Module):
    def __init__(
        self,
        input_shape: Union[int, Iterable],
        beta=0.995,
        epsilon=1e-5,
    ) -> None:
        super().__init__()

        self.input_shape = (
            torch.Size(input_shape)
            if isinstance(input_shape, Iterable)
            else torch.Size((input_shape,))
        )
        self.epsilon = epsilon
        self.beta = beta

        self.running_mean: torch.Tensor
        self.running_mean_sq: torch.Tensor
        self.debiasing_term: torch.Tensor
        self.register_buffer("running_mean", torch.zeros(input_shape))
        self.register_buffer("running_mean_sq", torch.zeros(input_shape))
        self.register_buffer("debiasing_term", torch.tensor(0.0))

        self.reset_parameters()

    def reset_parameters(self):
        self.running_mean.zero_()
        self.running_mean_sq.zero_()
        self.debiasing_term.zero_()

    def running_mean_var(self):
        debiased_mean = self.running_mean / self.debiasing_term.clamp(min=self.epsilon)
        debiased_mean_sq = self.running_mean_sq / self.debiasing_term.clamp(
            min=self.epsilon
        )
        debiased_var = (debiased_mean_sq - debiased_mean**2).clamp(min=1e-2)
        return debiased_mean, debiased_var

    @torch.no_grad()
    def update(self, input_vector: torch.Tensor):
        assert input_vector.shape[-len(self.input_shape) :] == self.input_shape
        dim = tuple(range(input_vector.dim() - len(self.input_shape)))
        batch_mean = input_vector.mean(dim=dim)
        batch_sq_mean = (input_vector**2).mean(dim=dim)

        weight = self.beta

        self.running_mean.mul_(weight).add_(batch_mean * (1.0 - weight))
        self.running_mean_sq.mul_(weight).add_(batch_sq_mean * (1.0 - weight))
        self.debiasing_term.mul_(weight).add_(1.0 * (1.0 - weight))

    def normalize(self, input_vector: torch.Tensor):
        assert input_vector.shape[-len(self.input_shape) :] == self.input_shape
        mean, var = self.running_mean_var()
        out = (input_vector - mean) / torch.sqrt(var)
        return out

    def denormalize(self, input_vector: torch.Tensor):
        assert input_vector.shape[-len(self.input_shape) :] == self.input_shape
        mean, var = self.running_mean_var()
        out = input_vector * torch.sqrt(var) + mean
        return out

def make_mlp(num_units):
    layers = []
    for n in num_units:
        layers.append(nn.LazyLinear(n))
        layers.append(nn.LeakyReLU())
        layers.append(nn.LayerNorm(n))
    return nn.Sequential(*layers)

class IndependentNormal(torch.distributions.Independent):
    arg_constraints = {"loc": torch.distributions.constraints.real, "scale": torch.distributions.constraints.positive} 
    def __init__(self, loc, scale, validate_args=None):
        scale = torch.clamp_min(scale, 1e-6)
        base_dist = torch.distributions.Normal(loc, scale)
        super().__init__(base_dist, 1, validate_args=validate_args)

class IndependentBeta(torch.distributions.Independent):
    arg_constraints = {"alpha": torch.distributions.constraints.positive, "beta": torch.distributions.constraints.positive}

    def __init__(self, alpha, beta, validate_args=None):
        beta_dist = torch.distributions.Beta(alpha, beta)
        super().__init__(beta_dist, 1, validate_args=validate_args)

class Actor(nn.Module):
    def __init__(self, action_dim: int) -> None:
        super().__init__()
        self.actor_mean = nn.LazyLinear(action_dim)
        self.actor_std = nn.Parameter(torch.zeros(action_dim)) 
    
    def forward(self, features: torch.Tensor):
        loc = self.actor_mean(features)
        scale = torch.exp(self.actor_std).expand_as(loc)
        return loc, scale

class BetaActor(nn.Module):
    def __init__(self, action_dim: int) -> None:
        super().__init__()
        self.alpha_layer = nn.LazyLinear(action_dim)
        self.beta_layer = nn.LazyLinear(action_dim)
        self.alpha_softplus = nn.Softplus()
        self.beta_softplus = nn.Softplus()
    
    def forward(self, features: torch.Tensor):
        alpha = 1. + self.alpha_softplus(self.alpha_layer(features)) + 1e-6
        beta = 1. + self.beta_softplus(self.beta_layer(features)) + 1e-6
        # print("alpha: ", alpha)
        # print("beta: ", beta)
        return alpha, beta

class GAE(nn.Module):
    def __init__(self, gamma, lmbda):
        super().__init__()
        self.register_buffer("gamma", torch.tensor(gamma))
        self.register_buffer("lmbda", torch.tensor(lmbda))
        self.gamma: torch.Tensor
        self.lmbda: torch.Tensor
    
    def forward(
        self, 
        reward: torch.Tensor, 
        terminated: torch.Tensor, 
        value: torch.Tensor, 
        next_value: torch.Tensor
    ):
        num_steps = terminated.shape[1]
        advantages = torch.zeros_like(reward)
        not_done = 1 - terminated.float()
        gae = 0
        for step in reversed(range(num_steps)):
            delta = (
                reward[:, step] 
                + self.gamma * next_value[:, step] * not_done[:, step] 
                - value[:, step]
            )
            advantages[:, step] = gae = delta + (self.gamma * self.lmbda * not_done[:, step] * gae) 
        returns = advantages + value
        return advantages, returns

def make_batch(tensordict: TensorDict, num_minibatches: int):
    tensordict = tensordict.reshape(-1) 
    perm = torch.randperm(
        (tensordict.shape[0] // num_minibatches) * num_minibatches,
        device=tensordict.device,
    ).reshape(num_minibatches, -1)
    for indices in perm:
        yield tensordict[indices]



def vec_to_new_frame(vec, goal_direction):
    if (len(vec.size()) == 1):
        vec = vec.unsqueeze(0)
    # print("vec: ", vec.shape)

    # goal direction x
    goal_direction_x = goal_direction / goal_direction.norm(dim=-1, keepdim=True)
    z_direction = torch.tensor([0, 0, 1.], device=vec.device)
    
    # goal direction y
    goal_direction_y = torch.cross(z_direction.expand_as(goal_direction_x), goal_direction_x)
    goal_direction_y /= goal_direction_y.norm(dim=-1, keepdim=True)
    
    # goal direction z
    goal_direction_z = torch.cross(goal_direction_x, goal_direction_y)
    goal_direction_z /= goal_direction_z.norm(dim=-1, keepdim=True)

    n = vec.size(0)
    if len(vec.size()) == 3:
        vec_x_new = torch.bmm(vec.view(n, vec.shape[1], 3), goal_direction_x.view(n, 3, 1)) 
        vec_y_new = torch.bmm(vec.view(n, vec.shape[1], 3), goal_direction_y.view(n, 3, 1))
        vec_z_new = torch.bmm(vec.view(n, vec.shape[1], 3), goal_direction_z.view(n, 3, 1))
    else:
        vec_x_new = torch.bmm(vec.view(n, 1, 3), goal_direction_x.view(n, 3, 1))
        vec_y_new = torch.bmm(vec.view(n, 1, 3), goal_direction_y.view(n, 3, 1))
        vec_z_new = torch.bmm(vec.view(n, 1, 3), goal_direction_z.view(n, 3, 1))

    vec_new = torch.cat((vec_x_new, vec_y_new, vec_z_new), dim=-1)

    return vec_new


def vec_to_world(vec, goal_direction):
    world_dir = torch.tensor([1., 0, 0], device=vec.device).expand_as(goal_direction)
    
    # directional vector of world coordinate expressed in the local frame
    world_frame_new = vec_to_new_frame(world_dir, goal_direction)

    # convert the velocity in the local target coordinate to the world coodirnate
    world_frame_vel = vec_to_new_frame(vec, world_frame_new)
    return world_frame_vel

# State transformation
def get_robot_state(pos, goal, vel, target_dir, device="cuda"):
    rpos = np.zeros(3)
    rpos[:2] = goal - pos
    vel3 = np.zeros(3)
    vel3[:2] = vel

    distance = np.linalg.norm(rpos)
    distance_2d = np.linalg.norm(rpos[:2])
    distance_z = 0

    target_dir_2d = np.zeros(3)
    target_dir_2d[:2] = target_dir

    rpos_clipped = rpos / max(distance, 1e-6)

    rpos_clipped_g = vec_to_new_frame(torch.tensor(rpos_clipped, dtype=torch.float),
                                      torch.tensor(target_dir_2d, dtype=torch.float))
    vel_g = vec_to_new_frame(torch.tensor(vel3, dtype=torch.float),
                             torch.tensor(target_dir_2d, dtype=torch.float))

    d2 = torch.tensor(distance_2d, dtype=torch.float).view(1, 1, 1)
    dz = torch.tensor(distance_z, dtype=torch.float).view(1, 1, 1)

    return torch.cat([rpos_clipped_g, d2, dz, vel_g], dim=-1).squeeze(0).to(device)

# Raycasting (geometry-based)
def ray_cast_distance(robot_pos, angle, obstacles, max_range=4.0, safety_margin=0.1):
    dx = np.cos(angle)
    dy = np.sin(angle)
    min_dist = max_range

    for ox, oy, r in obstacles:
        cx = ox - robot_pos[0]
        cy = oy - robot_pos[1]

        proj = cx * dx + cy * dy
        if proj < 0 or proj > max_range:
            continue

        closest_x = robot_pos[0] + proj * dx
        closest_y = robot_pos[1] + proj * dy

        dist_to_center = np.hypot(ox - closest_x, oy - closest_y)
        if dist_to_center <= r + safety_margin:
            adjusted_dist = max(proj - r - safety_margin, 0.0)
            min_dist = min(min_dist, adjusted_dist)

    return min_dist

def get_ray_cast(robot_pos, obstacles, max_range=4.0,
                          hres_deg=10.0,
                          vfov_angles_deg=[-10.0, 0.0, 10.0, 20.0],
                          start_angle_deg=0.0,
                          device="cuda"):
                          
    num_h = int(360 / hres_deg)
    num_v = len(vfov_angles_deg)

    range_matrix = np.full((num_h, num_v), max_range)
    v0_idx = vfov_angles_deg.index(0.0)
    ray_segments_2d = []

    for h in range(num_h):
        h_angle_deg = start_angle_deg + h * hres_deg
        h_angle_rad = np.deg2rad(h_angle_deg)

        dist = ray_cast_distance(robot_pos, h_angle_rad, obstacles, max_range, 0.0)
        range_matrix[h, 1:4] = dist

        x_end = robot_pos[0] + dist * np.cos(h_angle_rad)
        y_end = robot_pos[1] + dist * np.sin(h_angle_rad)
        ray_segments_2d.append(((robot_pos[0], robot_pos[1]), (x_end, y_end)))

    static_obs_input = np.maximum(range_matrix, 0.1)
    static_obs_input = max_range - static_obs_input
    static_obs_input = torch.tensor(static_obs_input, dtype=torch.float, device=device).unsqueeze(0).unsqueeze(0)
    return static_obs_input, range_matrix, ray_segments_2d


def get_dyn_obs_state(pos, vel, robot_positions, robot_velocities, target_dir, robot_size=0.25, max_range=2.0, max_num=5, device="cuda"):
    """
    pos:              torch.Tensor (2,)     - current robot position
    vel:              torch.Tensor (2,)     - current robot velocity
    robot_positions:  List[np.ndarray]      - positions of all robots
    robot_velocities: List[np.ndarray]      - velocities of all robots
    """

    # Convert input
    pos = torch.tensor(pos, dtype=torch.float, device=device)
    vel = torch.tensor(vel, dtype=torch.float, device=device)
    others_pos = torch.tensor(robot_positions, dtype=torch.float, device=device)
    others_vel = torch.tensor(robot_velocities, dtype=torch.float, device=device)

    # Filter out self by checking if position matches
    dists = torch.norm(others_pos - pos, dim=-1) # 返回每个机器人与当前机器人的距离
    mask = dists > 1e-4  # exclude self
    others_pos = others_pos[mask]
    others_vel = others_vel[mask]
    dists = dists[mask]

    # Keep only those within range
    in_range_mask = dists < max_range
    others_pos = others_pos[in_range_mask]
    others_vel = others_vel[in_range_mask]
    dists = dists[in_range_mask]
    if len(others_pos) == 0:
        return torch.zeros((1, 1, max_num, 10), dtype=torch.float, device=device)

    # Sort by distance
    sorted_indices = torch.argsort(dists)
    others_pos = others_pos[sorted_indices]
    others_vel = others_vel[sorted_indices]

    # Select top-k
    num_dyn = min(max_num, others_pos.shape[0])
    closest_pos = others_pos[:num_dyn]
    closest_vel = others_vel[:num_dyn]

    # Relative position (3D) and velocity (3D)
    rel_pos = torch.zeros((num_dyn, 1, 3), device=device)
    rel_vel = torch.zeros((num_dyn, 1, 3), device=device)
    rel_pos[:, :, :2] = (closest_pos.squeeze(1) - pos).unsqueeze(1)
    rel_vel[:, :, :2] = closest_vel.unsqueeze(1)
    target_dir_3d = torch.zeros(num_dyn, 3, device=device)
    target_dir_3d[:, :2] = target_dir[:, :, :2]

    # Transform to local frame
    rel_pos_g = vec_to_new_frame(rel_pos, target_dir_3d)
    rel_vel_g = vec_to_new_frame(rel_vel, target_dir_3d)

    # Distance components
    dist_2d = rel_pos_g[:, :, :2].norm(dim=-1, keepdim=True)
    dist_z = torch.zeros(num_dyn, 1, dtype=torch.float, device=device).unsqueeze(-1)
    rel_pos_gn = rel_pos_g / rel_pos.norm(dim=-1, keepdim=True).clamp(min=1e-6)

    # Width and height (for now fixed or placeholder)
    width = torch.zeros((num_dyn, 1), device=device)
    height = torch.zeros((num_dyn, 1), device=device)


    # Compose state
    dyn_state = torch.cat([
        rel_pos_gn,         # (x, y, z) unit vec
        dist_2d,            # scalar
        dist_z,             # scalar
        rel_vel_g,          # (vx, vy, vz)
        width.unsqueeze(1), height.unsqueeze(1)       # size hints
    ], dim=-1).squeeze(1)

    # Pad if needed
    if num_dyn < max_num:
        padding = torch.zeros((max_num - num_dyn, 10), device=device)
        dyn_state = torch.cat([dyn_state, padding], dim=0)

    return dyn_state.unsqueeze(0).unsqueeze(0) # [1, 1, max_num, 10]


class PPO(TensorDictModuleBase):
    def __init__(self, cfg, observation_spec, action_spec, device):
        super().__init__()
        self.cfg = cfg
        self.device = device

        
        # Feature extractor for LiDAR
        feature_extractor_network = nn.Sequential(
            nn.LazyConv2d(out_channels=4, kernel_size=(5, 3), padding=(2, 1)), nn.ELU(), 
            nn.LazyConv2d(out_channels=16, kernel_size=(5, 3), stride=(2, 1), padding=(2, 1)), nn.ELU(),
            nn.LazyConv2d(out_channels=16, kernel_size=(5, 3), stride=(2, 2), padding=(2, 1)), nn.ELU(),
            Rearrange("n c w h -> n (c w h)"),
            nn.LazyLinear(128), nn.LayerNorm(128),
        ).to(self.device)
        
        # Dynamic obstacle information extractor
        dynamic_obstacle_network = nn.Sequential(
            Rearrange("n c w h -> n (c w h)"),
            make_mlp([128, 64])
        ).to(self.device)

        # Feature extractor
        self.feature_extractor = TensorDictSequential(
            TensorDictModule(feature_extractor_network, [("agents", "observation", "lidar")], ["_cnn_feature"]),
            TensorDictModule(dynamic_obstacle_network, [("agents", "observation", "dynamic_obstacle")], ["_dynamic_obstacle_feature"]),
            CatTensors(["_cnn_feature", ("agents", "observation", "state"), "_dynamic_obstacle_feature"], "_feature", del_keys=False), 
            TensorDictModule(make_mlp([256, 256]), ["_feature"], ["_feature"]),
        ).to(self.device)

        # Actor network: infer dimensions robustly from CompositeSpec
        agents_action_spec = None
        try:
            agents_action_spec = action_spec["agents"]["action"]
        except Exception:
            try:
                agents_action_spec = action_spec.get(("agents", "action"))
            except Exception:
                agents_action_spec = None
        if agents_action_spec is not None and hasattr(agents_action_spec, "shape"):
            self.action_dim = int(agents_action_spec.shape[-1])
        else:
            self.action_dim = 3
        try:
            self.n_agents = int(action_spec.shape[0])
        except Exception:
            self.n_agents = 1
        self.actor = ProbabilisticActor(
            TensorDictModule(BetaActor(self.action_dim), ["_feature"], ["alpha", "beta"]),
            in_keys=["alpha", "beta"],
            out_keys=[("agents", "action_normalized")], 
            distribution_class=IndependentBeta,
            return_log_prob=True
        ).to(self.device)

        # Critic network
        self.critic = TensorDictModule(
            nn.LazyLinear(1), ["_feature"], ["state_value"] 
        ).to(self.device)
        self.value_norm = ValueNorm(1).to(self.device)

        # Loss related
        self.gae = GAE(0.99, 0.95) # generalized adavantage esitmation
        self.critic_loss_fn = nn.HuberLoss(delta=10) # huberloss (L1+L2): https://pytorch.org/docs/stable/generated/torch.nn.HuberLoss.html

        # Optimizer
        self.feature_extractor_optim = torch.optim.Adam(self.feature_extractor.parameters(), lr=cfg.feature_extractor.learning_rate)
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor.learning_rate)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=cfg.actor.learning_rate)

        # Dummy Input for nn lazymodule
        dummy_input = observation_spec.zero()
        # print("dummy_input: ", dummy_input)


        self.__call__(dummy_input)

        # Initialize network
        def _init(m):
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, 1)
                nn.init.constant_(m.bias, 0.)
        self.actor.apply(_init)
        self.critic.apply(_init)

    def __call__(self, tensordict):
        self.feature_extractor(tensordict)
        self.actor(tensordict)
        self.critic(tensordict)

        # Cooridnate change: transform local to world
        actions = (2 * tensordict["agents", "action_normalized"] * self.cfg.actor.action_limit) - self.cfg.actor.action_limit
        actions_world = vec_to_world(actions, tensordict["agents", "observation", "direction"])
        tensordict["agents", "action"] = actions_world
        return tensordict

    def train(self, tensordict):
        # tensordict: (num_env, num_frames, dim), batchsize = num_env * num_frames
        next_tensordict = tensordict["next"]
        with torch.no_grad():
            next_tensordict = torch.vmap(self.feature_extractor)(next_tensordict) # calculate features for next state value calculation
            next_values = self.critic(next_tensordict)["state_value"]
        rewards = tensordict["next", "agents", "reward"] # Reward obtained by state transition
        dones = tensordict["next", "terminated"] # Whether the next states are terminal states

        values = tensordict["state_value"] # This is calculated stored when we called forward to obtain actions
        values = self.value_norm.denormalize(values) # denomalize values based on running mean and var of return
        next_values = self.value_norm.denormalize(next_values)

        # calculate GAE: Generalized Advantage Estimation
        adv, ret = self.gae(rewards, dones, values, next_values)
        adv_mean = adv.mean()
        adv_std = adv.std()
        adv = (adv - adv_mean) / adv_std.clip(1e-7)
        self.value_norm.update(ret) # update running mean and var for return
        ret = self.value_norm.normalize(ret)  # normalize return
        tensordict.set("adv", adv)
        tensordict.set("ret", ret)

        # Training
        infos = []
        for epoch in range(self.cfg.training_epoch_num):
            batch = make_batch(tensordict, self.cfg.num_minibatches)
            for minibatch in batch:
                infos.append(self._update(minibatch))
        # Aggregate metrics
        out = {}
        for td in infos:
            for k, v in td.items():
                out.setdefault(k, []).append(v.detach())
        out = {k: torch.stack(vs).mean().item() for k, vs in out.items()}
        return out    

    
    def _update(self, tensordict): # tensordict shape (batch_size, )
        self.feature_extractor(tensordict)

        # Get action from the current policy
        action_dist = self.actor.get_dist(tensordict) # this does an actor forward to get "loc" and "scale" and use them to build multivariate normal distribution
        log_probs = action_dist.log_prob(tensordict[("agents", "action_normalized")]) # based on the gaussian, we can calculate the log prob of the action from the current policy

        # Entropy Loss
        action_entropy = action_dist.entropy()
        entropy_loss = -self.cfg.entropy_loss_coefficient * torch.mean(action_entropy)

        # Actor Loss
        advantage = tensordict["adv"] # the advantage is calculated based on GAE in hte previous step
        ratio = torch.exp(log_probs - tensordict["sample_log_prob"]).unsqueeze(-1)
        surr1 = advantage * ratio
        surr2 = advantage * ratio.clamp(1.-self.cfg.actor.clip_ratio, 1.+self.cfg.actor.clip_ratio)
        actor_loss = -torch.mean(torch.min(surr1, surr2)) * self.action_dim 

        # Critic Loss 
        b_value = tensordict["state_value"]
        ret = tensordict["ret"] # Return G
        value = self.critic(tensordict)["state_value"] 
        value_clipped = b_value + (value - b_value).clamp(-self.cfg.critic.clip_ratio, self.cfg.critic.clip_ratio) # this guarantee that critic update is clamped
        critic_loss_clipped = self.critic_loss_fn(ret, value_clipped)
        critic_loss_original = self.critic_loss_fn(ret, value)
        critic_loss = torch.max(critic_loss_clipped, critic_loss_original)

        # Total Loss
        loss = entropy_loss + actor_loss + critic_loss

        # Optimize
        self.feature_extractor_optim.zero_grad()
        self.actor_optim.zero_grad()
        self.critic_optim.zero_grad()
        loss.backward()

        actor_grad_norm = nn.utils.clip_grad.clip_grad_norm_(self.actor.parameters(), max_norm=5.) # to prevent gradient growing too large
        critic_grad_norm = nn.utils.clip_grad.clip_grad_norm_(self.critic.parameters(), max_norm=5.)
        self.feature_extractor_optim.step()
        self.actor_optim.step()
        self.critic_optim.step()
        explained_var = 1 - F.mse_loss(value, ret) / ret.var()
        return TensorDict({
            "actor_loss": actor_loss,
            "critic_loss": critic_loss,
            "entropy": entropy_loss,
            "actor_grad_norm": actor_grad_norm,
            "critic_grad_norm": critic_grad_norm,
            "explained_var": explained_var
        }, [])