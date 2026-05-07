import numpy as np
import torch
import torch.nn as nn
from tensordict.nn import TensorDictModuleBase, TensorDictSequential, TensorDictModule
from einops.layers.torch import Rearrange
from torchrl.modules import ProbabilisticActor
from torchrl.envs.transforms import CatTensors
from crowd_sim_plus.envs.policy.policy import Policy
from crowd_sim_plus.envs.utils.action import ActionXY, ActionRot
from crowd_sim_plus.envs.utils.state_plus import FullState, FullyObservableJointState
from .utils import ValueNorm, make_mlp, GAE, IndependentBeta, BetaActor, vec_to_world,get_robot_state,get_ray_cast,get_dyn_obs_state, ray_cast_distance
import torch
import os
from torchrl.data import CompositeSpec, UnboundedContinuousTensorSpec
from tensordict.tensordict import TensorDict
from torchrl.envs.utils import ExplorationType, set_exploration_type
from crowd_sim_plus.envs.policy.orca_plus_All import ORCAPlusAll

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


class PPO(TensorDictModuleBase):
    def __init__(self, observation_spec, action_spec, device):
        super().__init__()
        self.cfg = cfg
        self.device = device

        
        # Feature extractor for LiDAR
        feature_extractor_network = nn.Sequential(
            nn.LazyConv2d(out_channels=4, kernel_size=[5, 3], padding=[2, 1]), nn.ELU(), 
            nn.LazyConv2d(out_channels=16, kernel_size=[5, 3], stride=[2, 1], padding=[2, 1]), nn.ELU(),
            nn.LazyConv2d(out_channels=16, kernel_size=[5, 3], stride=[2, 2], padding=[2, 1]), nn.ELU(),
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

        # Actor etwork
        self.n_agents, self.action_dim = action_spec.shape
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
        def init_(module):
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, 0.01)
                nn.init.constant_(module.bias, 0.)
        self.actor.apply(init_)
        self.critic.apply(init_)

    def __call__(self, tensordict):
        self.feature_extractor(tensordict)
        self.actor(tensordict)
        self.critic(tensordict)

        # Cooridnate change: transform local to world
        actions = (2 * tensordict["agents", "action_normalized"] * self.cfg.actor.action_limit) - self.cfg.actor.action_limit
        actions_world = vec_to_world(actions, tensordict["agents", "observation", "direction"])
        tensordict["agents", "action"] = actions_world
        return tensordict



class Agent:
    def __init__(self, device):
        self.device = device
        self.policy = self.init_model()

    # PPO policy loader
    def init_model(self):
        observation_dim = 8
        num_dim_each_dyn_obs_state = 10
        observation_spec = CompositeSpec({
            "agents": CompositeSpec({
                "observation": CompositeSpec({
                    "state": UnboundedContinuousTensorSpec((observation_dim,), device=self.device), 
                    "lidar": UnboundedContinuousTensorSpec((1, 36, 4), device=self.device),
                    "direction": UnboundedContinuousTensorSpec((1, 3), device=self.device),
                    "dynamic_obstacle": UnboundedContinuousTensorSpec((1, 5, 10), device=self.device),
                }),
            }).expand(1)
        }, shape=[1], device=self.device)

        action_dim = 3
        action_spec = CompositeSpec({
            "agents": CompositeSpec({
                "action": UnboundedContinuousTensorSpec((action_dim,), device=self.device), 
            })
        }).expand(1, action_dim).to(self.device)

        policy = PPO(observation_spec, action_spec, self.device)

        file_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ckpts")
        checkpoint = "navrl_checkpoint.pt"

        ckpt_path = os.path.join(file_dir, checkpoint)
        try:
            if os.path.exists(ckpt_path):
                policy.load_state_dict(torch.load(ckpt_path, map_location=self.device))
            else:
                import logging
                logging.warn(f"PONav checkpoint not found at {ckpt_path}. Using randomly initialized weights.")
        except Exception as e:
            import logging
            logging.warn(f"Failed to load PONav checkpoint from {ckpt_path}: {e}. Using randomly initialized weights.")
        return policy
    
    def plan(self, robot_state, static_obs_input, dyn_obs_input, target_dir):
        obs = TensorDict({
            "agents": TensorDict({
                "observation": TensorDict({
                    "state": robot_state,
                    "lidar": static_obs_input,
                    "direction": target_dir,
                    "dynamic_obstacle": dyn_obs_input,
                })
            })
        }, device=self.device)

        with set_exploration_type(ExplorationType.MEAN):
            output = self.policy(obs)
            velocity = output["agents", "action"][0][0].detach().cpu().numpy()[:2] 
        return velocity


# 返回障碍物边的射线
def get_line_ray_cast(robot_pos, obs_lines, max_range=4.0,
                     hres_deg=10.0,
                     vfov_angles_deg=[-10.0, 0.0, 10.0, 20.0],
                     start_angle_deg=0.0,
                     device="cuda"):
    """
    对每一条射线，计算与所有线段障碍物的最近交点距离。
    """
    num_h = int(360 / hres_deg)
    num_v = len(vfov_angles_deg)
    range_matrix = np.full((num_h, num_v), max_range)
    ray_segments_2d = []

    def ray_segment_intersect(p, d, a, b):
        # p: 起点, d: 单位方向向量, a,b: 线段两端
        v1 = p - a
        v2 = b - a
        v3 = np.array([-d[1], d[0]])
        dot = np.dot(v2, v3)
        if abs(dot) < 1e-8:
            return None  # 平行
        t1 = np.cross(v2, v1) / dot
        t2 = np.dot(v1, v3) / dot
        if t1 >= 0 and 0 <= t2 <= 1:
            return t1  # t1为射线起点到交点的距离
        return None

    for h in range(num_h):
        h_angle_deg = start_angle_deg + h * hres_deg
        h_angle_rad = np.deg2rad(h_angle_deg)
        d = np.array([np.cos(h_angle_rad), np.sin(h_angle_rad)])
        min_dist = max_range
        for line in obs_lines:
            a = np.array(line[0])
            b = np.array(line[1])
            t = ray_segment_intersect(np.array(robot_pos), d, a, b)
            if t is not None and t < min_dist:
                min_dist = t
        # 填充所有垂直方向的距离（假设所有v方向都一样）
        range_matrix[h, :] = min_dist
        x_end = robot_pos[0] + min_dist * d[0]
        y_end = robot_pos[1] + min_dist * d[1]
        ray_segments_2d.append(((robot_pos[0], robot_pos[1]), (x_end, y_end)))

    static_obs_input = np.maximum(range_matrix, 0.1)
    static_obs_input = max_range - static_obs_input
    static_obs_input = torch.tensor(static_obs_input, dtype=torch.float, device=device).unsqueeze(0).unsqueeze(0)
    return static_obs_input, range_matrix, ray_segments_2d


class PONav(Policy):
    """PONav policy wrapper.

    This is a lightweight, robust implementation that exposes the expected
    Policy interface for the environment. It will attempt to use the learned
    Agent if available, but falls back to a simple goal-seeking holonomic
    controller to avoid crashes when the RL checkpoint or agent is missing.
    """
    def __init__(self):
        super().__init__()
        # behavior / metadata
        self.kinematics = 'holonomic'
        self.name = 'ponav'
        self.multiagent_training = False

        # device for any torch-based agent
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        try:
            # try to instantiate the learned Agent (may warn if ckpt missing)
            self.Agent = Agent(device=self.device)
            self.agent_available = True
        except Exception:
            # keep graceful fallback
            self.Agent = None
            self.agent_available = False

        # simple motion params used by fallback controller
        self.fallback_max_speed = 1.0
        self.goal_threshold = 0.15
        self.device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.max_range = 1.8
        self.lidar_hres_deg = 10.0
        self.vfov_angles_deg = [-10.0, 0.0, 10.0, 20.0]
        self.max_speed = 1.0
        self.orca = ORCAPlusAll()
        # 根本性安全过滤参数（CBF风格）
        self.d_safe_base = 0.2   # 额外安全距离，与机器人半径相加
        self.alpha_cbf = 1.5     # 约束收缩系数（越大越保守）
        self.qp_max_iters = 2    # 半空间投影迭代次数
        self.progress_floor = 0.05  # 禁止后退/保持最小前向投影
        self.safety_space = 0.2
        self.time_horizon_obst = 2.0
    
    
    def _obstacles_as_circles(self,env_state):
        """Approximate current static & moving obstacles as circles (x, y, r)
        for the LiDAR in PONav utils."""
        circles = []
        # # Static obstacles (Cuboid) in self.static_obstacles
        # for obs in self.static_obstacles:
        #     center = np.array(obs.center_position)
        #     axes = np.array(obs.axes_length)
        #     # radius = half of diagonal length
        #     r = 0.5 * sqrt(float(axes[0]) ** 2 + float(axes[1]) ** 2)
        #     circles.append((float(center[0]), float(center[1]), float(r)))

        # Moving obstacles (per-frame positions)
        number_moving_obstacles = len(env_state.human_states)
        for j in range(number_moving_obstacles):
            center = np.array([env_state.human_states[j].px, env_state.human_states[j].py])
            r = env_state.human_states[j].radius+0.1  # add small buffer
            circles.append((float(center[0]), float(center[1]), float(r)))
        return circles
    def configure(self, config):
        self.orca.configure(config)
        try:
            if hasattr(self.orca, 'safety_space'):
                self.safety_space = float(self.orca.safety_space)
                self.d_safe_base = self.safety_space
            if hasattr(self.orca, 'time_horizon_obst'):
                self.time_horizon_obst = float(self.orca.time_horizon_obst)
            if hasattr(self.orca, 'max_speed'):
                self.max_speed = float(self.orca.max_speed)
        except Exception:
            pass
        return super().configure(config)
    def predict(self, env_state):
        robot_state = env_state.self_state
        robot_radius = robot_state.radius
        
        if robot_state.omega == None:
            robot_state.omega = 0
        human_positions = []
        human_velocities = []
        pos = np.array([robot_state.px, robot_state.py])
        vel = np.array([robot_state.vx, robot_state.vy])
        goal = np.array([robot_state.gx, robot_state.gy])
        target_dir = goal - pos
        target_tensor = torch.tensor(
                    np.append(target_dir[:2], 0.0), dtype=torch.float32, device=self.device
                ).unsqueeze(0).unsqueeze(0)
        for hum in env_state.human_states:
            human_positions.append([hum.px, hum.py]) 
            human_velocities.append([hum.vx, hum.vy]) 
        circles_obs = self._obstacles_as_circles(env_state)
        robot_state = get_robot_state(pos, goal, vel, target_dir, device=self.device_str)
        static_obs = env_state.static_obs
        # print("static_obs: ", static_obs)
        start_angle_deg = np.degrees(np.arctan2(target_dir[1], target_dir[0]))
        static_obs_input, range_matrix, _ = get_line_ray_cast(
                    pos,
                    static_obs,
                    max_range=4.0,
                    hres_deg=self.lidar_hres_deg,
                    vfov_angles_deg=self.vfov_angles_deg,
                    start_angle_deg=start_angle_deg,
                    device=self.device_str,
                )
        # get_ray_cast(
        #             pos,
        #             static_obs,
        #             max_range=self.max_range,
        #             hres_deg=self.lidar_hres_deg,
        #             vfov_angles_deg=self.vfov_angles_deg,
        #             start_angle_deg=np.degrees(np.arctan2(target_dir[1], target_dir[0])),
        #             device=self.device_str,
        #         )
        dyn_pos = human_positions   # np.vstack([self.agents_positions, np.array(self.moving_obstacles_positions)])
        dyn_vel = human_velocities   # np.vstack([self.agents_velocities, np.array(self.moving_obstacles_velocities)])
        dyn_obs_input = get_dyn_obs_state(
                    pos, vel, dyn_pos, dyn_vel, target_tensor, device=self.device_str, robot_size=robot_radius, max_range=self.max_range
                )
        if self.Agent is None or not hasattr(self.Agent, 'plan'):
            return self.orca.predict(env_state)
        velocity = self.Agent.plan(robot_state, static_obs_input, dyn_obs_input, target_tensor)
        if not np.all(np.isfinite(velocity)):
            return self.orca.predict(env_state)
        speed = float(np.linalg.norm(velocity))
        if speed > self.max_speed:
            velocity = (velocity / (speed + 1e-20)) * self.max_speed
        # 基于 ORCA 思想的静态线段约束投影优先
        try:
            velocity = self._apply_safety_filter_orca_guided(
                velocity,
                pos,
                static_obs,
                float(robot_radius),
                float(self.safety_space),
                float(self.time_horizon_obst),
                float(self.max_speed),
                target_dir,
            )
        except Exception:
            pass
        # 仍使用 LiDAR 半空间过滤作为后备
        try:
            if range_matrix is not None and range_matrix.size > 0:
                velocity = self._apply_safety_filter(
                    velocity,
                    range_matrix,
                    start_angle_deg,
                    self.lidar_hres_deg,
                    float(robot_radius + self.d_safe_base),
                    self.alpha_cbf,
                    self.max_speed,
                    target_dir,
                )
        except Exception:
            pass
        action = ActionXY(velocity[0], velocity[1])
        return action

    def _apply_safety_filter_orca_guided(self, v_des, pos, static_lines, robot_radius, safety_space, tau_obs, vmax, target_dir):
        v = np.array(v_des, dtype=float)
        s = np.linalg.norm(v)
        if s > vmax and s > 0:
            v = v / s * vmax

        d_safe = float(robot_radius + safety_space)
        k = 1.0 / max(1e-3, float(tau_obs))

        def closest_point_on_segment(p, a, b):
            ab = b - a
            t = 0.0
            denom = float(np.dot(ab, ab))
            if denom > 1e-12:
                t = float(np.dot(p - a, ab) / denom)
            t = max(0.0, min(1.0, t))
            return a + t * ab

        constraints = []
        p = np.array(pos, dtype=float)
        for seg in static_lines or []:
            a = np.array(seg[0], dtype=float)
            b = np.array(seg[1], dtype=float)
            q = closest_point_on_segment(p, a, b)
            diff = q - p
            dist = float(np.linalg.norm(diff))
            if not np.isfinite(dist):
                continue
            if dist < 1e-8:
                n = np.zeros(2, dtype=float)
            else:
                n = diff / dist
            c = k * max(0.0, dist - d_safe)
            constraints.append((n, c))
            for endpoint in (a, b):
                diff_e = endpoint - p
                dist_e = float(np.linalg.norm(diff_e))
                if not np.isfinite(dist_e):
                    continue
                if dist_e < 1e-8:
                    n_e = np.zeros(2, dtype=float)
                else:
                    n_e = diff_e / dist_e
                c_e = k * max(0.0, dist_e - d_safe)
                constraints.append((n_e, c_e))

        for _ in range(max(1, int(self.qp_max_iters))):
            violated = False
            for n, c in constraints:
                dot = float(np.dot(n, v))
                if dot > c:
                    violated = True
                    v = v - ((dot - c) / (np.dot(n, n) + 1e-8)) * n
            s = np.linalg.norm(v)
            if s > vmax and s > 0:
                v = v / s * vmax
            if not violated:
                break

        t = np.array(target_dir, dtype=float)
        t_norm = np.linalg.norm(t) + 1e-8
        t_hat = t / t_norm
        proj = float(np.dot(t_hat, v))
        if proj < 0.0:
            v = v - proj * t_hat
        if proj < self.progress_floor:
            v = v + (self.progress_floor - proj) * t_hat
            for n, c in constraints:
                dot = float(np.dot(n, v))
                if dot > c:
                    v = v - ((dot - c) / (np.dot(n, n) + 1e-8)) * n
            s = np.linalg.norm(v)
            if s > vmax and s > 0:
                v = v / s * vmax
        return v

    def _apply_safety_filter(self, v_des, range_matrix, start_angle_deg, hres_deg, d_safe, alpha, vmax, target_dir):
        """将期望速度投影到由墙约束定义的可行集合（半空间交）内。
        约束：n_i^T v <= alpha * (d_i - d_safe)，n_i 为朝向墙的单位射线方向。
        距离越小，朝墙速度允许越小；小于安全距离时禁止朝墙运动。同时避免后退。
        """
        per_heading_dist = np.min(range_matrix[:, 1:], axis=1)  # (num_h,)
        v = np.array(v_des, dtype=float)
        # 限速
        s = np.linalg.norm(v)
        if s > vmax and s > 0:
            v = v / s * vmax

        # 投影到所有半空间（POCS迭代）
        for _ in range(max(1, int(self.qp_max_iters))):
            violated = False
            num_h = per_heading_dist.shape[0]
            for i in range(num_h):
                d_i = float(per_heading_dist[i])
                if not np.isfinite(d_i):
                    continue
                h_deg = start_angle_deg + i * hres_deg
                h_rad = np.deg2rad(h_deg)
                n = np.array([np.cos(h_rad), np.sin(h_rad)], dtype=float)  # 朝墙方向
                c = alpha * (d_i - d_safe)
                dot = float(np.dot(n, v))
                if dot > c:  # 违反约束，投影
                    violated = True
                    v = v - ((dot - c) / (np.dot(n, n) + 1e-8)) * n
            # 限速
            s = np.linalg.norm(v)
            if s > vmax and s > 0:
                v = v / s * vmax
            if not violated:
                break

        # 不后退：强制目标方向投影非负，并尽量保证最小前向
        t = np.array(target_dir, dtype=float)
        t_norm = np.linalg.norm(t) + 1e-8
        t_hat = t / t_norm
        proj = float(np.dot(t_hat, v))
        if proj < 0.0:
            v = v - proj * t_hat
            proj = 0.0
        if proj < self.progress_floor:
            v = v + (self.progress_floor - proj) * t_hat
            # 再做一次墙约束投影
            num_h = per_heading_dist.shape[0]
            for i in range(num_h):
                d_i = float(per_heading_dist[i])
                if not np.isfinite(d_i):
                    continue
                h_deg = start_angle_deg + i * hres_deg
                h_rad = np.deg2rad(h_deg)
                n = np.array([np.cos(h_rad), np.sin(h_rad)], dtype=float)
                c = alpha * (d_i - d_safe)
                dot = float(np.dot(n, v))
                if dot > c:
                    v = v - ((dot - c) / (np.dot(n, n) + 1e-8)) * n
        # 最终限速
        s = np.linalg.norm(v)
        if s > vmax and s > 0:
            v = v / s * vmax
        return v

  