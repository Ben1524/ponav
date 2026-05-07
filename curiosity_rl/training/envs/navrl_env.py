import os
import sys
import math
from typing import List, Tuple

import numpy as np
import torch
from tensordict.tensordict import TensorDict
from torchrl.data import CompositeSpec, UnboundedContinuousTensorSpec
from torchrl.envs.common import EnvBase


# Resolve quick-demos utils for observation builders
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

QD_DIR = os.path.join(ROOT_DIR, "sicnav/policy")
if QD_DIR not in sys.path:
    sys.path.append(QD_DIR)

import importlib.util as _ilu


def _load_module_from(path: str, name: str):
    spec = _ilu.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module {name} from {path}")
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_utils_mod = _load_module_from(os.path.join(QD_DIR, "utils.py"), "qd_utils")
get_robot_state = _utils_mod.get_robot_state
get_ray_cast = _utils_mod.get_ray_cast
get_dyn_obs_state = _utils_mod.get_dyn_obs_state


class NavRLEnvTorch(EnvBase):
    """A lightweight TorchRL env that mimics the nav_rl 2D world and emits
    observations matching quick-demos Agent/Student specs.

    Reward (simple shaping):
    - +1.0 * (prev_dist - new_dist)
    - -0.01 alive cost per step
    - -10.0 on collision, episode done
    - +5.0 on goal reach, episode done
    """

    def __init__(self, cfg):
        # config fallbacks
        device = getattr(cfg, "device", "cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device)
        self.device_str = "cuda" if self.device.type == "cuda" else "cpu"
        self.num_envs = int(getattr(getattr(cfg, "env", cfg), "num_envs", 8))
        self.dt = float(getattr(getattr(cfg, "env", cfg), "dt", 0.1))
        self.x_lim = [-20.0, 20.0]
        self.y_lim = [-20.0, 20.0]
        self.goal_reached_threshold = 0.2
        self.max_steps = int(getattr(cfg, "max_frame_num", 2000))

        super().__init__(device=self.device, batch_size=torch.Size([self.num_envs]))

        # Observation/Action specs (match quick-demos)
        observation_dim = 8
        self.observation_spec = CompositeSpec({
            "agents": CompositeSpec({
                "observation": CompositeSpec({
                    "state": UnboundedContinuousTensorSpec(torch.Size([observation_dim]), device=self.device),
                    "lidar": UnboundedContinuousTensorSpec(torch.Size([1, 36, 4]), device=self.device),
                    "direction": UnboundedContinuousTensorSpec(torch.Size([1, 3]), device=self.device),
                    "dynamic_obstacle": UnboundedContinuousTensorSpec(torch.Size([1, 5, 10]), device=self.device),
                }),
            }).expand(torch.Size([self.num_envs]))
        }, shape=torch.Size([self.num_envs]), device=self.device)

        action_dim = 3
        self.action_spec = CompositeSpec({
            "agents": CompositeSpec({
                "action": UnboundedContinuousTensorSpec(torch.Size([action_dim]), device=self.device),
            })
        }).expand(torch.Size([self.num_envs])).to(self.device)
        # Reward and Done specs
        from torchrl.data import DiscreteTensorSpec
        self.reward_spec = CompositeSpec({
            "agents": CompositeSpec({
                "reward": UnboundedContinuousTensorSpec(torch.Size([1]), device=self.device)
            })
        }).expand(torch.Size([self.num_envs])).to(self.device)
        self.done_spec = CompositeSpec({
            "done": DiscreteTensorSpec(2, torch.Size([1]), dtype=torch.bool),
            "terminated": DiscreteTensorSpec(2, torch.Size([1]), dtype=torch.bool),
            "truncated": DiscreteTensorSpec(2, torch.Size([1]), dtype=torch.bool),
        }).expand(torch.Size([self.num_envs])).to(self.device)

        # State buffers
        self._positions = np.zeros((self.num_envs, 2), dtype=np.float32)
        self._velocities = np.zeros((self.num_envs, 2), dtype=np.float32)
        self._goals = np.zeros((self.num_envs, 2), dtype=np.float32)
        self._prev_dist = np.zeros((self.num_envs,), dtype=np.float32)
        self._moving_positions = []  # list of arrays per step, shared across envs for simplicity
        self._moving_velocities = []
        self._moving_shapes = []
        self._steps = np.zeros((self.num_envs,), dtype=np.int32)

        # Static obstacles: (cx, cy, a, b)
        self._static_cuboids = [
            (0.0, 0.0, 0.5, 3.0),
            (3.0, 3.0, 2.0, 0.5),
        ]

    def _set_seed(self, seed: int):
        torch.manual_seed(seed)
        np.random.seed(seed)
        return seed

    def _reset(self, tensordict=None):
        # Initialize robots across batch
        xs = np.linspace(self.x_lim[0] + 1.0, self.x_lim[0] + 3.0, self.num_envs)
        positions = []
        goals = []
        for x in xs:
            start = np.array([x, np.random.uniform(self.y_lim[0] + 1.0, self.y_lim[1] - 1.0)], dtype=np.float32)
            goal = np.array([
                np.random.uniform(self.x_lim[1] - 3.0, self.x_lim[1] - 1.0),
                np.random.uniform(self.y_lim[0] + 1.0, self.y_lim[1] - 1.0)
            ], dtype=np.float32)
            positions.append(start)
            goals.append(goal)
        self._positions = np.stack(positions)
        self._velocities = np.zeros_like(self._positions)
        self._goals = np.stack(goals)
        self._prev_dist = np.linalg.norm(self._goals - self._positions, axis=-1)
        self._steps = np.zeros((self.num_envs,), dtype=np.int32)

        # Moving obstacles (shared set)
        self._moving_positions, self._moving_velocities, self._moving_shapes = self._reset_moving()

        obs = self._compute_observation()
        td = TensorDict(obs, batch_size=self.batch_size, device=self.device)
        td.set("done", torch.zeros(self.batch_size + torch.Size([1]), dtype=torch.bool, device=self.device))
        td.set("terminated", torch.zeros(self.batch_size + torch.Size([1]), dtype=torch.bool, device=self.device))
        td.set("truncated", torch.zeros(self.batch_size + torch.Size([1]), dtype=torch.bool, device=self.device))
        return td

    def _step(self, tensordict):
        action = tensordict.get(("agents", "action"))  # shape [num_envs, 1, 3] or [num_envs, 3]
        if action.dim() == 3:
            action = action[:, 0, :]  # [num_envs, 3]
        v = action[..., :2].detach().cpu().numpy()
        # clamp speed
        speeds = np.linalg.norm(v, axis=-1, keepdims=True)
        v = np.where(speeds > 2.0, v / (speeds + 1e-8) * 2.0, v)

        # advance moving obstacles
        self._update_moving()
        circles = self._obstacles_as_circles()

        rewards = np.zeros((self.num_envs,), dtype=np.float32)
        done = np.zeros((self.num_envs,), dtype=bool)

        # compute next pos with simple collision check
        next_pos = self._positions + v * self.dt
        # bound clamp
        next_pos[:, 0] = np.clip(next_pos[:, 0], self.x_lim[0], self.x_lim[1])
        next_pos[:, 1] = np.clip(next_pos[:, 1], self.y_lim[0], self.y_lim[1])

        # distances & reward shaping
        dist = np.linalg.norm(self._goals - next_pos, axis=-1)
        delta = self._prev_dist - dist
        rewards += delta * 1.0
        rewards -= 0.01

        # collision penalty & termination
        collided = np.array([self._check_collision(next_pos[i], circles) for i in range(self.num_envs)], dtype=bool)
        rewards[collided] -= 10.0
        done = np.logical_or(done, collided)

        # reach goal bonus & termination
        reached = dist < self.goal_reached_threshold
        rewards[reached] += 5.0
        done = np.logical_or(done, reached)

        # update state
        self._positions = next_pos
        self._velocities = v
        self._prev_dist = dist
        self._steps = self._steps + 1
        timeout = self._steps >= self.max_steps
        done = np.logical_or(done, timeout)

        obs = self._compute_observation()
        td = TensorDict(obs, batch_size=self.batch_size, device=self.device)
        td.set(("agents", "reward"), torch.tensor(rewards, device=self.device).unsqueeze(-1))
        td.set("done", torch.tensor(done, device=self.device).unsqueeze(-1))
        td.set("terminated", torch.tensor(done, device=self.device).unsqueeze(-1))
        td.set("truncated", torch.tensor(timeout, device=self.device).unsqueeze(-1))
        return td

    # ---- Helpers ----
    def _reset_moving(self, num=8):
        mpos, mvel, mshape = [], [], []
        for _ in range(num):
            pos = np.array([
                np.random.uniform(self.x_lim[0] + 2.0, self.x_lim[1] - 2.0),
                np.random.uniform(self.y_lim[0] + 2.0, self.y_lim[1] - 2.0),
            ], dtype=np.float32)
            ang = np.random.uniform(0, 2 * math.pi)
            spd = np.random.uniform(0.9, 1.2)
            vel = np.array([spd * math.cos(ang), spd * math.sin(ang)], dtype=np.float32)
            mpos.append(pos)
            mvel.append(vel)
            mshape.append('ellipse' if np.random.rand() > 0.5 else 'cuboid')
        return mpos, mvel, mshape

    def _update_moving(self):
        for i in range(len(self._moving_positions)):
            self._moving_positions[i] = self._moving_positions[i] + self._moving_velocities[i] * self.dt
            for d in range(2):
                if d == 0:
                    if self._moving_positions[i][d] < self.x_lim[0] + 1:
                        self._moving_positions[i][d] = self.x_lim[0] + 1
                        self._moving_velocities[i][d] *= -1
                    elif self._moving_positions[i][d] > self.x_lim[1] - 1:
                        self._moving_positions[i][d] = self.x_lim[1] - 1
                        self._moving_velocities[i][d] *= -1
                else:
                    if self._moving_positions[i][d] < self.y_lim[0] + 1:
                        self._moving_positions[i][d] = self.y_lim[0] + 1
                        self._moving_velocities[i][d] *= -1
                    elif self._moving_positions[i][d] > self.y_lim[1] - 1:
                        self._moving_positions[i][d] = self.y_lim[1] - 1
                        self._moving_velocities[i][d] *= -1

    def _obstacles_as_circles(self) -> List[Tuple[float, float, float]]:
        circles: List[Tuple[float, float, float]] = []
        for cx, cy, a, b in self._static_cuboids:
            r = 0.5 * math.sqrt(a * a + b * b)
            circles.append((float(cx), float(cy), float(r)))
        for i, center in enumerate(self._moving_positions):
            cx, cy = float(center[0]), float(center[1])
            if self._moving_shapes[i] == 'ellipse':
                axes = (0.20, 0.6)
                r = 0.5 * max(axes)
            else:
                a = 0.7
                b = 0.7
                r = 0.5 * math.sqrt(a * a + b * b)
            circles.append((cx, cy, float(r)))
        return circles

    def _check_collision(self, pos: np.ndarray, circles: List[Tuple[float, float, float]]) -> bool:
        for ox, oy, r in circles:
            if np.hypot(pos[0] - ox, pos[1] - oy) <= r + 0.2:
                return True
        return False

    def _compute_observation(self):
        # Build observations for each env using quick-demos utils
        num_h = 36
        obs_list = []
        for i in range(self.num_envs):
            pos = self._positions[i]
            vel = self._velocities[i]
            goal = self._goals[i]
            target_dir = goal - pos
            robot_state = get_robot_state(pos, goal, vel, target_dir, device=self.device_str)
            # utils returns [1, 8]; env expects [8]
            robot_state = robot_state.squeeze(0)
            static_obs_input, _, _ = get_ray_cast(
                pos,
                self._obstacles_as_circles(),
                max_range=4.0,
                hres_deg=10.0,
                vfov_angles_deg=[-10.0, 0.0, 10.0, 20.0],
                start_angle_deg=np.degrees(np.arctan2(target_dir[1], target_dir[0])),
                device=self.device_str,
            )
            # utils returns [1,1,H,V]; env expects [1,H,V]
            static_obs_input = static_obs_input.squeeze(0)
            target_tensor = torch.tensor(
                np.append(target_dir[:2], 0.0), dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            dyn_obs_input = get_dyn_obs_state(
                pos,
                vel,
                np.vstack([self._positions, np.array(self._moving_positions, dtype=np.float32)]),
                np.vstack([self._velocities, np.array(self._moving_velocities, dtype=np.float32)]),
                target_tensor.unsqueeze(0),
                device=self.device_str,
            )
            # utils returns [1,1,N,10]; env expects [1,N,10]
            dyn_obs_input = dyn_obs_input.squeeze(0)
            obs_list.append({
                "state": robot_state,
                "lidar": static_obs_input,
                "direction": target_tensor,
                "dynamic_obstacle": dyn_obs_input,
            })

        # Stack batch into TensorDict compatible structure
        state = torch.stack([o["state"] for o in obs_list], dim=0).to(self.device)
        lidar = torch.stack([o["lidar"] for o in obs_list], dim=0).to(self.device)
        direction = torch.stack([o["direction"] for o in obs_list], dim=0).to(self.device)
        dyn = torch.stack([o["dynamic_obstacle"] for o in obs_list], dim=0).to(self.device)

        return {
            ("agents", "observation", "state"): state,
            ("agents", "observation", "lidar"): lidar,
            ("agents", "observation", "direction"): direction,
            ("agents", "observation", "dynamic_obstacle"): dyn,
        }
