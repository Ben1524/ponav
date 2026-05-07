import logging
import math
import os
from collections import deque
from typing import Deque, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from crowd_sim_plus.envs.policy.policy import Policy
from crowd_sim_plus.envs.utils.action import ActionRot


def log_normal_density(x, mean, log_std, std):
    var = std.pow(2)
    log_density = -(x - mean).pow(2) / (2 * var) - 0.5 * math.log(2 * math.pi) - log_std
    return log_density.sum(-1, keepdim=True)


class CNNPolicy(nn.Module):
    def __init__(self, frames: int, action_space: int) -> None:
        super().__init__()
        self.logstd = nn.Parameter(torch.zeros(action_space))

        self.act_fea_cv1 = nn.Conv1d(frames, 32, kernel_size=5, stride=2, padding=1)
        self.act_fea_cv2 = nn.Conv1d(32, 32, kernel_size=3, stride=2, padding=1)
        self.act_fc1 = nn.Linear(128 * 32, 256)
        self.act_fc2 = nn.Linear(256 + 2 + 2, 128)
        self.actor1 = nn.Linear(128, 1)
        self.actor2 = nn.Linear(128, 1)

        self.crt_fea_cv1 = nn.Conv1d(frames, 32, kernel_size=5, stride=2, padding=1)
        self.crt_fea_cv2 = nn.Conv1d(32, 32, kernel_size=3, stride=2, padding=1)
        self.crt_fc1 = nn.Linear(128 * 32, 256)
        self.crt_fc2 = nn.Linear(256 + 2 + 2, 128)
        self.critic = nn.Linear(128, 1)

    def forward(self, x, goal, speed):
        a = F.relu(self.act_fea_cv1(x))
        a = F.relu(self.act_fea_cv2(a))
        a = a.view(a.shape[0], -1)
        a = F.relu(self.act_fc1(a))
        a = torch.cat((a, goal, speed), dim=-1)
        a = F.relu(self.act_fc2(a))
        mean1 = torch.sigmoid(self.actor1(a))
        mean2 = torch.tanh(self.actor2(a))
        mean = torch.cat((mean1, mean2), dim=-1)

        logstd = self.logstd.expand_as(mean)
        std = torch.exp(logstd)
        action = torch.normal(mean, std)
        logprob = log_normal_density(action, mean, std=std, log_std=logstd)

        v = F.relu(self.crt_fea_cv1(x))
        v = F.relu(self.crt_fea_cv2(v))
        v = v.view(v.shape[0], -1)
        v = F.relu(self.crt_fc1(v))
        v = torch.cat((v, goal, speed), dim=-1)
        v = F.relu(self.crt_fc2(v))
        v = self.critic(v)

        return v, action, logprob, mean


class CNNRL(Policy):
    def __init__(self) -> None:
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: Optional[CNNPolicy] = None
        self.kinematics = 'unicycle'
        self.name = 'cnn_policy'
        self.multiagent_training = False

        self.laser_beam = 512
        self.laser_hist = 6
        self.max_range = 3.0
        self.action_linear = (-0.5, 1.0)
        self.action_angular = (-1.0, 1.0)
        self.stochastic_inference = False

        self.obs_history: Deque[np.ndarray] = deque(maxlen=self.laser_hist)
        self.angles: Optional[np.ndarray] = None
        self.checkpoint_path: Optional[str] = None
        self.prev_heading: Optional[float] = None

    def configure(self, config) -> None:
        section = 'cnn_policy'
        self.laser_beam = config.getint(section, 'laser_beam', fallback=self.laser_beam)
        self.laser_hist = config.getint(section, 'laser_hist', fallback=self.laser_hist)
        self.max_range = config.getfloat(section, 'max_range', fallback=self.max_range)
        lin_min = config.getfloat(section, 'linear_min', fallback=self.action_linear[0])
        lin_max = config.getfloat(section, 'linear_max', fallback=self.action_linear[1])
        ang_min = config.getfloat(section, 'angular_min', fallback=self.action_angular[0])
        ang_max = config.getfloat(section, 'angular_max', fallback=self.action_angular[1])
        self.action_linear = (lin_min, lin_max)
        self.action_angular = (ang_min, ang_max)
        self.stochastic_inference = config.getboolean(section, 'stochastic', fallback=False)

        # raise FileNotFoundError(f'CNN policy checkpoint {checkpoint} not found.')
        self.checkpoint_path = "/home/zhujingqi/MultiAgent/CrowdNavigationMPC/sicnav/policy/cnn_rl/stage2_6_2320.pth"

        self.model = CNNPolicy(self.laser_hist, 2).to(self.device)
        state_dict = torch.load(self.checkpoint_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        logging.info('Loaded CNN policy weights from %s', self.checkpoint_path)

        self.obs_history = deque(maxlen=self.laser_hist)
        self.angles = np.linspace(-math.pi, math.pi, self.laser_beam, endpoint=False).astype(np.float32)
        self.prev_heading = None

    def predict(self, state):
        if self.model is None:
            raise RuntimeError('CNN policy not configured')
        if self.time_step is None:
            raise RuntimeError('Policy time_step must be set before prediction')

        laser = self._build_laser_scan(state)
        episode_reset = self._episode_just_reset()
        if episode_reset or len(self.obs_history) < self.laser_hist:
            self.obs_history.clear()
            for _ in range(self.laser_hist):
                self.obs_history.append(laser)
        else:
            self.obs_history.append(laser)

        if episode_reset:
            self.prev_heading = state.self_state.theta

        obs_stack = np.stack(self.obs_history, axis=0)
        goal_vec, speed_vec = self._goal_and_speed(state)

        obs_tensor = torch.tensor(obs_stack, dtype=torch.float32, device=self.device).unsqueeze(0)
        goal_tensor = torch.tensor(goal_vec, dtype=torch.float32, device=self.device).unsqueeze(0)
        speed_tensor = torch.tensor(speed_vec, dtype=torch.float32, device=self.device).unsqueeze(0)

        with torch.no_grad():
            _, sampled_action, _, mean = self.model(obs_tensor, goal_tensor, speed_tensor)

        chosen = sampled_action if self.stochastic_inference else mean
        action = chosen.squeeze(0).cpu().numpy()
        linear_cmd = self._scale_linear(action[0])
        angular_cmd = self._scale_angular(action[1])

        return ActionRot(linear_cmd, angular_cmd * self.time_step)

    def _episode_just_reset(self) -> bool:
        if self.env is None:
            return False
        return getattr(self.env, 'global_time', 0.0) <= (self.time_step or 0.0) + 1e-3

    def _build_laser_scan(self, state) -> np.ndarray:
        """Approximate the legacy StageWorld lidar using current simulator geometry."""
        origin = np.array([state.self_state.px, state.self_state.py], dtype=np.float32)
        theta = state.self_state.theta
        humans = state.human_states
        static_obs = state.static_obs if state.static_obs is not None else []
        angles = self.angles if self.angles is not None else np.linspace(-math.pi, math.pi, self.laser_beam, endpoint=False)
        scan = np.full(self.laser_beam, self.max_range, dtype=np.float32)

        for i, rel_angle in enumerate(angles):
            direction = np.array([math.cos(theta + rel_angle), math.sin(theta + rel_angle)], dtype=np.float32)
            direction /= np.linalg.norm(direction) + 1e-8
            min_dist = self.max_range

            for human in humans:
                center = np.array([human.px, human.py], dtype=np.float32)
                hit = self._ray_circle(origin, direction, center, human.radius)
                if hit is not None:
                    min_dist = min(min_dist, hit)

            for obstacle in static_obs:
                hit = None
                if hasattr(obstacle, 'px') and hasattr(obstacle, 'radius'):
                    center = np.array([obstacle.px, obstacle.py], dtype=np.float32)
                    hit = self._ray_circle(origin, direction, center, getattr(obstacle, 'radius', 0.1))
                elif hasattr(obstacle, '__len__') and len(obstacle) == 2:
                    p1 = np.array(obstacle[0], dtype=np.float32)
                    p2 = np.array(obstacle[1], dtype=np.float32)
                    hit = self._ray_segment(origin, direction, p1, p2)
                if hit is not None:
                    min_dist = min(min_dist, hit)

            scan[i] = min_dist

        scan = np.clip(scan, 0.0, self.max_range)
        normalized = scan / max(self.max_range, 1e-6) - 0.5
        return normalized.astype(np.float32)

    @staticmethod
    def _ray_circle(origin, direction, center, radius):
        m = origin - center
        b = np.dot(m, direction)
        c = np.dot(m, m) - radius * radius
        if c > 0.0 and b > 0.0:
            return None
        discr = b * b - c
        if discr < 0.0:
            return None
        t = -b - math.sqrt(discr)
        return t if t >= 0.0 else None

    @staticmethod
    def _ray_segment(origin, direction, p1, p2):
        v1 = origin - p1
        v2 = p2 - p1
        denom = direction[0] * (-v2[1]) - direction[1] * (-v2[0])
        if abs(denom) < 1e-8:
            return None
        t = (v1[0] * (-v2[1]) - v1[1] * (-v2[0])) / denom
        u = (direction[0] * v1[1] - direction[1] * v1[0]) / denom
        if t >= 0.0 and 0.0 <= u <= 1.0:
            return t
        return None

    def _goal_and_speed(self, state):
        robot = state.self_state
        rel_goal = np.array([robot.gx - robot.px, robot.gy - robot.py], dtype=np.float32)
        rotation = np.array([
            [math.cos(robot.theta), math.sin(robot.theta)],
            [-math.sin(robot.theta), math.cos(robot.theta)],
        ], dtype=np.float32)
        goal_body = rotation @ rel_goal
        vel_world = np.array([robot.vx, robot.vy], dtype=np.float32)
        vel_body = rotation @ vel_world
        linear_speed = float(vel_body[0])
        if getattr(robot, 'omega', None) is not None:
            angular_speed = float(robot.omega)
            self.prev_heading = robot.theta
        else:
            angular_speed = self._estimate_angular(robot.theta)
        speed_vec = np.array([linear_speed, angular_speed], dtype=np.float32)
        return goal_body.astype(np.float32), speed_vec

    def _estimate_angular(self, heading: float) -> float:
        if self.time_step is None:
            self.prev_heading = heading
            return 0.0
        if self.prev_heading is None:
            self.prev_heading = heading
            return 0.0
        delta = self._wrap_angle(heading - self.prev_heading)
        self.prev_heading = heading
        return delta / self.time_step

    @staticmethod
    def _wrap_angle(angle: float) -> float:
        return (angle + math.pi) % (2 * math.pi) - math.pi

    def _scale_linear(self, value: float) -> float:
        v = np.clip(value, 0.0, 1.0)
        v_min, v_max = self.action_linear
        return v_min + (v_max - v_min) * v

    def _scale_angular(self, value: float) -> float:
        w = np.clip(value, -1.0, 1.0)
        w_min, w_max = self.action_angular
        return w_min + (w + 1.0) * 0.5 * (w_max - w_min)
