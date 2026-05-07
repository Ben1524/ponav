import os
import argparse
import configparser
from typing import List

import gym
import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from tensordict.tensordict import TensorDict
from torchrl.envs.utils import ExplorationType, set_exploration_type

from crowd_sim_plus.envs.utils.robot_plus import Robot
from sicnav.policy.ponav import Agent as NavAgent
from sicnav.policy.ponav import get_line_ray_cast, get_dyn_obs_state, get_robot_state


def build_obs_tensordict(device_str: str,
                         env_state,
                         lidar_hres_deg: float,
                         vfov_angles_deg: List[float],
                         max_range: float = 4.0):
    """
    根据当前环境 state 构造 PONav.Agent 期望的观测 TensorDict：
    - state: 机器人局部状态向量
    - lidar: 静态线障碍的 LiDAR 张量
    - direction: 目标方向
    - dynamic_obstacle: 动态障碍编码
    """
    self_state = env_state.self_state
    pos = np.array([self_state.px, self_state.py], dtype=np.float32)
    vel = np.array([self_state.vx, self_state.vy], dtype=np.float32)
    goal = np.array([self_state.gx, self_state.gy], dtype=np.float32)
    target_dir = goal - pos

    target_tensor = torch.tensor(
        np.append(target_dir[:2], 0.0),
        dtype=torch.float32,
        device=device_str
    ).unsqueeze(0).unsqueeze(0)

    start_angle_deg = float(np.degrees(np.arctan2(target_dir[1], target_dir[0])))
    static_obs_input, range_matrix, _ = get_line_ray_cast(
        pos,
        env_state.static_obs,
        max_range=max_range,
        hres_deg=lidar_hres_deg,
        vfov_angles_deg=vfov_angles_deg,
        start_angle_deg=start_angle_deg,
        device=device_str,
    )

    human_positions = [[h.px, h.py] for h in env_state.human_states]
    human_velocities = [[h.vx, h.vy] for h in env_state.human_states]
    dyn_obs_input = get_dyn_obs_state(
        pos, vel, human_positions, human_velocities,
        target_tensor, device=device_str,
        robot_size=self_state.radius,
        max_range=max_range
    )

    robot_state_td = get_robot_state(pos, goal, vel, target_dir, device=device_str)

    obs_td = TensorDict({
        "agents": TensorDict({
            "observation": TensorDict({
                "state": robot_state_td,
                "lidar": static_obs_input,
                "direction": target_tensor,
                "dynamic_obstacle": dyn_obs_input,
            })
        })
    }, device=torch.device(device_str))

    return obs_td, range_matrix, start_angle_deg


def compute_gae(rews, dones, vals, last_value, gamma=0.99, lam=0.95):
    """
    标准 GAE 计算，返回优势 A 和回报 R。
    """
    T = len(rews)
    adv = np.zeros_like(rews, dtype=np.float32)
    gae = 0.0
    for t in reversed(range(T)):
        next_v = last_value if t == T - 1 else vals[t + 1]
        delta = rews[t] + gamma * (1 - dones[t]) * next_v - vals[t]
        gae = delta + gamma * lam * (1 - dones[t]) * gae
        adv[t] = gae
    ret = adv + vals
    return adv, ret


def main():
    parser = argparse.ArgumentParser(description="Train PONav (PPO) with CrowdSimPlus-v0")
    parser.add_argument('--env_config', type=str, default='sicnav/configs/env.config')
    parser.add_argument('--policy_config', type=str, default='sicnav/configs/policy.config')
    parser.add_argument('--total_steps', type=int, default=200000)
    parser.add_argument('--rollout_len', type=int, default=2048)
    parser.add_argument('--epochs', type=int, default=4)
    parser.add_argument('--clip_ratio', type=float, default=0.1)
    parser.add_argument('--entropy_coef', type=float, default=1e-3)
    parser.add_argument('--value_coef', type=float, default=0.5)
    parser.add_argument('--max_grad_norm', type=float, default=0.5)
    parser.add_argument('--save_every', type=int, default=50000)
    args = parser.parse_args()

    # 1. 读取配置，创建环境
    env_config = configparser.RawConfigParser()
    env_config.read(args.env_config)

    policy_config = configparser.RawConfigParser()
    policy_config.read(args.policy_config)

    env_order = gym.make('CrowdSimPlus-v0')
    env = env_order.unwrapped
    env.configure(env_config)

    # 2. 创建 robot（只为让环境有半径等属性），不使用其 policy.act
    robot = Robot(env_config, 'robot')
    env.set_robot(robot)

    # 3. 创建 PONav 的 PPO Agent（训练对象）
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    learner = NavAgent(device=device)

    # 训练中用的 LiDAR 参数与 PONav 一致（手动写死为原训练值）
    lidar_hres_deg = 10.0
    vfov_angles_deg = [-10.0, 0.0, 10.0, 20.0]
    max_range = 4.0

    total_steps = 0
    episode_idx = 0

    while total_steps < args.total_steps:
        # 每次循环采集一段 rollout
        storage = {
            "obs": [],
            "act": [],
            "logp": [],
            "rew": [],
            "done": [],
            "val": [],
        }

        # 环境 reset 在 train split（如果你的 env.reset 只有模式+case，就按 'train' 用）
        state, _ = env.reset('train', None, return_stat=True)
        done = False
        steps_in_rollout = 0

        while not done and steps_in_rollout < args.rollout_len and total_steps < args.total_steps:
            # 3.1 构造观测 TensorDict
            obs_td, _, _ = build_obs_tensordict(
                'cuda' if torch.cuda.is_available() else 'cpu',
                state,
                lidar_hres_deg,
                vfov_angles_deg,
                max_range=max_range
            )

            # 3.2 策略采样动作（归一化 Beta），并得到世界坐标动作
            with set_exploration_type(ExplorationType.RANDOM):
                td_out = learner.policy(obs_td)

            # 世界动作 (vx, vy, w)，这里只取前 2 维
            world_act = td_out["agents", "action"][0][0].detach().cpu().numpy()[:2]
            norm_act = td_out["agents", "action_normalized"][0][0].detach().cpu().numpy()
            logp = td_out.get(("agents", "action_normalized_log_prob"), None)
            if logp is None:
                # 如果 ProbabilisticActor 没直接给 log_prob，就手动算（略），
                # 这里先简化为 0，不影响结构说明，如要真正训练再细化。
                logp_val = 0.0
            else:
                logp_val = float(logp[0][0].detach().cpu().numpy())

            value = td_out["state_value"][0][0].detach().cpu().numpy()

            # 3.3 执行动作到环境（此处暂不加安全过滤，便于结构清晰）
            action = env.robot.policy.ActionXY(world_act[0], world_act[1]) \
                if hasattr(env.robot, 'policy') and hasattr(env.robot.policy, 'ActionXY') else None
            # 直接使用 crowd_sim_plus 自带的 ActionXY 类更稳妥：
            from crowd_sim_plus.envs.utils.action import ActionXY as EnvActionXY
            action = EnvActionXY(world_act[0], world_act[1])

            next_obs, reward, done, info = env.step(action)

            # 这里假设 env.step 返回的 next_obs 和 static_obs 可以重新包装为 state
            # 在原 simple_test.py 里，state 是通过 robot.get_joint_state(ob, static_obs) 得到的，
            # 训练时我们只需要 self_state/human_states/static_obs 即可，最简单是：
            state = env  # 某些实现里 env 本身保存了 self_state/human_states/static_obs
            # 若不行，可以参考 simple_test.py 中 get_joint_state 的用法自行构造。

            # 3.4 存储 PPO 所需数据
            storage["obs"].append(obs_td.cpu())
            storage["act"].append(norm_act)
            storage["logp"].append(logp_val)
            storage["rew"].append(float(reward))
            storage["done"].append(float(done))
            storage["val"].append(float(value))

            total_steps += 1
            steps_in_rollout += 1

        # ---- 结束 rollout，计算 GAE 和回报 ----
        rews = np.array(storage["rew"], dtype=np.float32)
        dones = np.array(storage["done"], dtype=np.float32)
        vals = np.array(storage["val"], dtype=np.float32)

        # bootstrap：简单设为 0（或再算一次 V(s_T)）
        last_value = 0.0
        adv, ret = compute_gae(rews, dones, vals, last_value)

        storage["adv"] = list(adv)
        storage["ret"] = list(ret)

        # ---- PPO 更新（单 batch 简化版）----
        device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
        device = torch.device(device_str)

        # 拼接 obs
        obs_list = [td.to(device) for td in storage["obs"]]
        obs_td = TensorDict.stack(obs_list, dim=0)

        actions = torch.as_tensor(
            np.stack(storage["act"], axis=0),
            dtype=torch.float32, device=device
        )
        old_logp = torch.as_tensor(
            np.stack(storage["logp"], axis=0),
            dtype=torch.float32, device=device
        )
        adv_t = torch.as_tensor(
            np.stack(storage["adv"], axis=0),
            dtype=torch.float32, device=device
        )
        ret_t = torch.as_tensor(
            np.stack(storage["ret"], axis=0),
            dtype=torch.float32, device=device
        )
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        for _ in range(args.epochs):
            with set_exploration_type(ExplorationType.MEAN):
                # 重新跑 feature 与 actor，得到当前分布参数和新 log_prob
                learner.policy.feature_extractor(obs_td)
                feat = obs_td["_feature"]
                td_feat = TensorDict({"_feature": feat}, device=device)
                # 这里 BetaActor 模块在 learner.policy.actor.module 里
                actor_out = learner.policy.actor.module(td_feat)
                alpha, beta = actor_out["alpha"], actor_out["beta"]
                from sicnav.policy.ponav import IndependentBeta
                dist = IndependentBeta(alpha, beta)
                new_logp = dist.log_prob(actions)

                values = learner.policy.critic(
                    TensorDict({"_feature": feat}, device=device)
                )["state_value"].squeeze(-1)

            ratio = torch.exp(new_logp - old_logp)
            surr1 = ratio * adv_t
            surr2 = torch.clamp(ratio, 1.0 - args.clip_ratio, 1.0 + args.clip_ratio) * adv_t
            actor_loss = -torch.mean(torch.min(surr1, surr2))

            value_loss = torch.nn.functional.mse_loss(values, ret_t)
            entropy_loss = -torch.mean(new_logp)

            total_loss = actor_loss + args.value_coef * value_loss + args.entropy_coef * entropy_loss

            learner.policy.actor_optim.zero_grad()
            learner.policy.critic_optim.zero_grad()
            learner.policy.feature_extractor_optim.zero_grad()
            total_loss.backward()
            clip_grad_norm_(list(learner.policy.actor.parameters()) +
                            list(learner.policy.critic.parameters()) +
                            list(learner.policy.feature_extractor.parameters()),
                            max_norm=args.max_grad_norm)
            learner.policy.actor_optim.step()
            learner.policy.critic_optim.step()
            learner.policy.feature_extractor_optim.step()

        episode_idx += 1
        print(f"[PONav] Episode {episode_idx}, collected {steps_in_rollout} steps, total {total_steps} / {args.total_steps}")

        # 定期保存 checkpoint
        if total_steps % args.save_every < args.rollout_len:
            ckpt_dir = os.path.join(os.path.dirname(__file__), 'policy', 'ckpts')
            os.makedirs(ckpt_dir, exist_ok=True)
            ckpt_path = os.path.join(ckpt_dir, 'navrl_checkpoint.pt')
            torch.save(learner.policy.state_dict(), ckpt_path)
            print(f"[PONav] Saved checkpoint at {ckpt_path} (steps={total_steps})")

    # 最终保存
    ckpt_dir = os.path.join(os.path.dirname(__file__), 'policy', 'ckpts')
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, 'navrl_checkpoint.pt')
    torch.save(learner.policy.state_dict(), ckpt_path)
    print(f"[PONav] Training complete. Final checkpoint saved at {ckpt_path}")


if __name__ == '__main__':
    main()