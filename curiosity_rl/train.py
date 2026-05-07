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
# Import policy factory to register 'ponav' policy
import sicnav.policy.policy_factory
from crowd_sim_plus.envs.policy.policy_factory import policy_factory
from sicnav.policy.ponav import PONav
# Ensure ponav is registered
if 'ponav' not in policy_factory:
    policy_factory['ponav'] = PONav

from sicnav.policy.ponav import Agent as NavAgent
from sicnav.policy.ponav import get_line_ray_cast, get_dyn_obs_state, get_robot_state
from curiosity_rl.curiosity import HICM

def build_obs_tensordict(device_str: str,
                         env_state,
                         lidar_hres_deg: float,
                         vfov_angles_deg: List[float],
                         max_range: float = 4.0):
    """
    Construct observation TensorDict for PONav.Agent
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

    return obs_td

def extract_hicm_state(obs_td):
    robot_state = obs_td["agents", "observation", "state"]
    lidar = obs_td["agents", "observation", "lidar"]
    dyn_obs = obs_td["agents", "observation", "dynamic_obstacle"]
    return (robot_state, lidar, dyn_obs)

def compute_gae(rews, dones, vals, last_value, gamma=0.99, lam=0.95):
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
    parser = argparse.ArgumentParser(description="Train PONav (PPO) with HICM Curiosity")
    parser.add_argument('--env_config', type=str, default='./configs/env.config')
    parser.add_argument('--policy_config', type=str, default='./configs/policy.config')
    parser.add_argument('--total_steps', type=int, default=200000)
    parser.add_argument('--rollout_len', type=int, default=2048)
    parser.add_argument('--epochs', type=int, default=4)
    parser.add_argument('--clip_ratio', type=float, default=0.1)
    parser.add_argument('--entropy_coef', type=float, default=1e-3)
    parser.add_argument('--value_coef', type=float, default=0.5)
    parser.add_argument('--max_grad_norm', type=float, default=0.5)
    parser.add_argument('--save_every', type=int, default=50000)
    
    # Curiosity params
    parser.add_argument('--curiosity_lr', type=float, default=1e-4)
    parser.add_argument('--curiosity_scale', type=float, default=0.01)
    
    args = parser.parse_args()

    # 1. Config & Env
    env_config = configparser.RawConfigParser()
    env_config.read(args.env_config)
    policy_config = configparser.RawConfigParser()
    policy_config.read(args.policy_config)

    env_order = gym.make('CrowdSimPlus-v0')
    env = env_order.unwrapped
    env.configure(env_config)

    robot = Robot(env_config, 'robot')
    env.set_robot(robot)

    # 2. Agent & Curiosity
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    learner = NavAgent(device=device)

    # HICM Initialization
    # Need to infer shapes from a dummy observation
    lidar_hres_deg = 10.0
    vfov_angles_deg = [-10.0, 0.0, 10.0, 20.0]
    max_range = 4.0
    
    # Dummy reset to get shapes
    ob, static_obs = env.reset('train', None, return_stat=True)
    state = robot.get_joint_state(ob, static_obs)
    dummy_obs_td = build_obs_tensordict(str(device), state, lidar_hres_deg, vfov_angles_deg, max_range)
    
    robot_state_dim = dummy_obs_td["agents", "observation", "state"].shape[-1]
    static_obs_shape = dummy_obs_td["agents", "observation", "lidar"].shape[1:] # (1, 36, 4) -> (1, 36, 4)
    dyn_obs_shape = dummy_obs_td["agents", "observation", "dynamic_obstacle"].shape[1:] # (1, 5, 10) -> (1, 5, 10)

    # Config object for HICM (needs action_space info if discrete, but we use continuous)
    class DummyConfig:
        class action_space:
            speed_samples = 1
            rotation_samples = 1
    
    hicm = HICM(
        config=DummyConfig(),
        robot_state_dim=robot_state_dim,
        dyn_obs_shape=dyn_obs_shape,
        static_obs_shape=static_obs_shape,
        device=device,
        lr=args.curiosity_lr,
        scaling_factor=args.curiosity_scale,
        policy="ponav",
        continuous_action=True,
        action_dim=2 # vx, vy
    )

    total_steps = 0
    episode_idx = 0

    while total_steps < args.total_steps:
        storage = {
            "obs": [],
            "act": [],
            "logp": [],
            "rew": [],
            "done": [],
            "val": [],
            "hicm_state": [], # Store tuple for HICM
            "hicm_next_state": [],
            "hicm_action": []
        }

        state, _ = env.reset('train', None, return_stat=True)
        # Construct JointState for PONav
        state = robot.get_joint_state(state, env.static_obstacles)

        done = False
        steps_in_rollout = 0
        episode_reward = 0
        episode_intrinsic_reward = 0

        # Initial observation
        obs_td = build_obs_tensordict(str(device), state, lidar_hres_deg, vfov_angles_deg, max_range)
        hicm_state = extract_hicm_state(obs_td)

        while not done and steps_in_rollout < args.rollout_len and total_steps < args.total_steps:
            
            # Policy Step
            with set_exploration_type(ExplorationType.RANDOM):
                td_out = learner.policy(obs_td)

            world_act = td_out["agents", "action"][0][0].detach().cpu().numpy()[:2]
            norm_act = td_out["agents", "action_normalized"][0][0].detach().cpu().numpy()
            logp = td_out.get(("agents", "action_normalized_log_prob"), None)
            logp_val = logp.item() if logp is not None else 0.0
            value = td_out["state_value"].item()

            # Env Step
            from crowd_sim_plus.envs.utils.action import ActionXY as EnvActionXY
            action = EnvActionXY(world_act[0], world_act[1])
            ob, reward, done, info = env.step(action)
            
            # Next Observation
            next_state = robot.get_joint_state(ob, env.static_obstacles)
            next_obs_td = build_obs_tensordict(str(device), next_state, lidar_hres_deg, vfov_angles_deg, max_range)
            hicm_next_state = extract_hicm_state(next_obs_td)
            
            # Curiosity Reward
            # Action for HICM: normalized action (network output) or world action?
            # Usually network output (normalized) is better for learning dynamics in latent space
            
            # Ensure norm_act is 2D for HICM (vx, vy)
            if norm_act.shape[0] >= 2:
                norm_act_hicm = norm_act[:2]
            else:
                # If < 2 dims (e.g. 1D), pad with zeros
                norm_act_hicm = np.pad(norm_act, (0, 2 - norm_act.shape[0]), 'constant')

            action_tensor = torch.tensor(norm_act_hicm, device=device).unsqueeze(0) # (1, 2)
            
            intrinsic_reward = hicm.compute_intrinsic_reward(hicm_state, hicm_next_state, action_tensor)
            
            total_reward = reward + intrinsic_reward

            # Storage
            storage["obs"].append(obs_td.cpu())
            storage["act"].append(norm_act)
            storage["logp"].append(logp_val)
            storage["rew"].append(float(total_reward)) # Use total reward for PPO
            storage["done"].append(float(done))
            storage["val"].append(float(value))
            
            # Store data for HICM update
            # We need to detach tensors to avoid graph retention if not needed, 
            # but for update we might need them. HICM update re-computes forward pass.
            # So we store the inputs.
            storage["hicm_state"].append([t.clone() for t in hicm_state])
            storage["hicm_next_state"].append([t.clone() for t in hicm_next_state])
            storage["hicm_action"].append(action_tensor.clone())

            # Update loop vars
            obs_td = next_obs_td
            hicm_state = hicm_next_state
            total_steps += 1
            steps_in_rollout += 1

        # GAE
        rews = np.array(storage["rew"], dtype=np.float32)
        dones = np.array(storage["done"], dtype=np.float32)
        vals = np.array(storage["val"], dtype=np.float32)
        last_value = 0.0 # Bootstrap
        adv, ret = compute_gae(rews, dones, vals, last_value)
        
        storage["adv"] = list(adv)
        storage["ret"] = list(ret)

        # Update Phase
        obs_list = [td.to(device) for td in storage["obs"]]
        obs_td_batch = TensorDict.stack(obs_list, dim=0)
        
        actions_batch = torch.as_tensor(np.stack(storage["act"], axis=0), dtype=torch.float32, device=device)
        old_logp_batch = torch.as_tensor(np.stack(storage["logp"], axis=0), dtype=torch.float32, device=device)
        adv_batch = torch.as_tensor(np.stack(storage["adv"], axis=0), dtype=torch.float32, device=device)
        ret_batch = torch.as_tensor(np.stack(storage["ret"], axis=0), dtype=torch.float32, device=device)
        
        # Normalize advantage
        adv_batch = (adv_batch - adv_batch.mean()) / (adv_batch.std() + 1e-8)

        # PPO Update
        for _ in range(args.epochs):
            with set_exploration_type(ExplorationType.MEAN):
                learner.policy.feature_extractor(obs_td_batch)
                feat = obs_td_batch["_feature"]
                td_feat = TensorDict({"_feature": feat}, device=device)
                actor_out = learner.policy.actor.module(td_feat)
                alpha, beta = actor_out["alpha"], actor_out["beta"]
                from sicnav.policy.ponav import IndependentBeta
                dist = IndependentBeta(alpha, beta)
                new_logp = dist.log_prob(actions_batch)
                
                values = learner.policy.critic(TensorDict({"_feature": feat}, device=device))["state_value"].squeeze(-1)

            ratio = torch.exp(new_logp - old_logp_batch)
            surr1 = ratio * adv_batch
            surr2 = torch.clamp(ratio, 1.0 - args.clip_ratio, 1.0 + args.clip_ratio) * adv_batch
            actor_loss = -torch.mean(torch.min(surr1, surr2))
            value_loss = torch.nn.functional.mse_loss(values, ret_batch)
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

        # HICM Update
        # Batch process HICM data
        # HICM optimize takes single step inputs in the provided code?
        # Let's check optimize method in curiosity.py
        # It takes state, next_state, action.
        # If we pass batches, the networks should handle it (LazyLinear/Linear usually handle batch dim)
        # But NavRLModel forward expects specific tuple structure.
        # We need to stack the tuples.
        
        hicm_state_batch = (
            torch.cat([s[0] for s in storage["hicm_state"]], dim=0),
            torch.cat([s[1] for s in storage["hicm_state"]], dim=0),
            torch.cat([s[2] for s in storage["hicm_state"]], dim=0)
        )
        hicm_next_state_batch = (
            torch.cat([s[0] for s in storage["hicm_next_state"]], dim=0),
            torch.cat([s[1] for s in storage["hicm_next_state"]], dim=0),
            torch.cat([s[2] for s in storage["hicm_next_state"]], dim=0)
        )
        hicm_action_batch = torch.cat(storage["hicm_action"], dim=0)
        
        # Run HICM optimization (maybe multiple epochs or just once per rollout?)
        # Usually once per rollout or batched.
        # The optimize method does one step.
        hicm.optimize(hicm_state_batch, hicm_next_state_batch, hicm_action_batch)

        episode_idx += 1
        print(f"[PONav+HICM] Episode {episode_idx}, Steps {steps_in_rollout}, Total {total_steps}")

        if total_steps % args.save_every < args.rollout_len:
            ckpt_dir = os.path.join(os.path.dirname(__file__), 'ckpts')
            os.makedirs(ckpt_dir, exist_ok=True)
            torch.save(learner.policy.state_dict(), os.path.join(ckpt_dir, 'navrl_hicm_checkpoint.pt'))

if __name__ == '__main__':
    main()
