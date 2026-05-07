import argparse
import os
import hydra
import datetime
try:
    import wandb
except Exception:
    wandb = None
import torch
from omegaconf import DictConfig, OmegaConf

from torchrl.envs.utils import ExplorationType
from ppo import PPO
import importlib.util as _ilu




FILE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "cfg")
@hydra.main(config_path=FILE_PATH, config_name="train", version_base=None)
def main(cfg):
    # Use Wandb to monitor training
    if wandb is None:
        if (cfg.wandb.run_id is None):
            run = wandb.init(
                project=cfg.wandb.project,
                name=f"{cfg.wandb.name}/{datetime.datetime.now().strftime('%m-%d_%H-%M')}",
                entity=cfg.wandb.entity,
                config=cfg,
                mode=cfg.wandb.mode,
                id=getattr(wandb.util, "generate_id", lambda: None)(),
            )
        else:
            run = wandb.init(
                project=cfg.wandb.project,
                name=f"{cfg.wandb.name}/{datetime.datetime.now().strftime('%m-%d_%H-%M')}",
                entity=cfg.wandb.entity,
                config=cfg,
                mode=cfg.wandb.mode,
                id=cfg.wandb.run_id,
                resume="must"
            )
    else:
        class _Dummy:
            dir = os.getcwd()
            def log(self, *_args, **_kwargs):
                pass
        run = _Dummy()

    # Navigation Training Environment (TorchRL ponav backend only)
    navrl_env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "envs", "navrl_env.py"))
    spec = _ilu.spec_from_file_location("navrl_env", navrl_env_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load ponav env from {navrl_env_path}")
    navrl_env_mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(navrl_env_mod)
    NavRLEnvTorch = getattr(navrl_env_mod, "NavRLEnvTorch")
    transformed_env = NavRLEnvTorch(cfg).train()
    transformed_env.set_seed(cfg.seed)

    # PPO Policy
    policy = PPO(cfg.algo, transformed_env.observation_spec, transformed_env.action_spec, cfg.device)

    # checkpoint = "/home/zhefan/catkin_ws/src/navigation_runner/scripts/ckpts/checkpoint_2500.pt"
    checkpoint = "/home/zhujingqi/MultiAgent/CrowdNavigationMPC/curiosity_rl/training/scripts/navrl_checkpoint.pt"
    policy.load_state_dict(torch.load(checkpoint))
    
    # Episode Stats Collector
    episode_stats_keys = [
        k for k in transformed_env.observation_spec.keys(True, True) 
        if isinstance(k, tuple) and k[0]=="stats"
    ]
    episode_stats = None

    # RL Data Collector
    from torchrl.collectors import SyncDataCollector as TorchRLSyncCollector
    collector = TorchRLSyncCollector(
        transformed_env,
        policy=policy,
        frames_per_batch=cfg.env.num_envs * cfg.algo.training_frame_num,
        total_frames=int(cfg.max_frame_num),
        device=cfg.device,
        exploration_type=ExplorationType.RANDOM,
    )

    # Training Loop
    for i, data in enumerate(collector):
        # print("data: ", data)
        # print("============================")
        # Log Info
        # Collector introspection may vary; safely compute basic stats
        frames = getattr(collector, "_frames", None)
        fps = getattr(collector, "_fps", None)
        info = {"env_frames": frames if frames is not None else 0}
        if fps is not None:
            info["rollout_fps"] = fps

        # Train Policy
        train_loss_stats = policy.train(data)
        info.update(train_loss_stats) # log training loss info

        # Calculate and log training episode stats
        if episode_stats is not None:
            episode_stats.add(data)
            if len(episode_stats) >= transformed_env.num_envs:
                stats = {
                    "train/" + (".".join(k) if isinstance(k, tuple) else k): torch.mean(v.float()).item() 
                    for k, v in episode_stats.pop().items(True, True)
                }
                info.update(stats)

        # Update wand info
        run.log(info)

        # Console log for quick monitoring
        mean_reward = data["next", "agents", "reward"].mean().item() if ("next", "agents", "reward") in data.keys(True, True) else 0.0
        actor_loss = info.get("actor_loss", 0.0)
        critic_loss = info.get("critic_loss", 0.0)
        entropy = info.get("entropy", 0.0)
        fps_print = info.get("rollout_fps", 0.0) if info.get("rollout_fps", None) is not None else 0.0
        env_frames_print = info.get("env_frames", 0)
        print(
            f"Iter {i:04d} | Frames: {env_frames_print} | FPS: {fps_print:.2f} | "
            f"Reward: {mean_reward:.4f} | Actor: {actor_loss:.4f} | Critic: {critic_loss:.4f} | Entropy: {entropy:.4f}",
            flush=True,
        )


        # Save Model
        if i % cfg.save_interval == 0:
            ckpt_path = os.path.join(run.dir, f"checkpoint_{i}.pt")
            torch.save(policy.state_dict(), ckpt_path)
            print("[PONav]: model saved at training step: ", i)

    ckpt_path = os.path.join(run.dir, "checkpoint_final.pt")
    torch.save(policy.state_dict(), ckpt_path)
    if wandb is not None:
        wandb.finish()

if __name__ == "__main__":
    main()
    