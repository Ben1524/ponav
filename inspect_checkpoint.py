import torch
import zipfile
import os
import io
import pickle

model_path = "/home/zhujingqi/MultiAgent/CrowdNavigationMPC/RL_nav/logs/rgl_sim_hallway_bottleneck_occlusion_False_rot_bound_8.75_speed_samps_3_rot_samps_10_humans_20_hmin_3_hmax_3_0/rgl_sim_hallway_bottleneck_occlusion_False_rot_bound_8.75_speed_samps_3_rot_samps_10_humans_20_hmin_3_hmax_3_66000_steps.zip"

try:
    with zipfile.ZipFile(model_path, "r") as archive:
        print(f"Files in zip: {archive.namelist()}")
        
        # Load data
        with archive.open("data") as f:
            data = pickle.load(f)
            print("Keys in data:", data.keys())
            if "optimizer" in data:
                print("Optimizer in data:", data["optimizer"])
            
        # Load params
        with archive.open("pytorch_variables.pth") as f:
            params = torch.load(f, map_location="cpu")
            print("Keys in params:", params.keys())
            if "policy.optimizer" in params:
                opt_state = params["policy.optimizer"]
                print("Optimizer state keys:", opt_state.keys())
                print("Param groups:", len(opt_state["param_groups"]))
                for i, group in enumerate(opt_state["param_groups"]):
                    print(f"Group {i} params:", len(group["params"]))

except Exception as e:
    print(f"Error inspecting file: {e}")
