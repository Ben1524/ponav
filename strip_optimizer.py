import zipfile
import os
import shutil

original_model_path = "/home/zhujingqi/MultiAgent/CrowdNavigationMPC/RL_nav/logs/rgl_sim_hallway_bottleneck_occlusion_False_rot_bound_8.75_speed_samps_3_rot_samps_10_humans_20_hmin_3_hmax_3_0/rgl_sim_hallway_bottleneck_occlusion_False_rot_bound_8.75_speed_samps_3_rot_samps_10_humans_20_hmin_3_hmax_3_66000_steps.zip"
new_model_path = "/home/zhujingqi/MultiAgent/CrowdNavigationMPC/RL_nav/logs/rgl_sim_hallway_bottleneck_occlusion_False_rot_bound_8.75_speed_samps_3_rot_samps_10_humans_20_hmin_3_hmax_3_0/rgl_sim_hallway_bottleneck_occlusion_False_rot_bound_8.75_speed_samps_3_rot_samps_10_humans_20_hmin_3_hmax_3_66000_steps_no_opt.zip"

def strip_optimizer(src, dst):
    with zipfile.ZipFile(src, 'r') as zin:
        with zipfile.ZipFile(dst, 'w') as zout:
            for item in zin.infolist():
                if item.filename != 'policy.optimizer.pth':
                    buffer = zin.read(item.filename)
                    zout.writestr(item, buffer)
                else:
                    print(f"Skipping {item.filename}")

if __name__ == "__main__":
    if os.path.exists(original_model_path):
        strip_optimizer(original_model_path, new_model_path)
        print(f"Created {new_model_path}")
    else:
        print(f"File not found: {original_model_path}")
