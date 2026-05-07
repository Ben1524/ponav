import os
import logging
import sys
import argparse

from RL_test import test

curr_dir = os.getcwd()

parser = argparse.ArgumentParser(description='Override Default Values')
parser.add_argument('--save_dir', type=str, default=os.path.join(curr_dir, "test_logs"), help='location of environment config file')

parser.add_argument('--model_name', type=str, default= "sarl_sim_hallway_bottleneck_occlusion_False_rot_bound_8.75_speed_samps_3_rot_samps_10_humans_3_hmin_3_hmax_3")
parser.add_argument('--version_num', type=int, default=5)
parser.add_argument('--trained_steps', type=int, default=4000)
parser.add_argument('--save_name', type=str, default=None,
                    help='Override default save name which stems from model name.')

args = parser.parse_args()

# configure logging and device
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

stdout_handler = logging.StreamHandler(sys.stdout)
log_file_path = os.path.join(curr_dir, 'logs')
if not os.path.exists(log_file_path):
    os.makedirs(log_file_path)
i=0
logfile_name = 'debug_log_{:}.log'.format(i)
while os.path.exists(os.path.join(log_file_path, logfile_name)):
    i+=1
    logfile_name = 'debug_log_{:}.log'.format(i)
file_handler = logging.FileHandler(os.path.join(log_file_path, logfile_name), mode='w')
# set logging config for both handlers. Set the level for stdout to INFO and the level for file to DEBUG
stdout_handler.setLevel(logging.INFO)
file_handler.setLevel(logging.DEBUG)
# set the format for both handlers
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s, %(levelname)s: %(message)s', handlers=[stdout_handler, file_handler],
                    datefmt='%Y-%m-%d %H:%M:%S')

run_log_dir = os.path.join(curr_dir, f"logs/{args.model_name}_{args.version_num}")
model_path = os.path.join(run_log_dir, f"{args.model_name}_{args.trained_steps}_steps.zip")

if not os.path.exists(model_path):
    raise FileNotFoundError(
        f"Could not find trained model archive at {model_path}. "
        "Ensure the run exists (check logs/<model>_<version>/) or override --model_name/--version_num/--trained_steps.")

# grab configs from model
env_config_file = os.path.join(run_log_dir, "env.config")
policy_config_file = os.path.join(run_log_dir, "sarl_policy.config")

if not os.path.exists(env_config_file):
    fallback_env = os.path.join(curr_dir, "configs/env.config")
    logging.warning("env.config not found at %s; falling back to %s", env_config_file, fallback_env)
    env_config_file = fallback_env

if not os.path.exists(policy_config_file):
    fallback_policy = os.path.join(curr_dir, "configs/sarl_policy.config")
    logging.warning("sarl_policy.config not found at %s; falling back to %s", policy_config_file, fallback_policy)
    policy_config_file = fallback_policy

test(args.save_dir, args.model_name, model_path, env_config_file, policy_config_file, False)