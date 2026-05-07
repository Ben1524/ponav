"""
Simple scenario runner: square area [-20,20] with ~40 humans.
This script is modeled after `sicnav/Test.py` but simplified: it
- loads the existing `sicnav/configs/env.config`
- overrides `rect_width`/`rect_height` to 40 (so coordinates in [-20,20])
- sets `sim.test_sim` to `no_walls` (free square) and `human_num` to 40
- configures and creates the env and robot, runs one reset and prints positions

Run:
    python sicnav/square_40.py

"""
import configparser
import logging
import os
import numpy as np
import gym

from crowd_sim_plus.envs.utils.robot_plus import Robot, RobotFullKnowledge
from crowd_sim_plus.envs.crowd_sim_plus import CrowdSimPlus


def main():
    # configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s, %(levelname)s: %(message)s')

    # load base config from sicnav folder
    base_cfg_path = os.path.join(os.path.dirname(__file__), 'configs', 'env.config')
    config = configparser.RawConfigParser()
    logging.info('Loading env config from %s', base_cfg_path)
    config.read(base_cfg_path)

    # override to form a square [-20,20]
    config.set('sim', 'rect_width', str(40))
    config.set('sim', 'rect_height', str(40))
    # choose a simulation rule that places agents within the rectangle / no walls
    # "no_walls" is accepted by generate_random_human_position in CrowdSimPlus
    config.set('sim', 'train_val_sim', 'no_walls')
    config.set('sim', 'test_sim', 'no_walls')

    # set approx 40 humans
    config.set('sim', 'human_num', str(40))

    # ensure starts_moving small to see initial motion (optional)
    config.set('sim', 'starts_moving', str(0))

    # create env
    env = gym.make('CrowdSimPlus-v0').unwrapped
    env.configure(config)

    # create robot from config and set it on the env
    # use simple Robot (not SB3)
    robot = Robot(config, 'robot')
    env.set_robot(robot)

    # run one reset and print info
    ob, static_obs = env.reset(phase='test', test_case=0, return_stat=True)
    logging.info('Robot position: %s, Robot goal: %s', env.robot.get_position(), env.robot.get_goal_position())

    # print first 10 humans' positions and goals
    for i, h in enumerate(env.humans[:10]):
        logging.info('Human %d: pos=(%.2f, %.2f), goal=(%.2f, %.2f), radius=%.2f', i, h.px, h.py, h.gx, h.gy, h.radius)

    # quick summary
    logging.info('Total humans in scene: %d', len(env.humans))
    logging.info('Static obstacles count: %d', len(env.static_obstacles))


if __name__ == '__main__':
    main()
