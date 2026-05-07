import os
from typing import Optional
import torch
import gym
import logging
import numpy as np
from crowd_sim_plus.envs.policy.policy import Policy
from sicnav.policy.training.networks.model import Policy as NetworkPolicy
from crowd_sim_plus.envs.utils.action import ActionRot, ActionXY 

import numpy as np
import os

'''
config with reduced routes in csl workspace, for sim2real
'''


class BaseConfig(object):
    def __init__(self):
        pass


class Config(object):
    # environment settings
    env = BaseConfig()
    # no obstacle in observation: 'CrowdSim3D-v0',
    # obstacle in observation: 'CrowdSim3DTbObs-v0'
    env.env_name = 'CrowdSim3DTbObs-v0'  # name of the gym environment
    env.action_space = 'discrete'  # discrete or continuous action space
    # recommended value: if goal dist in [7, 9]: 30, if goal dist < 5: 20
    env.time_limit = 50  # time limit of each episode (second)
    env.time_step = 0.1  # length of each timestep/control frequency (second)
    env.val_size = 100
    env.test_size = 500  # number of episodes for test.py
    env.randomize_attributes = False  # randomize the preferred velocity and radius of humans or not
    # todo: change this
    env.seed = 50569  # random seed for environment
    env.use_wrapper = False
    # whether we are collecting IL demo data or not, will be overwritten in main function
    env.il_env = False
    # circle_crossing: circle crossing humans, random robot init & goal poses, random obstacles
    # csl_workspace: human flow in a set of regions, robot init & goal poses in a set of regions, fixed obstacles
    # todo: change this
    env.scenario = 'circle_crossing'

    # workstation, or entrance
    env.csl_workspace_type = 'entrance'
    # sim or sim2real
    env.mode = 'sim'

    # robot action type
    action_space = BaseConfig()
    # holonomic or unicycle or turtlebot
    action_space.kinematics = "turtlebot"

    ob_space = BaseConfig()
    # the robot state contains absolute positions [px, py, gx, gy] or relative positions [gx-px, gy-py]
    # note: for best result, relative positions require info on static obstacles
    ob_space.robot_state = 'absolute'  # absolute or relative
    # True: human observation is [px, py, vx, vy], False: human observation is [px, py]
    if env.mode == 'sim':
        ob_space.add_human_vel = True
    else:
        ob_space.add_human_vel = False
    # include humans + obs in lidar pc, or only include obs
    # todo: change this
    ob_space.lidar_pc_include_humans = False
    # the human states are in robot frame or world frame
    if env.mode == 'sim':
        ob_space.human_state_frame = 'robot'
    else:
        ob_space.human_state_frame = 'world'
    # the human velocity values are absolute (w.r.t. a static frame) or relative (w.r.t. the robot's velocity)
    ob_space.human_vel = 'absolute'
    ob_space.sort_humans = False

    # reward function
    reward = BaseConfig()
    if action_space.kinematics in ["unicycle", "turtlebot"]:
        reward.success_reward = 20
    else:
        reward.success_reward = 10
    reward.collision_penalty = -20
    # discomfort distance
    reward.discomfort_dist = 0.25
    reward.discomfort_penalty_factor = 10
    if 'Hie' in env.env_name:
        reward.potential_reward_factor = 1
    else:
        reward.potential_reward_factor = 2
    if action_space.kinematics == 'unicycle':
        reward.spin_factor = 4.5
        reward.back_factor = 0.5
    elif action_space.kinematics == 'turtlebot':
        reward.spin_factor = 0.05
        reward.back_factor = 0.
    else:
        reward.spin_factor = 0
        reward.back_factor = 0
    # a constant penalty subtracted at every timestep, to prevent robot timeout especially when the task horizon is long
    reward.constant_penalty = -0.025
    reward.waypoint_reward = 1
    reward.gamma = 0.99  # discount factor for rewards

    # environment settings
    sim = BaseConfig()
    # controls the agent positions
    sim.circle_radius = 4
    # sim.robot_circle_radius = 5
    sim.robot_circle_radius = 4
    # controls the obstacle positions
    if env.mode == 'sim':
        sim.arena_size = 4.5
    else:
        # for om, om size = arena_size + 1
        if env.csl_workspace_type == 'workstation':
            sim.arena_size = 6
        elif env.csl_workspace_type == 'entrance':
            sim.arena_size = 11
    # number of dynamic humans
    sim.human_num = 7
    # the range of human_num is human_num-human_num_range~human_num+human_num_range
    sim.human_num_range = 2
    # number of static humans
    sim.static_human_num = 1
    sim.static_human_range = 1
    # actual human num is in [human_num-human_num_range, human_num+human_num_range]
    # warning: may have problems if human_num - human_num_range < observed_human_num

    # change human num within an episode periodically
    sim.change_human_num_in_episode = False
    # Group environment: set to true; FoV environment: false
    sim.group_human = False
    sim.human_pos_noise_range = 2
    # add static obstacles or not
    sim.static_obs = True
    # the position and size of obstacles are random or fixed
    if env.scenario == 'circle_crossing':
        sim.random_obs = True
        sim.obs_size_mean = 1
        sim.obs_size_std = 0.6
        sim.obs_max_size = 5
        sim.obs_min_size = 0.1
    else:
        sim.random_obs = False
    sim.static_obs_num = 10
    sim.static_obs_num_range = 2
    # whether we allow obstacles to overlap
    sim.obs_can_overlap = False
    # minimal distance between each pair of obstacles
    sim.obs_min_dist = 1
    # randomize the height of obstacles or not (if True, some obs will be too short and not detectable by lidar)
    sim.random_static_obs_height = False
    # add borders or not, the border will be a square centered at (0, 0) with width = 2*sim.arena_size
    sim.borders = True
    if env.scenario == 'csl_workspace':
        sim.borders = False
    sim.predict_steps = 5
    # 'const_vel': constant velocity model,
    # 'truth': ground truth future traj (with info in robot's fov)
    # 'inferred': inferred future traj from GST network
    # 'none': no prediction
    sim.predict_method = 'none'
    # render the simulation during training or not
    sim.render = False

    human_flow = BaseConfig()
    # r, g, b, alpha
    human_flow.colors = [[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1], [1, 0, 1, 1], [0, 1, 1, 1], [0, 0, 0, 1],
		          [1, 0, 0.5, 1], [0, 1, 0.5, 1], [0, 0.5, 1, 1], [0.5, 1, 0, 1], [0.5, 0, 1, 1],
		         [1, 0.5, 0.5, 1],[0.5, 1, 0.5, 1], [0.5, 0.5, 1, 1], [0.25, 0, 1, 1], [0, 1, 0.25, 1], [0.25, 0.25, 0, 1], [1, 0.25, 0, 1]]
    # assert len(human_flow.init_pos) == len(human_flow.final_pos)
    assert len(human_flow.colors) >= sim.human_num + sim.human_num_range

    if env.scenario == 'csl_workspace':
        fixed_obs = BaseConfig()
        # [width, height] of all obstacles
        # left vertical wall, right vertical wall,
        # 3 workstations (the middle two are combined) from upper to lower, the extra horizontal wall on the bottom (near 0, 0)
        # the upper left and upper right rooms, the horizontal wall on the upper right,
        # the vertical wall on the upper right, the vertical wall on the upper left

        if env.csl_workspace_type == 'workstation':
            # define obstacles based on map
            divider_width = 6
            # only includes the first 3 lines of workstation from bottom
            # [width, height] of all obstacles
            fixed_obs.sizes = np.array([[50, 808], [10, 992],
                                        [699, 84], [699, 84 * 2], [699, 84], [140, 250],
                                        [270 + 10, 250], [1137 + 10, 250], [650, 10],
                                        [10, 202], [10, 136],
                                        # vertical dividers that seperate desks and hallway
                                        [divider_width, 145], [divider_width, 144+159], [divider_width, 144]
                                        ]) / 100.
            # [x, y] coordinates of lower left corners of all obstacles
            fixed_obs.positions_lower_left = np.array([[-826 - 10, 0], [81, -250],
                                                       [-796, 724], [-796, 320], [-796, 0], [-221, -250],
                                                       [-846 - 10, 944], [-406, 944], [81, 742],
                                                       [731, 742], [-846 - 10, 808],
                                                       # vertical dividers that seperate desks and hallway
                                                       [-97-divider_width, 0], [-97-divider_width, 260], [-97-divider_width, 664]
                                                       ]) / 100.
            # 1: rectangular cube, 0: cylinder
            fixed_obs.shapes = np.array([1] * len(fixed_obs.sizes))

            # define human routes based on map
            # human_flow.static_regions = np.array([[-650, -250, 115, 290], [-650, -250, 520, 690]]) / 100.
            # human_flow.static_regions = np.array([[-650, -250, 115, 165], [-650, -250, 240, 290],
            #                                       [-650, -250, 520, 570], [-650, -250, 640, 690]]) / 100.
            human_flow.static_regions = np.array([[-650, -250, 115, 165], [-650, -250, 240, 290],
                                                  [-650, -250, 510, 550], [-650, -250, 650, 690]]) / 100.
            # will be triggered ONLY IF sim.static_obs = True and sim.random_obs = False
            # key: region number, value: [x_low, x_high, y_low, y_high] of the rectangular shaped region
            human_flow.regions = {1: np.array([-60, 20, -400, -300]) / 100.,
                                  2: np.array([-200, -100, 115, 290]) / 100.,
                                  # 3: np.array([-60, 40, 145, 260]) / 100.,
                                  # 3.5: np.array([-60, 40, 300, 400]) / 100.,
                                  3: np.array([-60, 20, 0, 260]) / 100.,
                                  3.5: np.array([-60, 20, 300, 400]) / 100.,
                                  4: np.array([-200, -100, 520, 690]) / 100.,
                                  5: np.array([-60, 40, 563, 664]) / 100.,
                                  6: np.array([-650, -406, 840, 910]) / 100.,
                                  7: np.array([-221, 40, 790, 910]) / 100.,
                                  8: np.array([100, 600, 790, 910]) / 100.
                                  }

            # the route of each human is chosen independently (less controlled), or they are correlated (more controlled)
            human_flow.route_type = 'correlated'
            # human routes does not cover 6, 8
            # human_flow.routes = [[1, 3, 5, 7], [7, 5, 3, 1],
            #                      [1, 3, 5, 4], [4, 5, 3, 1],
            #                      [2, 3, 5, 7], [7, 5, 3, 2],
            #                      [2, 3, 5, 4], [4, 5, 3, 2],
            #                      [1, 3, 2], [2, 3, 1],
            #                      [5, 3, 1], [1, 3, 5],
            #                      [7, 5, 3], [3, 5, 7],
            #                      [3, 5, 4], [4, 5, 3],
            #                      [4, 5, 7], [7, 5, 4]
            #                      ]
            human_flow.routes = [
                                # both human and robot's routes are straight lines
                                 [7, 5, 3, 1],
                                 [5, 3, 1],
                                 [3.5, 1],
                                # the human takes a turn and cross the robot
                                 [4, 5, 3, 1],
                                 [2, 3, 1],
                                # the human takes a turn and does not cross the robot
                                 [2, 3, 5, 7], [7, 5, 3, 2],
                                 [2, 3, 5, 4], [4, 5, 3, 2],
                                 [3.5, 5, 4],
                                 [7, 5, 4],
                                 ]
            human_flow.correlated_routes = [
                [[7, 5, 3], [3.5, 1]],
                [[4, 5, 3], [3.5, 1]],
                [[4, 5, 3], [2, 3, 1]],
                [[7, 5, 3], [2, 3, 1]]
            ]

        elif env.csl_workspace_type == 'entrance':
            fixed_obs.cylinder_radius = 0.5 # 0.45
            fixed_obs.cylinder_height = 0.75
            fixed_obs.sizes = np.array([[600, 10], [374, 474], # lower wall, left room
                                        [74, 242], [283, 339],  # sofas, right room
                                        [381, 646], [590, 10], [10, 646], [116, 14], # upper right room (hca lab), upper wall of cafe, left wall of cafe, lower left wall of cafe
                                        [90, 55], [90, 100], [75, 182], # trash cans, vending machine, rectangle table
                                        [fixed_obs.cylinder_radius*200, fixed_obs.cylinder_radius*200],
                                        [fixed_obs.cylinder_radius*200, fixed_obs.cylinder_radius*200],
                                        [fixed_obs.cylinder_radius*200, fixed_obs.cylinder_radius*200], # treat circles as rectangles
                                        ]) / 100.
            # [x, y] coordinates of lower left corners of all rectangles, and centers of all cylinders
            fixed_obs.positions_lower_left = np.array([[-158.5, -146], [-532.5, -136], # lower wall, left room
                                                       [84.5, 48.5], [158.5, 0], # sofas, right room
                                                       [60.5, 509], [-529.5, 1155], [-539.5, 509], [-529.5, 509],# upper right room (hca lab), upper wall of cafe, left wall of cafe, lower left wall of cafe
                                                       [-404.5, 1100], [-314.5, 1090], [-14.5, 654],# trash cans, vending machine, rectangle table
                                                       [-367.5 - fixed_obs.cylinder_radius*100, 882 - fixed_obs.cylinder_radius*100],
                                                       [-98.5 - fixed_obs.cylinder_radius*100, 975 - fixed_obs.cylinder_radius*100],
                                                       [-119.5 - fixed_obs.cylinder_radius*100, 635 - fixed_obs.cylinder_radius*100] # lower left corner of 3 round tables
                                                      ]) / 100.
            # 1: rectangular cube, 0: cylinder
            fixed_obs.shapes = np.array([1] * 11 + [0] * 3)
            # fixed_obs.shapes = np.array([1] * 11)

            # define human routes based on map
            human_flow.static_regions = np.array([[-150, 60, 509, 1155],
                                                  [-520, -450, 509, 1155],
                                                  [74, 84.5+74, 48.5+242, 380], # sofa corner
                                                  [-158.5, 158.5, -120, -60], # near glass entrance door

                                                  ]) / 100.

            # will be triggered ONLY IF sim.static_obs = True and sim.random_obs = False
            # key: region number, value: [x_low, x_high, y_low, y_high] of the rectangular shaped region
            human_flow.regions = {0: np.array([300, 400, -100, 0]) / 100.,
                                  1: np.array([-120, -20, -100, 0]) / 100.,
                                  2: np.array([-700, -400, 360, 470]) / 100.,
                                  3: np.array([-150, -20, 360, 470]) / 100.,
                                  3.5: np.array([-150, -20, 360, 600]) / 100.,
                                  4: np.array([400, 700, 360, 470]) / 100.,
                                  5: np.array([-450, -300, 500, 600]) / 100.,
                                  6: np.array([-200, -50, 650, 750]) / 100.,
                                  # 6: np.array([-200, -150, 500, 600]) / 100.,
                                  7: np.array([-300, -150, 900, 1000]) / 100.,
                                  8: np.array([-300, -150, 500, 600]) / 100.
                                  }

            # the route of each human is chosen independently (less controlled), or they are correlated (more controlled)
            human_flow.route_type = 'independent'

            human_flow.routes = [
                                 [2, 3, 4], [4, 3, 2],
                                 # [5, 6, 5, 6, 5, 6, 5, 6, 5, 6, 5, 6, 5, 6, 5, 6, 5, 6, 5, 6],
                                [7, 8, 2], [7, 8, 4],
                                [7, 8, 3, 1],
                                [7, 8, 3, 1, 0],
                                [7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,8, 2],
                                [7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,  7, 7,8, 4],
                                [7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,7, 7, 7, 7, 7,  7, 7,  7, 7, 8, 3, 1, 0],
                                [3, 1],
                                [3, 1, 0],
                [3.5, 1],
                [3.5, 1, 0],
            ]
            # todo: change this
            human_flow.correlated_routes = [
                                            [[7, 8, 3, 1], [3, 1]],
                                            [[7, 8, 3, 1], [3, 1, 0]],

                                            [[2, 3, 4], [4, 3, 2]],
                                            [[7, 5, 3], [2, 3, 1]]
                                            ]
        else:
            raise ValueError("Unknown csl_workspace_type")
        assert len(fixed_obs.sizes) == len(fixed_obs.positions_lower_left)

        # change the static obs information based on the fixed_obs above
        sim.static_obs_num = len(fixed_obs.sizes)
        sim.static_obs_num_range = 0

        # make sure each route has a start region and at least one goal region
        for route in human_flow.routes:
            assert len(route) >= 2

        # adjust the human_num to prevent errors for correlated routes
        if human_flow.route_type == 'correlated':
            sim.human_num = sim.static_human_num + max(len(sublist) for sublist in human_flow.correlated_routes)
            sim.human_num_range = 0

    # for circle crossing sceanrio, humans start & goals are always sampled randomly
    else:
        human_flow.route_type = 'independent'


    # robot settings
    robot = BaseConfig()
    robot.visible = True  # the robot is visible to humans
    # If robot.visible = true, the probability that a human will react to the robot
    robot.visible_prob = 0.2
    # robot policy, with only human positions: srnn or selfAttn_merge_srnn
    # (ablation, RH only) robot policy, with only obstacle positions: dsrnn_human_obs
    # DSRNN + obstacle vertices are treated as humans: dsrnn_obs_vertex
    # DSRNN + obstacle lidar CNN: dsrnn_obs_pc
    # (Perez-D’Arpino et al baseline, with A*): with lidar scans: lidar_gru
    # with images: img_gru
    # with both lidar and rgbd images: fusion_gru
    # with both lidar and image pcs: pc_fusion_gru
    # (ours & ablation, RH + HH only) with both human positions and lidar: selfAttn_merge_srnn_lidar
    # (baseline, homogeneous attention graph network) homo_transformer_obs
    robot.policy = 'dsrnn_obs_vertex'  # todo: change this

    # Use env and network that can handle obstacles
    if robot.policy in ['dsrnn_obs', 'dsrnn_human_obs']:
        assert env.env_name == 'CrowdSim3DTbObs-v0'

    if action_space.kinematics == "turtlebot":
        robot.radius = 0.2
    else:
        robot.radius = 0.3  # radius of the robot
    robot.height = 0.45  # height of the robot
    robot.v_pref = 1  # max velocity of the robot
    robot.allow_backward = True
    # for turtlebot
    robot.v_max = 0.5
    if not robot.allow_backward:
        robot.v_min = 0
        reward.back_factor = 0.
    else:
        robot.v_min = -0.5
        reward.back_factor = 0.1
    robot.w_max = 1
    robot.w_min = -1
    # robot FOV = this values * PI
    robot.FOV = 2.
    # include (gx, gy) in the robot state in observation or not
    robot.visual_goal = True

    # for both circle_crossing and csl_workspace
    # range of distance between robot initial position and goal position
    # if you don't want to specify the range, set robot.min_goal_dist = 0 and robot.max_goal_dist = np.inf
    robot.min_goal_dist = 5  # 2
    robot.max_goal_dist = 6 # 4
    if env.mode == 'sim':
        robot.initTheta_range = [0, 2 * np.pi]
    else:
        robot.initTheta_range = [np.pi/2 - np.pi/6, np.pi/2 + np.pi/6]
    # for circle_crossing only
    # range of robot initial positions
    robot.initX_range = [-sim.robot_circle_radius, sim.robot_circle_radius]
    robot.initY_range = [-sim.robot_circle_radius, sim.robot_circle_radius]

    # range of robot goal positions
    robot.goalX_range = [-sim.robot_circle_radius, sim.robot_circle_radius]  # [-1.5, 0.4]
    robot.goalY_range = [-sim.robot_circle_radius, sim.robot_circle_radius]  # [7, 9]

    # key: region number, value: [x_low, x_high, y_low, y_high] of the rectangular shaped region
    if env.csl_workspace_type == 'workstation':
        robot.regions = {1: np.array([-0.5, 0.5, -0.3, 0.3]),
                         2: np.array([-0.3, 0.3, 5.5, 6]),
                         3: np.array([-7, -6, 5.5, 6.5]),
                         }
        # short-distance navigation
        robot.routes = [[1, 2]
                        ]
    elif env.csl_workspace_type == 'entrance':
        robot.regions = {1: np.array([-0.5, 0.5, -0.3, 0.3]),
                         2: np.array([-4, -3, 9, 10]),
                         3: np.array([-1.5, -0.5, 7.5, 8.5]),
                         }
        # short-distance navigation
        robot.routes = [[1, 3]
                        ]

    # config for sim2real
    sim2real = BaseConfig()
    # use dummy robot and human states or not
    sim2real.use_dummy_detect = False
    sim2real.test_nav_stack = False
    sim2real.record = False
    sim2real.load_act = False
    sim2real.ROSStepInterval = 0.03
    sim2real.fixed_time_interval = 0.1
    sim2real.use_fixed_time_interval = True
    # zed: only use zed2 camera to detect people
    # lidar: only use DR_SPAAM + LiDAR to detect people
    # fusion: use zed2 for people > 1m w.r.t. robot, use lidar for people < 1m w.r.t. robot
    sim2real.human_detector = 'lidar'
    sim2real.robot_localization = 't265'

    # config for data collection for training the GST predictor
    data = BaseConfig()
    data.tot_steps = 40000
    data.render = False
    data.collect_train_data = False
    data.num_processes = 5
    data.data_save_dir = 'gst_updated/datasets/orca_20humans_no_rand'
    # number of seconds between each position in traj pred model
    data.pred_timestep = 0.25

    # whether wrap the vec env with VecPretextNormalize class
    # = True only if we are using a network for human trajectory prediction (sim.predict_method = 'inferred')
    if sim.predict_method == 'inferred':
        env.use_wrapper = True
    else:
        env.use_wrapper = False

    # LIDAR config
    lidar = BaseConfig()
    lidar.add_lidar = True
    # angular resolution (offset angle between neighboring rays) in degrees
    lidar.angular_res = 2  # todo: 1
    # lidar range: see robot.sensor_range
    # the height of the lidar mounting point from floor
    lidar.height = 0.5
    lidar.sensor_range = 25  # based on official document of RPLidar R3
    lidar.visualize_rays = False  # should always be false to speed up training and testing without GUI

    # camera config
    camera = BaseConfig()
    camera.add_camera = False
    # camera field of view (in degrees)
    # todo: change this
    camera.fov = 360
    # angular resolution (offset angle between neighboring rays) in degrees
    # todo: change this
    camera.ray_angular_res = 2
    # mounting height of the camera
    camera.height = 0.55
    camera.img_width = 100  # * 2
    camera.img_height = 75  # * 2
    # width and height of the camera image in pixels
    camera.render_cam_fov = 120
    camera.render_cam_img_width = 2300 # * 2
    camera.render_cam_img_height = 2300 # * 2
    # the camera observation is the raw images (img) or reconstructed point clouds from depth image (pc)
    camera.ob_form = 'pc'
    camera.render_checkpoint = None # should always be None, will be changed in test.py

    # human settings
    humans = BaseConfig()
    humans.visible = True  # a human is visible to other humans and the robot
    # policy to control the humans: orca or social_force
    humans.policy = "orca"
    humans.radius = 0.25 # radius of each human
    humans.height = 0.7  # height of each human
    humans.v_pref = 0.5  # max velocity of each human
    # FOV = this values * PI
    humans.FOV = 2.

    # a human may change its goal before it reaches its old goal
    humans.random_goal_changing = False
    humans.goal_change_chance = 0.25

    # a human may change its goal after it reaches its old goal
    humans.end_goal_changing = True
    humans.end_goal_change_chance = 1.0

    # a human may change its radius and/or v_pref after it reaches its current goal
    humans.random_radii = False
    humans.random_v_pref = True

    # one human may have a random chance to be blind to other agents at every time step
    humans.random_unobservability = False
    humans.unobservable_chance = 0.3

    humans.random_policy_changing = False

    # add noise to observation or not
    noise = BaseConfig()
    noise.add_noise = False
    # uniform, gaussian
    noise.type = "uniform"
    noise.magnitude = 0.1

    # config for ORCA
    orca = BaseConfig()
    orca.neighbor_dist = 10
    orca.safety_space = 0.1
    orca.time_horizon = 5
    orca.time_horizon_obst = 5

    # config for social force
    sf = BaseConfig()
    sf.A = 2.
    sf.B = 1
    sf.KI = 1

    # cofig for RL ppo
    ppo = BaseConfig()
    ppo.num_mini_batch = 2  # number of batches for ppo
    ppo.num_steps = 30  # number of forward steps
    ppo.recurrent_policy = True  # use a recurrent policy
    ppo.epoch = 5  # number of ppo epochs
    ppo.clip_param = 0.2  # ppo clip parameter
    ppo.value_loss_coef = 0.5  # value loss coefficient
    ppo.entropy_coef = 0.01  # entropy term coefficient
    ppo.use_gae = True  # use generalized advantage estimation
    ppo.gae_lambda = 0.95  # gae lambda parameter

    # SRNN config
    SRNN = BaseConfig()
    SRNN.robot_embedding_size = 64
    SRNN.obs_embedding_size = 64
    SRNN.human_embedding_size = 64
    # RNN size
    SRNN.human_node_rnn_size = 43 # 128 # Size of Human Node RNN hidden state
    SRNN.human_human_edge_rnn_size = 43 # 128 # Size of Human Human Edge RNN hidden state

    # Input and output size
    SRNN.human_node_input_size = 5  # Dimension of the node features
    SRNN.human_human_edge_input_size = 2  # Dimension of the edge features
    SRNN.human_node_output_size = 256  # Dimension of the node output

    # Embedding size
    SRNN.human_node_embedding_size = 64  # Embedding size of node features
    SRNN.human_human_edge_embedding_size = 64  # Embedding size of edge features

    # Attention vector dimension
    # Attention vector dimension
    SRNN.hr_attention_size = 128  # robot-human Attention size
    SRNN.ho_attention_size = 128  # obstacle-human Attention size

    # for self attention
    SRNN.use_hr_attn = True  # RH attn
    SRNN.hr_attn_head_num = 1  # number of attention heads for RH attn
    SRNN.use_self_attn = True  # HH attn
    # todo: change this
    SRNN.use_oh_attn = True  # obstacle-human attn
    SRNN.oh_attn_head_num = 1  # number of attention heads for OH attn
    SRNN.self_attn_size = 128
    # use pretrained CNNs for image inputs or not
    SRNN.img_pretrained = False

    # imitation learning config
    il = BaseConfig()
    # please always set this to False, will be changed later in training scripts
    il.train_il = False
    il.data_save_dir = 'demo_data/orca_poseG_obsTall'
    # orca, selfAttn_merge_srnn
    il.expert_policy = 'orca'
    il.num_processes = 16
    # needed if the expert policy is a NN
    il.expert_policy_path = None
    # number of demonstration traj by expert policy
    il.expert_traj_num = 6200  # 5000
    il.expert_traj_len = 30
    il.data_load_dir = 'demo_data/orca_poseG_obsTall'
    il.batch_size = 64
    il.lr = 1e-4
    il.epoch_num = 100
    il.model_save_dir = 'data_il/orca_poseG_obsTall'
    il.log_interval = 1

    # training config
    training = BaseConfig()
    training.lr = 5e-5 # 1e-4  # learning rate (default: 8e-5)
    training.eps = 1e-5  # RMSprop optimizer epsilon
    training.alpha = 0.99  # RMSprop optimizer alpha
    training.max_grad_norm = 0.5  # max norm of gradients
    training.num_env_steps = 200e6 # number of environment steps to train: 10e6 for holonomic, 20e6 for unicycle
    training.use_linear_lr_decay = True  # use a linear schedule on the learning rate: True for unicycle, False for holonomic
    training.save_interval = 200  # save interval, one save per n updates
    training.log_interval = 20  # log interval, one log per n updates
    training.use_proper_time_limits = False  # compute returns taking into account time limits
    training.cuda_deterministic = False  # sets flags for determinism when using CUDA (potentially slow!)
    training.cuda = True  # use CUDA for training
    training.num_processes = 28  # was 16, how many training CPU processes to use
    # todo: change this
    training.output_dir = 'trained_models/DS_RNN'  # the saving directory for train.py
    # resume training from an existing checkpoint or not
    # none: train RL from scratch, rl: load a RL weight, il: load a IL weight
    training.resume = 'none'
    # if resume != 'none', load from the following checkpoint
    training.load_path = 'data/randEnv_3to5smallobs_9to11human_goal7to8/fakePC_RH_HH_OHattn_longTime/checkpoints/53200.pt'
    training.overwrite = True  # whether to overwrite the output directory in training
    training.num_threads = 1  # number of threads used for intraop parallelism on CPU
    # whether use curriculum with different stages or not
    training.use_curriculum = False
    # For each curriculum stage, the number of RL steps and config parameters to be changed
    # Note: if training.use_curriculum = true, the original values of the following config parameters will be ignored
    # todo: the parameters related to NN cannot be changed (e.x. ob_space, action space, SRNN.attention_size, etc)
    training.curriculum = [
        {
            # you MUST change seed for each stage!!! Otherwise, the policy will be trained with the same env scenarios as last stage
            ('env', 'seed'): 2351,
            ('training', 'num_env_steps'): 10e6,
            ('training', 'use_linear_lr_decay'): False,
            # ('sim', 'human_num'): 1,
            # ('robot', 'min_goal_dist'): 3,
            # ('robot', 'max_goal_dist'): 4,
            ('sim', 'static_obs_num'): 2
        },
        {
            # you MUST change seed for each stage!!! Otherwise, the policy will be trained with the same env scenarios as last stage
            ('env', 'seed'): 235581,
            ('training', 'num_env_steps'): 15e6,
            ('training', 'use_linear_lr_decay'): False,
            # ('sim', 'human_num'): 1,
            # ('robot', 'min_goal_dist'): 3,
            # ('robot', 'max_goal_dist'): 4,
            ('sim', 'static_obs_num'): 4
        },
        {
            ('env', 'seed'): 787358,  # you MUST change seed for each stage!!!
            ('training', 'num_env_steps'): 15e6,
            ('training', 'use_linear_lr_decay'): False,
            # ('sim', 'human_num'): 2,
            # ('robot', 'min_goal_dist'): 3,
            # ('robot', 'max_goal_dist'): 4,
            ('sim', 'static_obs_num'): 6
        },
        {
            ('env', 'seed'): 731908,  # you MUST change seed for each stage!!!
            ('training', 'num_env_steps'): 20e6,
            ('training', 'use_linear_lr_decay'): False,
            # ('sim', 'human_num'): 3,
            # ('robot', 'min_goal_dist'): 3,
            # ('robot', 'max_goal_dist'): 4
            ('sim', 'static_obs_num'): 8
        }
    ]

    # pybullet config
    # common env configuration
    pybullet = BaseConfig()
    pybullet.objList = ['cube', 'sphere', 'cone', 'cylinder']  # ['cube', 'sphere', 'cone', 'cylinder']
    pybullet.taskNum = len(pybullet.objList)
    pybullet.img_dim = (3, 96, 96)  # (channel, image_height, image_width)
    pybullet.sound_dim = (1, 100, 40)  # sound matrix dimension (1, frames, numFeat)

    pybullet.commonMediaPath = 'commonMedia'
    pybullet.mediaPath = 'crowd_sim/pybullet/media/'  # os.path.join("Envs", "pybullet", "turtlebot", "media")  # objects' model
    pybullet.envFolder = 'pybullet/'  # os.path.join('pybullet', 'turtlebot')

    pybullet.numRays = 11
    pybullet.rayLen = 4
    pybullet.rayHitColor = [1, 0, 0]
    pybullet.rayMissColor = [0, 1, 0]
    # simulation frequency (Note: this is different from
    pybullet.sim_timestep = 1. / 240  # recommended by PyBullet official
    pybullet.frameSkip = int(env.time_step / pybullet.sim_timestep)  # TODO: choose 36 if the control method is rotPose

    # robot_bases.py will use the name indicated by robotName as the robot body
    # the robot body represents the whole robot, which is usually the base of a mobile robot
    pybullet.robotName = 'base_link'
    pybullet.robotScale = 1

    # robot camera
    pybullet.robotCamOffset = 0.02  # it is used to adjust the near clipping plane of the camera
    pybullet.robotCamRenderSize = (75, 100, 3)  # simulation render (height, width, channel)
    pybullet.robotFov = 48.8
    # pybullet debug GUI viewing angle and distance
    pybullet.debugCam_dist = 1.8
    pybullet.debugCam_yaw = -90
    pybullet.debugCam_pitch = -65

    # objects and collision checking
    # the radius of the region for robot initial position.
    # e.g. 0.5->a circle with radius=0.5 centered at the world frame origin
    pybullet.robotInitRegion_radius = 0.5
    # we define the radius of an entity as the radius of the circle tangent to the entity's xy-plane bounding box
    pybullet.robotRadius = 0.143
    # the original radius of the models
    pybullet.objectsRadius = {'cube': 0.15, 'sphere': 0.10, 'cone': 0.065, 'cylinder': 0.05}
    # self.objectsExpandDistance = {'cube': 0.05, 'sphere': 0.15, 'cone': 0.05, 'cylinder': 0.15} # extend the original radius of models for collision checking
    pybullet.objectsExpandDistance = {'cube': 0.05, 'sphere': 0.05, 'cone': 0.05,
                                      'cylinder': 0.15}  # extend the original radius of models for collision checking
    # Originally, the robot's radius is 0.22. We add this value so that the
    # collision checking radius is 0.22+robotExpandDistance. See robot_locomotors.isCollide()
    pybullet.robotExpandDistance = 0.097

    pybullet.placementExtension = 0.25  # the locations of objects will be more sparse with bigger value

    # robot control
    pybullet.pointFollowerLinearGain = 1
    pybullet.pointFollowerAngularGain = 1
    pybullet.rotPosPGain = 1.5  # P control gain for rotPos control mode
    pybullet.robotWheelDistance = 0.287  # the distance between two wheels
    pybullet.robotWheelRadius = 0.033  # the radius of the wheels
    # Turtlebot3 max transitional velocity=0.26 m/s and max rotational velocity=1.82 rad/s (104.27 deg/s)
    pybullet.robotMaxTransVel = 0.25  # max translational velocity in m/s
    pybullet.robotMinTransVel = -0.1
    pybullet.robotMaxRotVel = 1.1  # max rotational velocity in radian

    # env control
    pybullet.ifReset = True  # if you want to reset the arena after an episode ends
    pybullet.domainRandomization = False  # True if you want to randomize textures for the wall and objects
    pybullet.numTexture = 700  # number of texture to load. Bigger number will take more memory

    # RL env configuration
    pybullet.RLEnvMaxSteps = 80  # the max number of actions (decisions) for an episode. Time horizon N.
    pybullet.RSI_ver = 2  # the version of robot sound interpretation
    RLEnvList = ['TurtleBot-RL-v1', 'TurtleBot-RL-v2']
    pybullet.RLEnvName = RLEnvList[int(pybullet.RSI_ver - 1)]
    pybullet.RLActionDim = (2,)
    pybullet.RLEnvSeed = 66

    # RL robot control
    # velocity: the action will be transitional velocity v and rotational velocity omega
    # rotPos: the action will be transitional velocity v and rotational position. P control is applied
    # debug: move holonomically, (x,y,theta)
    pybullet.RLRobotControl = 'rotPos'

    # RL task configuration
    pybullet.RLManualControl = False
    pybullet.RLTrain = False
    pybullet.RLRealTimePlot = False
    pybullet.RLLogDir = os.path.join('data', 'RL_model', 'TurtleBot')
    pybullet.calcMedoids = False
    pybullet.RLTask = 'approach'
    pybullet.numBatchMedoids = 10  # use numBatchMedoids*pretextTrainBatchSize to approximate medoids
    RLPolicyBaseList = ['mobileRobot_RSI1', 'mobileRobot', ]
    pybullet.RLPolicyBase = RLPolicyBaseList[int(pybullet.RSI_ver - 1)]

    pybullet.realRobot = False

    planner = BaseConfig()
    # the size of a grid
    planner.grid_resolution = 0.25
    # the min distance between robot goal/init pos and any obs is robot.radius * 2
    if planner.grid_resolution >= robot.radius * 2:
        raise ValueError("Increase grid resolution to avoid robot init or goal position being occupied in self.om")
    # unit: number of grids, not meter!!!
    planner.path_clearance = 1
    # After A* generates a path, sample a waypoint every "planner.path_resolution" waypoints
    planner.num_waypoints = int(3)
    # the maximum distance between every 2 waypoints
    planner.max_waypoint_dist = 1.25
    # sample a waypoint at most every k waypoints from A*
    planner.max_waypoint_resolution = int(np.ceil(planner.max_waypoint_dist/planner.grid_resolution))
    # replan every n timesteps
    planner.replan = True
    planner.replan_freq = 30
    planner.om_inludes_human = False

    if sim.predict_method == 'inferred' and env.use_wrapper == False:
        raise ValueError("If using inferred prediction, you must wrap the envs!")
    if sim.predict_method != 'inferred' and env.use_wrapper:
        raise ValueError("If not using inferred prediction, you must NOT wrap the envs!")


class DSRNNPolicy(Policy):
    def __init__(
        self,
        policy_name: str = 'dsrnn_obs_vertex',
        checkpoint_path: Optional[str] = "/home/zhujingqi/MultiAgent/CrowdNavigationMPC/sicnav/policy/training/random_env/DS_RNN/checkpoints/238000.pt",
    ):
        super().__init__()
        # behavior / metadata
        self.kinematics = 'holonomic'
        self.name = policy_name
        self.multiagent_training = False

        # device for any torch-based agent
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.config = Config()
        # ensure the network base selected matches the wrapper name
        self.config.robot.policy = policy_name
        self.max_human_num = 9
        self.max_obs_num = 10
        self.ray_num = 180  # for 360 degree lidar with 2 degree

        self.checkpoint_path = checkpoint_path

        self.observation_space = self.get_observation_space()
        self.action_space = gym.spaces.Box(low=np.array([-1.0, -np.pi]), high=np.array([1.0, np.pi]), dtype=np.float32)
        
        self.load_agent()

    def load_agent(self):
        try:
            # try to instantiate the learned Agent (may warn if ckpt missing)
            self.Agent = NetworkPolicy(
			self.observation_space.spaces,  # pass the Dict into policy to parse
			self.action_space,
			base_kwargs=self.config,
			base=self.config.robot.policy)

            self.agent_available = True
            if self.checkpoint_path:
                self.Agent.load_state_dict(
                    torch.load(self.checkpoint_path, map_location=self.device),
                    strict=False,
                )
            self.Agent.to(self.device)
		    # self.Agent.base.nenv = 1
            
            self.eval_recurrent_hidden_states = {}
            self.eval_masks = None
            self.reset()

        except Exception as e:
            # keep graceful fallback
            print(f"Failed to load agent {self.name}: {e}")
            self.Agent = None
            self.agent_available = False
    
    def configure(self, config):
        pass

    def reset(self):
        num_processes = 1
        config = self.config
        self.eval_recurrent_hidden_states = {}
        if config.robot.policy in ['srnn', 'dsrnn_obs_pc', 'dsrnn_obs_vertex']:
            node_num = 1
            edge_num = self.Agent.base.human_num + 1 + self.Agent.base.obs_num
            self.eval_recurrent_hidden_states['human_node_rnn'] = torch.zeros(num_processes, node_num,
                                                                        config.SRNN.human_node_rnn_size,
                                                                        device=self.device)

            self.eval_recurrent_hidden_states['human_human_edge_rnn'] = torch.zeros(num_processes, edge_num,
                                                                            config.SRNN.human_node_rnn_size,
                                                                            device=self.device)

        else:
            self.eval_recurrent_hidden_states['rnn'] = torch.zeros(num_processes, 1, config.SRNN.human_node_rnn_size,
                                                            device=self.device)
        
        self.eval_masks = torch.zeros(num_processes, 1, device=self.device)

    def predict(self, env_state):
        robot_state = env_state.self_state
        robot_radius = robot_state.radius
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
        """Evaluate the policy model (self.Agent) in multiple testing episodes.

        Parameters:
        self.Agent : torch.nn.Module
            The policy model to evaluate.
        eval_envs : VecEnv
            The vectorized environments for evaluation.
        num_processes : int
            Number of parallel environments to run.
        device : torch.device
            Device for running evaluation (CPU or CUDA).
        config : Config
            Configuration object with environment and training settings.
        logging : logging.Logger
            Logger for evaluation information.
        test_args : argparse.Namespace
            Additional testing arguments like visualization options.
        """

        test_size = self.config.env.test_size

        eval_episode_rewards = []
        config = self.config
        # initialize the RNN hidden states
        num_processes = 1
        
        # robot_node = reshapeT(inputs['robot_node'], seq_length, nenv)
        # temporal_edges = reshapeT(inputs['temporal_edges'], seq_length, nenv)
        # human_states = reshapeT(inputs['spatial_edges'], seq_length, nenv)
        # obs_states = reshapeT(inputs['obstacle_vertices'], seq_length, nenv)
        obs = {}
        obs['robot_node'] = torch.tensor(
            np.array([[0, 0,
                       goal[0] - robot_state.px, goal[1] - robot_state.py,
                       robot_state.theta]]), dtype=torch.float32, device=self.device).unsqueeze(0)
        obs['temporal_edges'] = torch.tensor(
            np.array([[vel[0], vel[1]]]), dtype=torch.float32, device=self.device).unsqueeze(0)
        human_num = len(human_positions)
        if self.config.ob_space.add_human_vel:
            human_states = np.zeros((self.max_human_num, 4), dtype=np.float32)
            for i in range(min(human_num, self.max_human_num)):
                rel_pos = np.array(human_positions[i]) - pos
                human_states[i, 0] = rel_pos[0]
                human_states[i, 1] = rel_pos[1]
                human_states[i, 2] = human_velocities[i][0]
                human_states[i, 3] = human_velocities[i][1]
        else:
            human_states = np.zeros((self.max_human_num, 2), dtype=np.float32)
            for i in range(min(human_num, self.max_human_num)):
                rel_pos = np.array(human_positions[i]) - pos
                human_states[i, 0] = rel_pos[0]
                human_states[i, 1] = rel_pos[1]
        obs['spatial_edges'] = torch.tensor(
            human_states, dtype=torch.float32, device=self.device).unsqueeze(0)
        obs['detected_human_num'] = torch.tensor(
            np.array([[human_num]]), dtype=torch.float32, device=self.device).unsqueeze(0)
        
        obs['obstacle_vertices'] = torch.zeros((1, max(1, self.max_obs_num), 8), dtype=torch.float32, device=self.device)
        obs['obstacle_num'] = torch.tensor(
            np.array([[0]]), dtype=torch.float32, device=self.device).unsqueeze(0)
        obs['point_clouds'] = torch.zeros((1, self.ray_num), dtype=torch.float32, device=self.device).unsqueeze(0)

        with torch.no_grad():
                _, action, _, self.eval_recurrent_hidden_states = self.Agent.act(
                    obs,
                    self.eval_recurrent_hidden_states,
                    self.eval_masks,
                    deterministic=True)
        self.eval_masks = torch.ones(1, 1, device=self.device)
        action_np = action.cpu().numpy()[0]
        return ActionXY(action_np[0], action_np[1])
        

    def get_observation_space(self):

        d={}
        # we use 'absolute' here
        if self.config.ob_space.robot_state == 'absolute':
            # robot px, py (in world frame), and theta (heading angle in z axis)
            d['robot_node'] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1, 5,), dtype=np.float32)
        else:
            # gx-px, gy-py, theta
            d['robot_node'] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,3,), dtype = np.float32)
        # robot vx, vy (in world frame)
        d['temporal_edges'] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1, 2,), dtype=np.float32)
        # make sure there's at least one human
        # add_human_vel is True
        if self.config.ob_space.add_human_vel:
            # [maximum number of humans, human state], where each human state = [human px - robot px, human py - robot py, human vx, human vy]
            # the frame of relative position can be changed in config.ob_space.human_state_frame in configs/config.py
            # the frame of velocity can be changed in in config.ob_space.human_vel in configs/config.py
            d['spatial_edges'] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(max(1, self.max_human_num), 4),
                                                dtype=np.float32)
        else:
            d['spatial_edges'] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(max(1, self.max_human_num), 2),
                                                dtype=np.float32)
        # number of humans detected at each timestep
        d['detected_human_num'] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1, ), dtype=np.float32)

        # obstacle representations methods:
        # Relative coordinates of 4 vertices w.r.t. the robot
        # # [lower left, lower_right, upper_right, upper_left]
        # where lower left = [lower left x  - robot.px, lower left y - robot.py], and same for others
        d['obstacle_vertices'] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(max(1, self.max_obs_num), 8,), dtype=np.float32)

        # number of obstacles
        d['obstacle_num'] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)
        # 3. raw lidar point cloud from robot's 2D lidar
        d['point_clouds'] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1, self.ray_num,), dtype=np.float32)

        return gym.spaces.Dict(d)
