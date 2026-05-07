import casadi as cs
import numpy as np
import logging
from crowd_sim_plus.envs.policy.policy import Policy
from crowd_sim_plus.envs.policy.orca_plus_All import ORCAPlusAll
from crowd_sim_plus.envs.utils.state_plus import FullState, FullyObservableJointState
from crowd_sim_plus.envs.utils.action import ActionXY, ActionRot

class NewMPC(Policy):
    def __init__(self):
        super().__init__()
        self.trainable = False
        self.kinematics = 'unicycle'
        self.multiagent_training = True
        
        # Environment-related variables
        self.time_step = 0.25 # Time step for MPC
        self.human_max_speed = 1
        
        # MPC-related variables
        self.horizon = 10 # Fixed time horizon
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)

    def predict(self, env_state):
        human_states = []
        robot_state = env_state.self_state
        robot_radius = robot_state.radius
        
        if robot_state.omega == None:
            robot_state.omega = 0

        for hum in env_state.human_states:               
            gx = hum.px + hum.vx * 2
            gy = hum.py + hum.vy * 2
            hum_state = FullState(px=hum.px, py=hum.py, vx=hum.vx, vy=hum.vy, 
                                  gx=gx, gy=gy, v_pref=self.human_max_speed, 
                                  theta=cs.atan2(hum.vy, hum.vx), radius=hum.radius)                    
            human_states.append(hum_state)

        state = FullyObservableJointState(self_state=robot_state, human_states=human_states, static_obs=env_state.static_obs)
        
        # Step 1: Predict future human positions over the time horizon using ORCA
        orca_policy = ORCAPlusAll()
        predicted_human_poses = orca_policy.predictAllForTimeHorizon(env_state, self.horizon)
        
        #logging.info(f"predict {predicted_human_poses}")

        # Step 2: Setup MPC using CasADi
        nx_r = 3  # Robot state: [px, py, theta]
        nu_r = 2  # Robot control inputs: [v, omega]
        
        # Initial state and current velocity
        logging.info(f"robot position {robot_state.px, robot_state.py, robot_state.gx, robot_state.gy}")
        x0 = cs.MX([robot_state.px, robot_state.py, robot_state.theta])  # Initial robot state
        # Concatenate vx and vy into a vector, then compute the squared sum
        u_current = cs.vertcat(cs.sumsqr(cs.vertcat(robot_state.vx, robot_state.vy)), robot_state.omega)
        logging.info(f"u_current {u_current}")

        
        # Create Opti object
        opti = cs.Opti()  # CasADi optimization problem
        U_opt = opti.variable(nu_r, self.horizon)  # Decision variables for control inputs

        # Define robot dynamics based on accumulated control inputs
        def dynamics(x0, U):
            states = []
            states.append(x0)
            
            for t in range(self.horizon):
                u_t = U[:,t]
                
                # Update the robot's state using the control input at the current time step
                epsilon = 1e-6
                next_state = states[t] + cs.vertcat(
                    u_t[0] * (cs.cos(states[t][2] + u_t[1] * self.time_step)),# + cs.cos(states[t][2])) / (u_t[1] + epsilon),
                    u_t[0] * (cs.sin(states[t][2] + u_t[1] * self.time_step)), #- cs.sin(states[t][2])) / (u_t[1] + epsilon),
                    u_t[1] * self.time_step
                )

                states.append(next_state)
            
            return states[1:]


        # Get predicted robot states based on control inputs
        X_pred = dynamics(x0, U_opt)
        
        goal_pos = cs.MX([robot_state.gx, robot_state.gy])

        # Step 3: Cost function for goal deviation and control effort
        Q_goal = 200 # Medium priority to reach the goal
        Q_smooth = 30 # Moderate weight for smooth control inputs
        Q_pref = 30    # Medium preference for stable velocity
        Q_terminal = 1500 # Strong weight to reach the goal at the terminal state
        Q_human = 1
 

        def cost_function(X_pred, U, human_states):
            cost = 0
            for t in range(self.horizon):
                dist_to_goal = cs.sumsqr(X_pred[t][:2] - goal_pos)  # Distance to the goal

                # Penalize control inputs (mainly smoothness in omega)
                if t > 0:
                    # Penalize change in omega between consecutive time steps
                    control_smooth = cs.sumsqr(U[1, t])# - U[1, t - 1])
                    control_pref = cs.sumsqr(U[0, t] - U[0, t-1])  # Prefer certain velocity
                else:
                    control_smooth = cs.sumsqr(U[1, t])                
                    current_velocity = cs.vertcat(robot_state.vx, robot_state.vy)
                    control_pref = cs.sumsqr(U[0, t] - cs.sumsqr(current_velocity))
                
                for hum in human_states[t][1:]:
                    human_pos = cs.vertcat(hum[0], hum[1])  # Human's position
                    dist_to_human_sqr = cs.sumsqr(X_pred[t][:2] - human_pos)
                    human_radius = hum[4]  # Human's radius
                    cost = Q_human*(1/(dist_to_human_sqr - (human_radius + robot_radius + 0.02)**2))

                cost += Q_goal * dist_to_goal + Q_smooth * control_smooth + control_pref * Q_pref
            
            # Terminal state goal deviation
            dist_terminal = cs.sumsqr(X_pred[0][:2] - goal_pos)
            cost += Q_terminal * dist_terminal
            return cost


        # Step 4: Collision avoidance constraints (humans)
        def collision_constraint(X_pred, human_states):
            constraints = []
            for t in range(self.horizon):
                robot_pos = X_pred[t][:2]
                for hum in human_states[t][1:]:
                    human_pos = cs.vertcat(hum[0], hum[1])  # Human's position
                    dist_to_human_sqr = cs.sumsqr(robot_pos - human_pos)
                    human_radius = hum[4]  # Human's radius
                    constraints.append(dist_to_human_sqr - (human_radius + robot_radius + 0.02) ** 2)  # Safety margin
            return constraints

        human_constraints = collision_constraint(X_pred, predicted_human_poses[0])

        # Add collision avoidance constraints for humans
        for constr in human_constraints:
            opti.subject_to(constr >= 0)  # No collisions

        # Add static obstacle constraints
        def static_obstacle_constraint(X_pred, static_obs):
            constraints = []
            
            for step in range(self.horizon):
                robot_pos = cs.vertcat(X_pred[step][0], X_pred[step][1])  # Robot's position at step

                for obs in static_obs:
                    start_pos = cs.vertcat(obs[0][0], obs[0][1])  # Start position of the line segment
                    end_pos = cs.vertcat(obs[1][0], obs[1][1])    # End position of the line segment

                    obs_vec = end_pos - start_pos
                    robot_to_start = robot_pos - start_pos

                    # Parametric distance along the segment (dot product projection)
                    t_param = cs.dot(robot_to_start, obs_vec) / cs.dot(obs_vec, obs_vec)
                    t_clamped = cs.fmax(0, cs.fmin(1, t_param))  # Clamp t_param to [0, 1]

                    # Closest point on the line segment to the robot
                    closest_point = start_pos + t_clamped * obs_vec

                    # Euclidean distance to the closest point
                    dist_to_obstacle = cs.norm_2(robot_pos - closest_point)

                    # Safety margin for robot radius
                    safety_margin = 0.08
                    constraints.append(dist_to_obstacle - (robot_radius + safety_margin))

            return constraints


        total_cost = cost_function(X_pred, U_opt, predicted_human_poses[0])

        # Add static obstacle constraints
        static_constraints = static_obstacle_constraint(X_pred, env_state.static_obs)
        for constr in static_constraints:
            opti.subject_to(constr >= 0)  # No collisions with static obstacles
            
        
        # Add control bounds
        opti.subject_to(U_opt[0, :] <= 1)  # Upper bound for v
        opti.subject_to(U_opt[0, :] >= 0)  # Lower bound for v
        

        # Minimize total cost
        opti.minimize(total_cost)

        

        # Set up the solver
        opti.solver('ipopt', {
            'ipopt.max_iter': 500,
            'ipopt.tol': 1e-3,
            'ipopt.acceptable_tol': 1e-2,
            'ipopt.acceptable_iter': 10,
            'ipopt.print_level': 0,
            'print_time': False
        })

        # Solve the optimization problem
        try:
            sol = opti.solve()
            
        except RuntimeError as e:
            logging.error(f"Solver failed with error: {e}")
            
            return ActionRot(0,0)  # Safe default action

        # Get the optimal control input for the first step
        u_mpc = sol.value(U_opt[:, 0])    
        logging.info(f"cost {sol.value(total_cost) }")
        logging.info(f"U_opt {sol.value(U_opt) }")
        logging.info(f"theta {robot_state.theta}")
        next_px = (robot_state.px + u_mpc[0]*np.cos(robot_state.theta+u_mpc[1]*self.time_step))
        logging.info(f"next_px {dynamics([robot_state.px, robot_state.py, robot_state.theta],sol.value(U_opt))}")
        
        action = ActionRot(u_mpc[0], u_mpc[1]*self.time_step)
        #action = ActionRot(0, 3.14/3)

        logging.info(f"Generated action: {action}")
        return action  # Return the optimal control action