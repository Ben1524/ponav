import casadi as cs
import numpy as np
import logging
from crowd_sim_plus.envs.policy.policy import Policy
from crowd_sim_plus.envs.policy.orca_plus_All import ORCAPlusAll
from crowd_sim_plus.envs.utils.state_plus import FullState, FullyObservableJointState
from crowd_sim_plus.envs.utils.action import ActionXY, ActionRot

class NewMPCChanging(Policy):

    def __init__(self):
        super().__init__()
        self.trainable = False
        #self.kinematics = 'unicycle'
        self.kinematics = 'holonomic'
        self.multiagent_training = True
        
        # Environment-related variables
        self.time_step = 0.1  # Time step for MPC
        self.human_max_speed = 1
        
        # MPC-related variables
        self.horizon = 2 # Fixed time horizon
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)

    def predict(self, env_state):
        human_states = []
        robot_state = env_state.self_state
        robot_radius = robot_state.radius
        x0 = [robot_state.px, robot_state.py]

        for hum in env_state.human_states:               
            gx = hum.px + hum.vx * 2
            gy = hum.py + hum.vy * 2
            hum_state = FullState(px=hum.px, py=hum.py, vx=hum.vx, vy=hum.vy, 
                                  gx=gx, gy=gy, v_pref=self.human_max_speed, 
                                  theta=np.arctan2(hum.vy, hum.vx), radius=hum.radius)                    
            human_states.append(hum_state)

        state = FullyObservableJointState(self_state=robot_state, human_states=human_states, static_obs=env_state.static_obs)
        
        # Step 1: Predict future human positions over the time horizon using ORCA
        orca_policy = ORCAPlusAll()
        predicted_human_poses = orca_policy.predictAllForTimeHorizon(env_state, self.horizon)
        
        #logging.info(f"predict {predicted_human_poses}")

        # Step 2: Setup MPC using CasADi
        nx_r = 2  # Robot state: [px, py]
        nu_r = 2  # Robot control inputs: [vx, vy]
        
        # Initial state and current velocity
        #x0 = cs.MX([robot_state.px, robot_state.py])  # Initial robot state
        u_current = cs.MX([robot_state.vx, robot_state.vy])  # Current robot velocity

        # Create Opti object
        opti = cs.Opti()  # CasADi optimization problem
        U_opt = opti.variable(nu_r, self.horizon)  # Decision variables for control inputs

        # Define robot dynamics based on accumulated control inputs
        def dynamics(x0, u_current, U):
            states = []
            states.append(x0)
            
            for t in range(self.horizon):
                # Use U at each time step directly instead of summing them
                u_t = U[:, t]
                
                # Update the robot's state using the control input at the current time step
                next_state = states[t] + u_t * self.time_step
                states.append(next_state)
            
            return states[1:]

        # Get predicted robot states based on control inputs
        X_pred = dynamics(x0, u_current, U_opt)

        # Step 3: Cost function for goal deviation and control effort
        goal_pos = cs.MX([robot_state.gx, robot_state.gy])  # Goal position
        Q_terminal = 100
        Q_goal = 100   # Weight for goal deviation
        Q_control = 30 # Weight for control inputs
        Q_pref = 50
        Q_human = 1

        def cost_function(X_pred, U, human_states):
            cost = 0
            for t in range(self.horizon):
                dist_to_goal = cs.sumsqr(X_pred[t] - goal_pos)  # Final state to goal
                if t == 0:
                    control_smooth = cs.sum2(cs.sumsqr(U[:, t]-u_current)) # Sum of control efforts
                else:
                    control_smooth = cs.sum2(cs.sumsqr(U[:, t]-U[:, t-1]))
                    
                control_pref = cs.sum2(cs.sumsqr(U[:, t])-0.5)
                
                for hum in human_states[t][1:]:
                    human_pos = cs.vertcat(hum[0], hum[1])  # Human's position
                    dist_to_human_sqr = cs.sumsqr(X_pred[t] - human_pos)
                    human_radius = hum[4]  # Human's radius
                    cost -= Q_human*(dist_to_human_sqr - (human_radius + robot_radius + 0.01)**2)  # Neede to changed
                cost += Q_goal * dist_to_goal + Q_control * control_smooth + control_pref*Q_pref
            dist_terminal = cs.sumsqr(X_pred[-1] - goal_pos)  # Final state to goal
            cost += Q_terminal * dist_terminal
            return cost
            



        # Step 4: Collision avoidance constraints (humans)
        def collision_constraint(X_pred, human_states):
            constraints = []
            for t in range(self.horizon):
                robot_pos = X_pred[t]
                for hum in human_states[t][1:]:
                    human_pos = cs.vertcat(hum[0], hum[1])  # Human's position
                    dist_to_human_sqr = cs.sumsqr(robot_pos - human_pos)
                    human_radius = hum[4]  # Human's radius
                    constraints.append(dist_to_human_sqr - (human_radius + robot_radius + 0.01)**2)  # Safety margin
            return constraints

        human_constraints = collision_constraint(X_pred, predicted_human_poses[0])

        # Add collision avoidance constraints for humans
        for constr in human_constraints:
            opti.subject_to(constr >= 0)  # No collisions

        # Add static obstacle constraints
        def static_obstacle_constraint(X_pred, static_obs):
            constraints = []
            
            for t in range(self.horizon):
                robot_state = X_pred[t]
                robot_pos = cs.vertcat(robot_state[0], robot_state[1])  # Robot's position

                for obs in static_obs:
                    start_pos = cs.vertcat(obs[0][0], obs[0][1])  # Start position of the line segment
                    end_pos = cs.vertcat(obs[1][0], obs[1][1])  # End position of the line segment

                    # Vector from start to end of the line segment
                    obs_vec = end_pos - start_pos
                    # Vector from start to robot position
                    robot_to_start = robot_pos - start_pos

                    # Project the robot position onto the line segment and clamp between [0, 1]
                    t = cs.dot(robot_to_start, obs_vec) / cs.dot(obs_vec, obs_vec)
                    t_clamped = cs.fmax(0, cs.fmin(1, t))  # Clamps t to the segment bounds

                    # Find the closest point on the segment
                    closest_point = start_pos + t_clamped * obs_vec

                    # Calculate the distance from the robot to the closest point on the segment
                    dist_to_obstacle = cs.norm_2(robot_pos - closest_point)

                    # Add collision constraint to maintain a safe distance
                    safety_margin = 0.01  # Safety margin for robot radius
                    constraints.append(dist_to_obstacle - (robot_radius + safety_margin))

            return constraints
            
        total_cost = cost_function(X_pred, U_opt,predicted_human_poses[0])

        # Add static obstacle constraints
        static_constraints = static_obstacle_constraint(X_pred, env_state.static_obs)
        for constr in static_constraints:
            opti.subject_to(constr >= 0)  # No collisions with static obstacles


        # Add control bounds
        opti.subject_to(U_opt[0, :] <= 0.5)  # Upper bound for vx
        opti.subject_to(U_opt[0, :] >= -0.5)  # Lower bound for vx
        opti.subject_to(U_opt[1, :] <= 0.5)  # Upper bound for vy
        opti.subject_to(U_opt[1, :] >= -0.5)  # Lower bound for vy

        
        # Minimize total cost
        opti.minimize(total_cost)

        opti.solver('ipopt', {
            'ipopt.max_iter': 100,
            'ipopt.tol': 1e-3,
            'ipopt.acceptable_tol': 1e-3,
            'ipopt.acceptable_iter': 10,
            'ipopt.print_level': 0,
            'print_time': False
        })

        try:
            sol = opti.solve()
        except RuntimeError as e:
            logging.error(f"Solver failed with error: {e}")
            return ActionXY(0,0)
            #return ActionRot(0,0)
            #return ActionXY(robot_state.vx / 2, robot_state.vy / 2)  # Return a safe default action

        # Get the optimal control input for the first step
        u_mpc = sol.value(U_opt[:, 0])        
        action = ActionXY(u_mpc[0], u_mpc[1])
        #action = ActionRot(np.sqrt(u_mpc[0]**2+ u_mpc[1]**2),np.arctan2(u_mpc[1],u_mpc[0])-robot_state.theta)

        logging.info(f"Generated action: {action}")
        return action  # Return the optimal control action
