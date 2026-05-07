import casadi as cs
import numpy as np
import logging
from crowd_sim_plus.envs.policy.policy import Policy
from crowd_sim_plus.envs.policy.orca_plus_All import ORCAPlusAll
from crowd_sim_plus.envs.utils.state_plus import FullState, FullyObservableJointState


class NewMPCReal(Policy):
    def __init__(self):
        
        # Environment-related variables
        self.time_step = 0.1  # Time step for MPC
        self.human_max_speed = 1
        
        # MPC-related variables
        self.horizon = 3 # Fixed time horizon
        

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
                                  theta=np.arctan2(hum.vy, hum.vx), radius=hum.radius)                    
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
        x0 = cs.MX([robot_state.px, robot_state.py, robot_state.theta])  # Initial robot state
        # Concatenate vx and vy into a vector, then compute the squared sum
        u_current = cs.vertcat(cs.sumsqr(cs.vertcat(robot_state.vx, robot_state.vy)), robot_state.omega)

        
        # Create Opti object
        opti = cs.Opti()  # CasADi optimization problem
        U_opt = opti.variable(nu_r, self.horizon)  # Decision variables for control inputs

        # Define robot dynamics based on accumulated control inputs
        def dynamics(x0, U):
            states = []
            states.append(x0)
            
            for t in range(self.horizon):
                u_t = U[:, t]
                
                # Update the robot's state using the control input at the current time step
                epsilon = 1e-6
                next_state = states[t] + cs.vertcat(
                    u_t[0] * (cs.sin(states[t][2] + u_t[1] * self.time_step) - cs.sin(states[t][2])) / (u_t[1] + epsilon),
                    u_t[0] * (-cs.cos(states[t][2] + u_t[1] * self.time_step) + cs.cos(states[t][2])) / (u_t[1] + epsilon),
                    u_t[1] * self.time_step
                )

                states.append(next_state)
            
            return states[1:]


        # Get predicted robot states based on control inputs
        X_pred = dynamics(x0, U_opt)
        
        goal_pos = cs.MX([robot_state.gx, robot_state.gy])

        # Step 3: Cost function for goal deviation and control effort
        Q_goal = 300  # Medium priority to reach the goal
        Q_control = 30 # Moderate weight for smooth control inputs
        Q_pref = 10    # Medium preference for stable velocity
        Q_terminal = 300 # Strong weight to reach the goal at the terminal state
        Q_human = 5
 

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
                    control_smooth = cs.sumsqr(U[1, t])# - robot_state.omega)
                
                    current_velocity = cs.vertcat(robot_state.vx, robot_state.vy)
                    control_pref = cs.sumsqr(U[0, t] - cs.sumsqr(current_velocity))
                
                for hum in human_states[t][1:]:
                    human_pos = cs.vertcat(hum[0], hum[1])  # Human's position
                    dist_to_human_sqr = cs.sumsqr(X_pred[t][:2] - human_pos)
                    human_radius = hum[4]  # Human's radius
                    cost -= Q_human*(dist_to_human_sqr - (human_radius + robot_radius + 0.05)**2)  

                cost += Q_goal * dist_to_goal + Q_control * control_smooth + control_pref * Q_pref
            
            # Terminal state goal deviation
            dist_terminal = cs.sumsqr(X_pred[-1][:2] - goal_pos)
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
                    constraints.append(dist_to_human_sqr - (human_radius + robot_radius + 0.08) ** 2)  # Safety margin
            return constraints

        human_constraints = collision_constraint(X_pred, predicted_human_poses[0])

        # Add collision avoidance constraints for humans
        for constr in human_constraints:
            opti.subject_to(constr >= 0)  # No collisions

        # Add static obstacle constraints
        


        total_cost = cost_function(X_pred, U_opt, predicted_human_poses[0])

            
        
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
            return (0,0) # Safe default action

        # Get the optimal control input for the first step
        u_mpc = sol.value(U_opt[:, 0])    
        action = (u_mpc[0], u_mpc[1])
        #action = ActionRot(0, 3.14)

        logging.info(f"Generated action: {action}")
        return action  # Return the optimal control action
