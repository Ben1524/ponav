import casadi as cs
import numpy as np
import logging
from crowd_sim_plus.envs.policy.policy import Policy
from crowd_sim_plus.envs.policy.orca_plus_All import ORCAPlusAll
from crowd_sim_plus.envs.utils.state_plus import FullState, FullyObservableJointState
from crowd_sim_plus.envs.utils.action import ActionXY

class NewMPCChanging(Policy):

    def __init__(self):
        super().__init__()
        self.trainable = False
        self.kinematics = 'holonomic'
        self.multiagent_training = True
        
        # Environment-related variables
        self.time_step = 0.1  # Time step for MPC
        self.human_max_speed = 1
        
        # MPC-related variables
        self.horiz = 5 # Time horizon
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)

    def predict(self, env_state):
    
        human_states = []
        robot_state = env_state.self_state
        robot_radius = robot_state.radius

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
        time_horizon = 5
        time_step = 0.1
        
        predicted_human_poses = orca_policy.predictAllForTimeHorizon(state, time_horizon)
        #logging.info(f"predicted {predicted_human_poses}")

        # Step 2: Setup MPC using CasADi
        nx_r = 2  # Robot state: [px, py, vx, vy]
        nu_r = 2  # Robot control inputs: [vx, vy]
        
        # Define robot dynamics
        def dynamics(x, u):
            px = x[0]  # x-position
            py = x[1]  # y-position
            

            vx_cmd = u[0]  # control input for x-velocity
            vy_cmd = u[1]  # control input for y-velocity

            next_px = px + vx_cmd * self.time_step  # next x-position
            next_py = py + vy_cmd * self.time_step  # next y-position

            return cs.vertcat(next_px, next_py)  # next state


        x = cs.MX.sym('x', nx_r)  # Current robot state
        u = cs.MX.sym('u', nu_r)  # Robot control input
        
        # Dynamics function
        dynamics_func = cs.Function('dynamics', [x, u], [dynamics(x, u)])


        # Step 3: Cost function for goal deviation
        # Step 3: Cost function for goal deviation and control
        goal_pos = cs.MX([robot_state.gx, robot_state.gy])  # Goal position
        Q_goal = 100    # Weight for goal deviation
        Q_control = 10  # Weight for control inputs

        def cost_function(x):
            dist_to_goal = cs.norm_2(x[:2] - goal_pos)
            #control_cost = cs.norm_2(u)  # Control cost is the magnitude of the control inputs (vx, vy)
            return Q_goal * dist_to_goal #+ Q_control * control_cost

        # Step 4: Collision avoidance constraints (humans)
        def collision_constraint(robot_state, human_states):
            constraints = []
            robot_pos = cs.vertcat(robot_state[0], robot_state[1])  # Robot's position
            
            #logging.info(f"human States: {human_states}")
            
            for hum in human_states:
                human_pos = cs.vertcat(hum[0], hum[1])  # Human's position
                #logging.info(f"human_pos size: {human_pos}")
                
                dist_to_human = cs.norm_2(robot_pos - human_pos)
                #robot_radius = robot_state.radius # Robot's radius
                human_radius = hum[4]  # Human's radius
                #logging.info(f"Constraints {dist_to_human - (human_radius + robot_radius + 0.1)}")
                
                # Collision constraint
                constraints.append(dist_to_human - (human_radius + robot_radius + 0.1))  

            return constraints

        # Step 6: Solve MPC over time horizon
        opti = cs.Opti()  # CasADi optimization problem
        
        # Variables for states and control inputs over the horizon
        X = opti.variable(nx_r, time_horizon + 1)  # States at each time step
        U = opti.variable(nu_r, time_horizon)  # Control inputs at each time step

        # Initial condition
        opti.subject_to(X[:, 0] == [robot_state.px, robot_state.py])

        # Total cost
        total_cost = 0
        
        for t in range(time_horizon):
            # Predict future human states at time t
            future_human_states = predicted_human_poses[0][t][1:]
            opti.subject_to(U[0, t] <= 0.3)  # Upper bound for u[0]
            opti.subject_to(U[0, t] >= -0.3)  # Lower bound for u[0]
            opti.subject_to(U[1, t] <= 0.3)  # Upper bound for u[1]
            opti.subject_to(U[1, t] >= -0.3)  # Lower bound for u[1]
            #logging.info(f"future_human_states: {future_human_states}")
            
            #future_human_states = [[env_state.human_states[0].px,env_state.human_states[0].py,env_state.human_states[0].vx,env_state.human_states[0].vy,env_state.human_states[0].radius]]
            
            # Add goal deviation cost
            total_cost += cost_function(X[:, t])

            
            # Add collision avoidance constraints
            human_constraints = collision_constraint(X[:, t], future_human_states)
            
            
            for constr in human_constraints:
                opti.subject_to(constr >= 0)  # No collisions

            # Add dynamics constraints for the next state
            opti.subject_to(X[:, t + 1] == dynamics_func(X[:, t], U[:, t]))

        # Minimize total cost
        opti.minimize(total_cost)
        
        opti.solver('ipopt', {
            'ipopt.max_iter': 100,        # Increase maximum number of iterations
            'ipopt.tol': 1e-3,             # Increase tolerance to allow faster convergence
            'ipopt.acceptable_tol': 1e-2,  # Less strict acceptance tolerance
            'ipopt.acceptable_iter': 10,   # Accept solution after fewer iterations
            'ipopt.print_level': 0,        # Suppress IPOPT output
            'print_time': False
        })

      
        try:
            sol = opti.solve()
        except RuntimeError as e:
            #logging.error(f"Solver failed with error: {e}")
            #logging.info(f"Initial state: {opti.debug.value(X[:, 0])}")
            #logging.info(f"Control input: {opti.debug.value(U[:, 0])}")
            return ActionXY(predicted_human_poses[0][0][0][2],predicted_human_poses[0][0][0][3])  
            #return ActionXY(robot_state.vx/2, robot_state.vy/2) # Return a safe default action in case of failure
            #return ActionXY(0,0)

        # Get the optimal control input for the first step
        u_mpc = sol.value(U[:, 0])        
        # Generate action
        
        logging.info(f"Generated {sol.value(U)}")
        logging.info(f"Generated {u_mpc}")
        
        action = ActionXY(u_mpc[0], u_mpc[1])
        #action = ActionXY(0,1)
        logging.info(f"Generated action: {action}")

        return action  # Return the optimal control action