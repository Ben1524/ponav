import logging
import numpy as np
from crowd_sim_plus.envs.policy.orca import *


class ORCAPlusAll(ORCA):
    """
    Template to implement a teleoperation.
    This placeholder is a copy of crowd_sim_plus.envs.policy.linear
    """

    def __init__(self):
        super().__init__()

    def configure(self, config, section='orca_plus'):
    
        self.time_step = 0.1# Default time step
        try:
            self.time_step = config.getfloat('env', 'time_step')
        except:
            logging.warn("[ORCA_PLUS POLICY] problem with policy config")
        # self.neighbor_dist = config.getfloat('human', 'neighbor_dist')
        # self.max_neighbors = config.getint('orca', 'max_neighbors')
        # self.time_horizon = config.getfloat('orca', 'time_horizon')
        # self.time_horizon_obst = config.getfloat('orca', 'time_horizon_obst')
        self.radius = config.getfloat(section, 'radius')
        self.safety_space = config.getfloat(section, 'safety_space')
        # self.max_speed = config.getfloat(section, 'safety_space')
        return

    def predict(self, state):
        """
        Create an RVO2 simulation at each time step and run one step.
        ORCAPlus extends the original ORCA Policy by reading static obstacles from the state 
        and adding them to the simulation (via the Python-RVO2 API).
        """
        self_state = state.self_state
        params = self.neighbor_dist, self.max_neighbors, self.time_horizon, self.time_horizon_obst
        #logging.info('[NisalaOrca] Config {:} = {:}'.format('Running ORCA', self_state.radius))

        # Initialize or reset simulation if necessary
        if self.sim is not None and self.sim.getNumAgents() != len(state.human_states) + 1:
            del self.sim
            self.sim = None
        if self.sim is None:
            self.sim = rvo2.PyRVOSimulator(self.time_step, *params, self.radius, self.max_speed)
            for i, line in enumerate(state.static_obs):
                self.sim.addObstacle(line)
            if state.static_obs:
                self.sim.processObstacles()

            # Add the robot agent
            self.sim.addAgent(self_state.position, *params, self_state.radius + 0.01 + self.safety_space,
                              self_state.v_pref, self_state.velocity)

            # Add human agents
            for human_state in state.human_states:
                self.sim.addAgent(human_state.position, *params, human_state.radius + 0.01 + self.safety_space,
                                  self.max_speed, human_state.velocity)
        else:
            self.sim.setAgentPosition(0, self_state.position)
            self.sim.setAgentVelocity(0, self_state.velocity)
            for i, human_state in enumerate(state.human_states):
                self.sim.setAgentPosition(i + 1, human_state.position)
                self.sim.setAgentVelocity(i + 1, human_state.velocity)

        # Set the preferred velocity toward the goal
        velocity = np.array((self_state.gx - self_state.px, self_state.gy - self_state.py))
        speed = np.linalg.norm(velocity)
        epsilon = 1e-3
        pref_vel = velocity / speed * (self_state.v_pref - epsilon) if speed > (self_state.v_pref - epsilon) else velocity

        self.sim.setAgentPrefVelocity(0, tuple(pref_vel))

        # Set preferred velocities for human agents (assuming unknown goals for humans)
        for i, human_state in enumerate(state.human_states):
            self.sim.setAgentPrefVelocity(i + 1, (0, 0))

        # Perform one simulation step
        self.sim.doStep()
        
        # Retrieve the action for the robot (agent 0)
        action = ActionXY(*self.sim.getAgentVelocity(0))
        self.last_state = state

        # Simulate for the time horizon
        predicted_trajectories = self.predictAllForTimeHorizon(state, 8)
        
        # Open the file in append mode ('a')
        # Convert list to string before writing To visualize orca path prediction
        with open('output.txt', 'a') as file:
            file.write(str(predicted_trajectories[0]) + '\n')  # Convert list to string and append new line

        #logging.info(f"predicted_trajectories{predicted_trajectories[0]}")
        
        

        self.sim = None  # Reset for SB3 saving

        return action
        
    def predictAll(self, state):
        
        """
        Function to get action array for robot and Humans using ORCA
        """
        self.time_step = 0.1
        
        self_state = state.self_state
        params = self.neighbor_dist, self.max_neighbors, self.time_horizon, self.time_horizon_obst
        #logging.info('[NisalaOrca] Config {:} = {:}'.format('Running ORCA', self_state.radius))
        
        #logging.info("time_step: %s, params: %s, radius: %s, max_speed: %s", self.time_step, params, self.radius, self.max_speed)


        if self.sim is not None and self.sim.getNumAgents() != len(state.human_states) + 1:
            del self.sim
            self.sim = None
        if self.sim is None:
            self.sim = rvo2.PyRVOSimulator(self.time_step, *params, self.radius, self.max_speed)
            for i, line in enumerate(state.static_obs):
                self.sim.addObstacle(line)
            if state.static_obs:
                self.sim.processObstacles()

            self.sim.addAgent(self_state.position, *params, self_state.radius + 0.01 + self.safety_space,
                              self_state.v_pref, self_state.velocity)
            for human_state in state.human_states:
                self.sim.addAgent(human_state.position, *params, human_state.radius + 0.01 + self.safety_space,
                                  self.max_speed, human_state.velocity)
        else:
            #logging.info("Have a simulation")
            self.sim.setAgentPosition(0, self_state.position)
            self.sim.setAgentVelocity(0, self_state.velocity)
            for i, human_state in enumerate(state.human_states):
                self.sim.setAgentPosition(i + 1, human_state.position)
                self.sim.setAgentVelocity(i + 1, human_state.velocity)

        # Set the preferred velocity to be a vector of unit magnitude (speed) in the direction of the goal.
        velocity = np.array((self_state.gx - self_state.px, self_state.gy - self_state.py))
        speed = np.linalg.norm(velocity)
        epsilon = 1e-3
        pref_vel = velocity / speed * (self_state.v_pref - epsilon) if speed > (self_state.v_pref - epsilon) else velocity

        # Perturb a little to avoid deadlocks due to perfect symmetry.
        # perturb_angle = np.random.random() * 2 * np.pi
        # perturb_dist = np.random.random() * 0.01
        # perturb_vel = np.array((np.cos(perturb_angle), np.sin(perturb_angle))) * perturb_dist
        # pref_vel += perturb_vel

        self.sim.setAgentPrefVelocity(0, tuple(pref_vel))
        for i, human_state in enumerate(state.human_states):
            # unknown goal position of other humans
            self.sim.setAgentPrefVelocity(i + 1, (0, 0))

        self.sim.doStep()
        
        
        action = ActionXY(*self.sim.getAgentVelocity(0))
        
        action_array = []
        
        #logging.info('[NisalaOrca] Config {:} = {:}'.format('RobotPositionX', self_state.px))
        #logging.info('[NisalaOrca] Config {:} = {:}'.format('RobotPositionY', self_state.py))
        #logging.info('[NisalaOrca] Config {:} = {:}'.format('RobotVelX', self_state.vx))
        #logging.info('[NisalaOrca] Config {:} = {:}'.format('RobotVelY', self_state.vy))
        
        for i  in range(len(state.human_states)+1):
            action1 = (self.sim.getAgentVelocity(i))
            action_array.append(action1)

        self.last_state = state
        self.sim = None # need this for SB3 saving

        return action_array
        
    def predictAllForTimeHorizon(self, state, time_horizon):
        """
        Function to get action array for robot and Humans using ORCA for all the time horizon.
        Outputs: Array(size(time_horizon, num_agents, state)) with format for the state ((self.px, self.py, self.vx, self.vy, self.radius))
        """
        predicted_states = []
        predicted_actions = []

        current_state = state
        
        
        
        
        """
        #For plotting the initial state of all agents. If you use this code segment change the time horizon of Plot.py to time_horizon + 1
        
        cur_state = []
        cur_action = []

        # Store the updated robot state (without goal info, since it may be ObservableState)
        robot_current_state = (current_state.self_state.px, current_state.self_state.py, current_state.self_state.vx, current_state.self_state.vy, current_state.self_state.radius)
        cur_state.append(robot_current_state)
        rob_action = (current_state.self_state.vx, current_state.self_state.vy)
        cur_action.append(rob_action)

        # Update human states based on the predicted actions
        for i, hum in enumerate(current_state.human_states):
            # Update velocities dynamically at each time step
            human_current_state = (hum.px, hum.py, hum.vx, hum.vy, hum.radius)
            cur_state.append(human_current_state)
            hum_action = (hum.vx, hum.vy)
            cur_action.append(hum_action)
            
        predicted_states.append(cur_state)
        predicted_actions.append(cur_action)
        """
        
        #logging.info(f"human state_current {current_state.human_states[0]}")
        


        for t in range(time_horizon):
            # Get the actions for all agents at the current timestep using ORCA
            actions = self.predictAll(current_state)
            #logging.info(f"currentState {current_state.self_state.vx, current_state.self_state.vy}")
            #logging.info(f"actions {actions}")

            next_state = []

            # Update the robot state based on the predicted action
            robot_action = actions[0]
            robot_next_px = current_state.self_state.px + robot_action[0] * self.time_step
            robot_next_py = current_state.self_state.py + robot_action[1] * self.time_step
            robot_next_vx = robot_action[0]
            robot_next_vy = robot_action[1]

            # Store the updated robot state (without goal info, since it may be ObservableState)
            robot_next_state = (robot_next_px, robot_next_py, robot_next_vx, robot_next_vy, current_state.self_state.radius)
            next_state.append(robot_next_state)

            # Update human states based on the predicted actions
            for i, human_action in enumerate(actions[1:]):
                human_state = current_state.human_states[i]
                human_next_px = human_state.px + human_action[0] * self.time_step
                human_next_py = human_state.py + human_action[1] * self.time_step
                human_next_vx = human_action[0]
                human_next_vy = human_action[1]

                # Update velocities dynamically at each time step
                human_next_state = (human_next_px, human_next_py, human_next_vx, human_next_vy, human_state.radius)
                next_state.append(human_next_state)

            # Store the predicted state and actions for this timestep
            predicted_states.append(next_state)
            predicted_actions.append(actions)

            # Update the current state to be the next state for the next iteration
            current_state = self.update_state_with_predicted(current_state, next_state)
            
        #logging.info(f"agents {predicted_states}")
        
        #logging.info(f"{predicted_states}")
            
        return predicted_states, predicted_actions


    
    def update_state_with_predicted(self, current_state, predicted_state):
        """
        Update the current state with the predicted state. This function is needed to keep track of the new state
        after applying the ORCA predictions, so the subsequent predictions use the updated state.
        
        """
        #logging.info(f"Predicted State {predicted_state}")
        # Update the robot state
        current_state.self_state.px = predicted_state[0][0]
        current_state.self_state.py = predicted_state[0][1]
        current_state.self_state.vx = predicted_state[0][2]
        current_state.self_state.vy = predicted_state[0][3]
        current_state.self_state.theta = np.arctan2(predicted_state[0][3], predicted_state[0][2])
        #current_state.self_state.radius = current_state.self_state.radius
        #current_state.self_state.gx = current_state.self_state.gx
        #current_state.self_state.gy = current_state.self_state.gy

        # Update human states
        for i in range(len(current_state.human_states)):
            current_state.human_states[i].px = predicted_state[i+1][0]
            current_state.human_states[i].py = predicted_state[i+1][1]
            current_state.human_states[i].vx = predicted_state[i+1][2]
            current_state.human_states[i].vy = predicted_state[i+1][3]
            current_state.human_states[i].theta = np.arctan2(predicted_state[i+1][3], predicted_state[i+1][2])
            #current_state.human_states[i].radius = current_state.self_state.radius

            

        return current_state
