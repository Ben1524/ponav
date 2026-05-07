import casadi as cs
import numpy as np
import logging
# from policy.NewMPCReal import predict

class SelfState:
    def __init__(self, px, py, vx, vy, theta, omega, gx, gy, radius, v_pref):
        # Robot (self) state initialization
        self.px = px  # Position x
        self.py = py  # Position y
        self.vx = vx  # Velocity x
        self.vy = vy  # Velocity y
        self.theta = theta  # Orientation angle
        self.omega = omega  # Angular velocity
        self.gx = gx  # Goal position x
        self.gy = gy  # Goal position y
        self.radius = radius  # Robot's radius
        self.v_pref = v_pref  # Preferred velocity


class HumanState:
    def __init__(self, px, py, vx, vy, theta, omega, gx, gy, radius, v_pref):
        # Human state initialization
        self.px = px  # Position x
        self.py = py  # Position y
        self.vx = vx  # Velocity x
        self.vy = vy  # Velocity y
        self.theta = theta  # Orientation angle
        self.omega = omega  # Angular velocity
        self.gx = gx  # Goal position x
        self.gy = gy  # Goal position y
        self.radius = radius  # Human's radius
        self.v_pref = v_pref  # Preferred velocity


class EnvState:
    def __init__(self, self_state, human_states, static_obs=[]):
        # Environment state initialization
        self.self_state = self_state  # The robot's (self) state
        self.human_states = human_states  # A list of HumanState objects
        self.static_obs = static_obs  # Optional list of static obstacles, each obstacle could be represented as coordinates or an object


def make_sample_state():
    # Create a sample self (robot) state
    self_state = SelfState(px=0, py=0, vx=0, vy=0, theta=0, omega=0, gx=5, gy=5, radius=0.5, v_pref=1.0)

    # Create sample human states
    human1 = HumanState(px=2, py=2, vx=0.1, vy=0.1, theta=0, omega=0, gx=4, gy=4, radius=0.3, v_pref=1.0)
    human2 = HumanState(px=-2, py=-2, vx=-0.1, vy=-0.1, theta=0, omega=0, gx=-4, gy=-4, radius=0.3, v_pref=1.0)

    human_states = [human1, human2]  # Add more humans if needed

    # Create the environment state with the robot and humans
    env_state = EnvState(self_state, human_states)

    return env_state


def main():
    # Generate a sample environment state
    state = make_sample_state()

    # Placeholder for actions (actions_array could store velocity, omega, or other actions for the robot)
    actions_array = []

    # For example, append an action here based on the environment state
    action = None  # Example action as a tuple of velocity and angular velocity
    
    
    
    
    actions_array.append(action)

    print(f"Robot state: px={state.self_state.px}, py={state.self_state.py}")
    print(f"First human state: px={state.human_states[0].px}, py={state.human_states[0].py}")
    print(f"Action taken: {action}")

    return actions_array


if __name__ == '__main__':
    main()
