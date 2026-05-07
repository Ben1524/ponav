import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import re

# Load future predictions from the extracted file
with open('extracted_futures.txt', 'r') as file:
    future_data = file.readlines()

# Function to parse positions from the text file
def parse_future_positions(data):
    predictions_per_cycle = []
    future_regex = r"Future \[(.*?)\]"
    
    for line in data:
        match = re.findall(future_regex, line)
        if match:
            positions = eval(match[0])  # Convert string to list of tuples
            predictions_per_cycle.append(positions)
    return predictions_per_cycle

# Parse the future positions from the text file
predictions_per_cycle = parse_future_positions(future_data)

# Prepare the figure and axis limits
fig, ax = plt.subplots()
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)

# Plot objects for the humans' trajectories (lines) and start positions (dots)
human_trajectories = [ax.plot([], [], label=f'Human {i+1}')[0] for i in range(5)]
human_start_points = [ax.plot([], [], 'o', label=f'Start {i+1}')[0] for i in range(5)]

# Initialize the animation
def init():
    for traj, start in zip(human_trajectories, human_start_points):
        traj.set_data([], [])
        start.set_data([], [])
    return human_trajectories + human_start_points

# Update the animation: plot all timesteps at once, and mark start positions
def update(frame):
    # Clear the previous trajectories and start points
    for traj, start in zip(human_trajectories, human_start_points):
        traj.set_data([], [])
        start.set_data([], [])
    
    # Plot all future predictions for the current time horizon at once
    time_horizon = 10  # Time horizon to plot all predictions
    cycle_start = frame * time_horizon
    if cycle_start + time_horizon > len(predictions_per_cycle):
        return human_trajectories + human_start_points  # End the animation if we run out of cycles
    
    for i in range(4):  # For each human
        # Get the predicted trajectory for the time horizon
        trajectory = [(predictions_per_cycle[t][i][0], predictions_per_cycle[t][i][1]) 
                      for t in range(cycle_start, cycle_start + time_horizon)]
        x_vals, y_vals = zip(*trajectory)  # Separate x and y coordinates
        
        # Plot the full trajectory
        human_trajectories[i].set_data(x_vals, y_vals)
        
        # Mark the start position as a dot
        human_start_points[i].set_data(x_vals[0], y_vals[0])

    return human_trajectories + human_start_points

# Create the animation, iterating over time horizons
num_cycles = len(predictions_per_cycle) // 10 # Number of time horizons
ani = FuncAnimation(fig, update, frames=num_cycles, init_func=init, blit=True, repeat=True)

# Show the plot
plt.legend()
plt.xlabel('X position')
plt.ylabel('Y position')
plt.title('Predicted Human Trajectories with Start Positions')
plt.show()
