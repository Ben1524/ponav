import matplotlib.pyplot as plt
import matplotlib.animation as animation
import re
import numpy as np
import datetime
import os

num_humans = 10 #env.config num_of_humans + 1
time_horizon = 8 #ORCA plus all time horizon (in the predict funtion) 
plot_x = 3 # environment width 2*plot_x
plot_y = 3 # environment height 2*plot_y

def extract_data_from_file(filename):
    data = []

    with open(filename, 'r') as file:
        for line in file:
            line_data = re.findall(r'[-+]?\d*\.\d+e[-+]?\d+|[-+]?\d*\.\d+|[-+]?\d+', line)
            float_values = [float(value) for value in line_data]
            
            human_positions = [[] for _ in range(num_humans)]
            human_radii = []

            for time_step in range(time_horizon):
                for human_index in range(num_humans):
                    index = (time_step * num_humans + human_index) * 5
                    if index + 4 < len(float_values):
                        px = float_values[index]
                        py = float_values[index + 1]
                        radius = float_values[index + 4]
                        human_positions[human_index].append((px, py))
                        if time_step == 0:
                            human_radii.append(radius)

            data.append((human_positions, human_radii))
    return data

def plot_trajectories(data):
    fig, ax = plt.subplots(figsize=(10, 10))
    colors = plt.cm.hsv(np.linspace(0, 1, num_humans))

    def update(frame):
        ax.clear()
        human_positions, human_radii = data[frame]

        for human_index, positions in enumerate(human_positions):
            x_coords = [pos[0] for pos in positions]
            y_coords = [pos[1] for pos in positions]
            ax.plot(x_coords, y_coords, marker='o', label=f'Human {human_index}', color=colors[human_index])

            for (px, py) in positions:
                radius = human_radii[human_index]
                circle = plt.Circle((px, py), radius, color=colors[human_index], alpha=0.2, fill=True)
                ax.add_artist(circle)

        ax.set_xlabel('X position')
        ax.set_ylabel('Y position')
        ax.set_title('Human Agent Trajectories with Radius')
        ax.legend()
        ax.grid(True)
        ax.axis('equal')

        # Set fixed axis limits
        ax.set_xlim(-plot_x, plot_x)
        ax.set_ylim(-plot_y, plot_y)

    ani = animation.FuncAnimation(fig, update, frames=len(data), repeat=False)

    # Ensure the predictions folder exists
    if not os.path.exists('predictions'):
        os.makedirs('predictions')

    # Create a unique filename with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'predictions/human_trajectories_with_radius_{timestamp}.mp4'
    ani.save(filename, writer='ffmpeg', fps=1)

    plt.show()

# Example usage:
filename = 'output.txt'
data = extract_data_from_file(filename)

# Plot the trajectories and save as MP4
plot_trajectories(data)
