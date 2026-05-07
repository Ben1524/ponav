import matplotlib.pyplot as plt
import re

num_humans = 10
time_horizon = 10

def extract_data_from_file(filename):

    data = []

    # Open and read the file line by line
    with open(filename, 'r') as file:
        for line in file:
            # Extracting future predictions including scientific notation
            line_data = re.findall(r'[-+]?\d*\.\d+e[-+]?\d+|[-+]?\d*\.\d+|[-+]?\d+', line)  # Updated regex
            
            # Convert the string values to floats
            float_values = [float(value) for value in line_data]
            
            # Extract positions for each human agent
            human_positions = [[] for _ in range(num_humans)]  # Assuming 5 humans
            
            for time_step in range(time_horizon):  # 5 time steps
                for human_index in range(num_humans):  # 5 humans
                    index = (time_step * num_humans + human_index) * 5  # Each human has 5 values: (px, py, vx, vy, radius)
                    if index + 4 < len(float_values):  # Ensure there are enough values
                        px = float_values[index]
                        py = float_values[index + 1]
                        human_positions[human_index].append((px, py))

            data.append(human_positions)
    return data

def plot_trajectories(data):
    plt.figure(figsize=(10, 10))
    
    # Initialize lists to store each agent's trajectory
  # Assuming there are always 5 humans

    # Plotting loop for dynamic updates
    for human_positions in data:
        plt.clf()  # Clear the current figure

        # Plot each human's trajectory
        colors = plt.colormaps['tab10']
        for human_index, positions in enumerate(human_positions):
            x_coords = [pos[0] for pos in positions]
            y_coords = [pos[1] for pos in positions]
            plt.plot(x_coords, y_coords, marker='o', label=f'Human {human_index}', color=colors(human_index))

        # Plot customization
        plt.xlabel('X position')
        plt.ylabel('Y position')
        plt.title('Human Agent Trajectories')
        plt.legend()
        plt.grid(True)
        plt.axis('equal')  # Keep the aspect ratio square

        plt.pause(1)  # Pause for a second to visualize the current frame

    plt.show()

# Example usage:
filename = 'output.txt'
data = extract_data_from_file(filename)

# Plot the trajectories dynamically
plot_trajectories(data)
