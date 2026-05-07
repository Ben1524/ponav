import numpy as np
import cv2
import matplotlib.pyplot as plt

def extract_obstacles_from_slam_map(map_image, threshold=127):
    """
    Extracts static obstacles from a SLAM map and returns them as line segments,
    including both outer and inner walls.
    
    Parameters:
    - map_image: The input map in grayscale where obstacles are represented (e.g., black for obstacles, white for free space).
    - threshold: Pixel intensity threshold to differentiate between obstacles and free space.
    
    Returns:
    - line_segments: A list of tuples representing the line segments in the form [((start_x, start_y), (end_x, end_y)), ...]
    """
    
    # Apply threshold to extract obstacle areas (assumes binary or grayscale map)
    _, binary_map = cv2.threshold(map_image, threshold, 255, cv2.THRESH_BINARY_INV)
    
    # Optionally, use edge detection to emphasize edges (can help detect inner walls)
    edges = cv2.Canny(binary_map, 50, 150)
    
    # Detect contours with hierarchy to capture inner and outer walls
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    
    line_segments = []
    
    # Loop through each detected contour and use points directly
    for contour in contours:
        # Loop through all points in the contour
        for i in range(len(contour) - 1):
            start_point = tuple(contour[i][0])
            end_point = tuple(contour[i + 1][0])
            line_segments.append((start_point, end_point))
    
    return line_segments

def visualize_line_segments(line_segments, map_image=None):
    """
    Visualizes the extracted line segments using matplotlib.
    
    Parameters:
    - line_segments: List of tuples representing the line segments in the form [((start_x, start_y), (end_x, end_y)), ...]
    - map_image: (Optional) The original map image to plot as the background.
    """
    plt.figure(figsize=(8, 8))
    
    # Plot the original map image as background (optional)
    if map_image is not None:
        plt.imshow(map_image, cmap='gray')
    
    # Plot each line segment
    for line in line_segments:
        (x1, y1), (x2, y2) = line
        plt.plot([x1, x2], [y1, y2], color='red', linewidth=2)

    plt.title("Visualized Line Segments from SLAM Map")
    plt.gca().invert_yaxis()  # Invert Y-axis to match image coordinates
    plt.show()

# Example usage:
if __name__ == "__main__":
    # Load SLAM map (grayscale image with obstacles in black and free space in white)
    map_image = cv2.imread("slam_map.png", cv2.IMREAD_GRAYSCALE)
    
    # Extract static obstacles as line segments
    obstacles = extract_obstacles_from_slam_map(map_image)
    
    # Visualize the extracted line segments
    visualize_line_segments(obstacles, map_image)
