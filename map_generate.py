from scipy.ndimage import label  # Use to check connectivity
import numpy as np
import random

def create_random_shape_map(map_size, target_area, num_of_obstacles):
    '''Create a random map with a specified area and number of obstacles'''
    # Start with a fully open map
    map = np.ones(map_size)  # Fully blocked map (1 = obstacle, 0 = free)
    
    # Randomly open cells to form the desired area while ensuring connectivity
    open_cells = set()
    initial_pos = (np.random.randint(1, map_size[0]-1), np.random.randint(1, map_size[1]-1))
    open_cells.add(initial_pos)
    map[initial_pos] = 0  # Initial position is open space

    while len(open_cells) < target_area: # Keep expanding until the desired area is reached
        # Pick a random open cell and expand around it
        current_cell = random.choice(list(open_cells))
        neighbors = get_neighbors(current_cell, map_size)

        for neighbor in neighbors:
            if map[neighbor] == 1:  # Only open blocked cells
                map[neighbor] = 0
                open_cells.add(neighbor)
            if len(open_cells) == target_area:
                break

    # Add random obstacles while ensuring the map remains fully connected
    obstacle_count = 0
    while obstacle_count < num_of_obstacles:
        candidate_pos = (np.random.randint(1, map_size[0]-1), np.random.randint(1, map_size[1]-1))
        
        if map[candidate_pos] == 0:  # Only add obstacles to open spaces
            temp_map = map.copy()
            temp_map[candidate_pos] = 1  # Tentatively place obstacle
            
            # Check if the map is still fully connected after adding the obstacle
            labeled_map, num_features = label(temp_map == 0)  # Find connected components
            if num_features == 1:
                map[candidate_pos] = 1  # Confirm the obstacle
                obstacle_count += 1

    return map

def get_neighbors(cell, map_size):
    neighbors = []
    x, y = cell
    if x > 1: neighbors.append((x-1, y))  # Up
    if x < map_size[0] - 2: neighbors.append((x+1, y))  # Down
    if y > 1: neighbors.append((x, y-1))  # Left
    if y < map_size[1] - 2: neighbors.append((x, y+1))  # Right
    return neighbors
