import numpy as np
import matplotlib.pyplot as plt
import os
import gym
from gym import spaces
from map_generate import create_random_shape_map
import MapBuilder

'''
    Observation Space: 
        - FOV (Field of View) around the car's position
        - Car's position in the map
        - Visit count of each grid
    Action Space:
        - Up, Down, Left, Right
    Reward:
        - Penalty for colliding with obstacles (-1)
        - Reward for exploring a new area (0.01)
        - Penalty for revisiting a previously explored area (-0.01 * visit count)
        - Movement Penalty (-0.1)
        - Stationary Penalty (-0.5)
        - Big reward for completing exploration (5)
'''

# Constants
obstacle = 1
free_space = 0
car = 2
explored = -1

class DummyGym(gym.Env):
    def __init__(self, init_pos=(2, 3), car_size=(1,1), step_size=1, map_size=(10, 10), num_of_obstacles=5, FOV=(5, 5), see_through=False, map_file_path=None, is_slippery=False):
        super(DummyGym, self).__init__()
        self.car_size = car_size
        self.step_size = step_size
        self.map_size = map_size
        self.num_of_obstacles = num_of_obstacles
        self.init_pos = init_pos if init_pos else self._place_car()
        self.FOV = FOV
        self.see_through = see_through
        self.is_slippery = is_slippery
        self.car_pos = self.init_pos
        
        # Create the map
        self.map = self._create_map(map_file_path)
        
        # Initialize visit counts for each grid
        self.visit_count = np.zeros(self.map_size)
        
        # Action space: 4 discrete actions (Up, Down, Left, Right)
        self.action_space = spaces.Discrete(4)  # 0: Up, 1: Down, 2: Left, 3: Right
        
        # Observation space: 2D map (grayscale FOV), visit counts
        self.observation_space = spaces.Box(low=0, high=1, shape=FOV, dtype=np.float32)
        
        # Initial state
        self.state = self._observe()

    def _create_map(self, map_file_path):
        if map_file_path:
            if not os.path.exists(map_file_path):
                raise ValueError("Invalid map file path!")
            map = np.loadtxt(map_file_path)
            return map
        else:
            return MapBuilder.MapBuilder(self.map_size[0], self.map_size[1])

    def _place_car(self):
        while True:
            self.car_pos = (np.random.randint(0, self.map_size[0] - self.car_size[0] + 1), 
                            np.random.randint(0, self.map_size[1] - self.car_size[1] + 1))
            is_empty = True
            for i in range(self.car_size[0]):
                for j in range(self.car_size[1]):
                    if self.map[self.car_pos[0] + i, self.car_pos[1] + j] != 0:
                        is_empty = False
                        break
                if not is_empty:
                    break
            if is_empty:
                break

    # Get FOV grid locations
    def _get_fov_grids_location(self):
        x = self.car_pos[0]
        y = self.car_pos[1]
        fov_x_min = max(0, x - self.FOV[0] // 2)
        fov_x_max = min(self.map_size[0], x + self.FOV[0] // 2)
        fov_y_min = max(0, y - self.FOV[1] // 2)
        fov_y_max = min(self.map_size[1], y + self.FOV[1] // 2)
        fov_grids_location = [(i, j) for i in range(fov_x_min, fov_x_max) for j in range(fov_y_min, fov_y_max)]
        return fov_x_min, fov_x_max, fov_y_min, fov_y_max, fov_grids_location

    def _observe(self):
        fov_x_min, fov_x_max, fov_y_min, fov_y_max, fov_grids_location = self._get_fov_grids_location()
        self.fov_map = self.map[fov_x_min:fov_x_max, fov_y_min:fov_y_max]
        return self.visit_count, self.fov_map, self.car_pos

    # Step function for interacting with the environment
    def step(self, action):
        old_pos = self.car_pos

        # Map actions to movement
        if action == 0:  # Up
            new_pos = (max(self.car_pos[0]-self.step_size, 0), self.car_pos[1])
        elif action == 1:  # Down
            new_pos = (min(self.car_pos[0]+self.step_size, self.map_size[0]-1), self.car_pos[1])
        elif action == 2:  # Left
            new_pos = (self.car_pos[0], max(self.car_pos[1]-self.step_size, 0))
        elif action == 3:  # Right
            new_pos = (self.car_pos[0], min(self.car_pos[1]+self.step_size, self.map_size[1]-1))
        else:
            raise ValueError(f"Invalid action: {action}")

        reward = 0
        done = False

        # Collision check
        if self.map[new_pos] == 1:  # Obstacle collision
            reward -= 1
            new_pos = old_pos
        else:
            reward -= 0.1  # Movement penalty

        # Get FOV grid locations and update visit counts
        _, _, _, _, fov_grids_location = self._get_fov_grids_location()
        for grid_location in fov_grids_location:
            if self.visit_count[grid_location] == 0:  # New grid
                reward += 0.01
            else:
                reward -= 0.01 * self.visit_count[grid_location]  # Penalty for revisiting
            self.visit_count[grid_location] += 1

        # Update car position
        self.car_pos = new_pos

        # Update state
        self.state = self._observe()

        # Check if map is fully explored
        if np.all(self.visit_count != 0):
            reward += 5  # Big reward for completing exploration
            done = True

        return self.state, reward, done, {}

    # Reset environment
    def reset(self):
        self._place_car()
        self.visit_count = np.zeros(self.map_size)  # Reset visit counts
        self.state = [self.visit_count, self.get_fov_map()]  # Reset state
        return self.state

    # Render the environment
    def render(self, map_type):
        if map_type == 'visit_count':
            display_map = np.copy(self.visit_count)
        elif map_type == 'fov_map':
            display_map = np.copy(self.fov_map)
        elif map_type == 'Map':
            display_map = np.copy(self.map)
        display_map[self.car_pos] = car
        plt.imshow(display_map, cmap='gray', interpolation='none')
        plt.title(map_type)
        plt.colorbar()
        plt.show()

# Example usage:
env = DummyGym(init_pos=(2,3), map_size=(30,30), num_of_obstacles=140, FOV=(5,5))
env.render('Map')
env.render('fov_map')
env.render('visit_count')
print(env.action_space.n)

# Perform a step
env.step(1)

# Render the maps after the step
env.render('Map')
env.render('fov_map')
env.render('visit_count')
