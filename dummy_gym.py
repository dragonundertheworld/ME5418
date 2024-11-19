'''
    Observation Space: 
        - FOV (Field of View) around the car's position
        - Car's position in the map
        - Visit count of each grid
    Action Space:
        - Up, Down, Left, Right
    Reward:
        - Penalty for colliding with obstacles (-1)
        - Reward for exploring a new area (0.01 * visit count)
        - Penalty for revisiting a previously explored area (-0.01 * visit count)
        - Movement Penalty (-0.1)
        - Big reward for completing exploration (5)
'''
import numpy as np
import matplotlib.pyplot as plt
import os
import gym
from gym import spaces
from MapBuilder import MapBuilder
import math
import unittest

# Map grid values
OBSTACLE   = 0
UNEXPLORED = 1
CAR        = 2
UNKNOWN    = -1

# Initial values
INIT_POS         = (7, 14)
CAR_SIZE         = (1, 1)
CAR_MATRIX       = np.full(CAR_SIZE, CAR) # car matrix is a 10x10 matrix with 2s
STEP_SIZE        = 1
MAP_SIZE         = (10, 10)
NUM_OF_OBSTACLES = 10
NUM_OF_RAYS      = 50
FOV              = (3, 3)
FIGURE_SIZE      = (12, 8)

# Rewards and penalties
COLLISION_PENALTY    = -1
EXPLORATION_REWARD   = 0.01
REVISIT_PENALTY      = -0.01
MOVEMENT_PENALTY     = -0.1
FINISH_REWARD        = 5

# MAP_PATH = './test_map/map_2024-10-17_19-06-07.txt'
MAP_PATH = './test_map/smaller_map.txt'
# MAP_PATH = './test_map/smaller_map_2024-11-04_19-06-31.txt'

class Car:
    """
    Car class represents a vehicle with specific size, step size, and field of view (FOV).
    It maintains its position and automatically updates its bounds based on the position and size.

    Attributes:
        size (tuple): The size of the car.
        step_size (int): The step size for the car's movement.
        fov (tuple): The field of view of the car.
        _pos (tuple): The internal storage for the car's position.
        up_bound (int): The upper bound of the car based on its position and size.
        down_bound (int): The lower bound of the car based on its position and size.
        left_bound (int): The left bound of the car based on its position and size.
        right_bound (int): The right bound of the car based on its position and size.
        up_bound_fov (int): The upper bound of the car's field of view.
        down_bound_fov (int): The lower bound of the car's field of view.
        left_bound_fov (int): The left bound of the car's field of view.
        right_bound_fov (int): The right bound of the car's field of view.

    Methods:
        pos:
            Property getter for the car's position.
        pos(new_pos):
            Property setter for the car's position. Automatically updates bounds when position changes.
        _update_bounds():
            Updates the bounds based on the current position and size.
    """
    def __init__(self):
        self.size = CAR_SIZE
        self.step_size = STEP_SIZE
        self.fov = FOV
        self._pos = INIT_POS  # use an underscore to indicate internal storage
        self._update_bounds()
        self.up_bound_under_fov = self.fov[0]//2-self.size[0]//2 # car's upper bound under fov(10*10 ) coordinate
        self.down_bound_under_fov = self.fov[0]//2+self.size[0]//2
        self.left_bound_under_fov = self.fov[1]//2-self.size[1]//2
        self.right_bound_under_fov = self.fov[1]//2+self.size[1]//2

    @property
    def pos(self):
        return self._pos

    @pos.setter
    def pos(self, new_pos):
        self._pos = new_pos
        self._update_bounds()  # automatically update bounds when pos changes

    def _update_bounds(self):
        '''
        Update bounds based on current position and size in the map
        '''
        # Update bounds based on current position and size
        self.up_bound = self._pos[0] - math.floor(self.size[0] / 2)
        self.down_bound = self._pos[0] + math.ceil(self.size[0] / 2)
        self.left_bound = self._pos[1] - math.floor(self.size[1] / 2)
        self.right_bound = self._pos[1] + math.ceil(self.size[1] / 2)

        # Update FOV bounds
        self.up_bound_fov = self._pos[0] - math.floor(self.fov[0] / 2) # 存疑
        self.down_bound_fov = self._pos[0] + math.ceil(self.fov[0] / 2)
        self.left_bound_fov = self._pos[1] - math.floor(self.fov[1] / 2)
        self.right_bound_fov = self._pos[1] + math.ceil(self.fov[1] / 2)


class DummyGym(gym.Env):
    """
    DummyGym is a custom OpenAI Gym environment for simulating a car navigating a map with obstacles.
    Attributes:
        car (Car): The car object navigating the map.
        map_size (tuple): The size of the map (height, width).
        num_of_obstacles (int): The number of obstacles in the map.
        see_through (bool): Whether the map is see-through.
        is_slippery (bool): Whether the map has slippery surfaces.
        map (np.ndarray): The map of the environment.
        fov_map (np.ndarray): The field of view map.
        visit_count (np.ndarray): The visit counts for each grid in the map.
        action_space (gym.spaces.Discrete): The action space (4 discrete actions: Up, Down, Left, Right).
        observation_space (gym.spaces.Box): The observation space (2D map and visit counts).
        state (tuple): The current state of the environment.
    Methods:
        __init__(self, car=Car(), map_size=MAP_SIZE, num_of_obstacles=NUM_OF_OBSTACLES, see_through=False, map_file_path=None, is_slippery=False):
            Initializes the DummyGym environment.
        _create_map(self, map_file_path):
            Creates the map for the environment.
        _place_car(self):
            Places the car in the map without overlapping with obstacles.
        _get_fov_grids_location(self):
            Gets the field of view grid locations.
        observe(self):
            Observes the current state of the environment.
        step(self, action):
            Executes a step in the environment based on the given action.
        calculate_reward_and_done(self):
            Calculates the reward and checks if the episode is done.
        reset(self):
            Resets the environment to the initial state.
        render(self, map_type):
            Renders the environment based on the given map type.
    """
    
    def __init__(self, car=Car(), map_size=MAP_SIZE, 
                num_of_obstacles=NUM_OF_OBSTACLES, see_through=False, map_file_path=MAP_PATH, is_slippery=False):
        super(DummyGym, self).__init__()
        self.car              = car
        self.user_map_size    = map_size
        self.num_of_obstacles = num_of_obstacles
        self.see_through      = see_through
        self.is_slippery      = is_slippery
        
        # Create the map
        self.map = self._create_map(map_file_path)
        self.map_size = self.map.shape

        # Place the car in the map
        self._place_car()

        # Initialize fov_map
        self.fov_map = np.zeros(self.car.fov)
        self.detect_map = np.zeros(self.car.fov)
        
        # Initialize visit counts for each grid
        self.visit_count = np.zeros(self.map.shape)
        
        # Action space: 4 discrete actions (Up, Down, Left, Right)
        self.action_space = spaces.Discrete(4)  # 0: Up, 1: Down, 2: Left, 3: Right
        
        # Observation space: 2D map (grayscale fov), visit counts
        self.observation_space = spaces.Box(low=0, high=1, shape=self.car.fov, dtype=np.float32)
        
        # Initial state
        self.state = self.observe()

    def _create_map(self, map_file_path):
        if map_file_path:
            if not os.path.exists(map_file_path):
                raise ValueError("Invalid map file path!")
            map = np.loadtxt(map_file_path)
            self.map_size = map.shape
            return map
        else:
            occupancy = self.num_of_obstacles / (self.user_map_size[0] * self.user_map_size[1])
            return MapBuilder(self.user_map_size[0], self.user_map_size[1], occupancy, self.car.size[0])

    def _place_car(self):
        if np.any(self.map[self.car.up_bound:self.car.down_bound, self.car.left_bound:self.car.right_bound]==OBSTACLE): # np.any() returns True if any element is non-zero
            # Randomly place the car in the map without car_size overlapping with obstacles
            while True:
                self.car.pos = (np.random.randint(math.ceil(self.car.size[0]/2), math.ceil(self.map_size[0]-self.car.size[0]/2)), 
                                np.random.randint(math.ceil(self.car.size[1]/2), math.ceil(self.map_size[1]-self.car.size[1]/2)))
                # print(self.car.down_bound, self.car.right_bound)
                if not np.any(self.map[self.car.up_bound:self.car.down_bound, self.car.left_bound:self.car.right_bound]==OBSTACLE):
                    break

    # Get fov grid locations
    def _get_fov_grids_location(self):
        """
        Calculate the field of view (FOV) grid locations for the car within the map boundaries.
        This method determines the grid locations that fall within the car's field of view (FOV) 
        while considering the boundaries of the map. It ensures that the FOV does not exceed the 
        map limits and adjusts the FOV grid locations accordingly.
        Returns:
            tuple: A tuple containing the following elements:
            - fov_x_min_in_map (int): Minimum x-coordinate of the FOV in the map.
            - fov_x_max_in_map (int): Maximum x-coordinate of the FOV in the map.
            - fov_y_min_in_map (int): Minimum y-coordinate of the FOV in the map.
            - fov_y_max_in_map (int): Maximum y-coordinate of the FOV in the map.
            - fov_x_min_in_fov (int): Minimum x-coordinate of the FOV in the FOV grid.
            - fov_x_max_in_fov (int): Maximum x-coordinate of the FOV in the FOV grid.
            - fov_y_min_in_fov (int): Minimum y-coordinate of the FOV in the FOV grid.
            - fov_y_max_in_fov (int): Maximum y-coordinate of the FOV in the FOV grid.
            - fov_grids_location (list): List of tuples representing the grid locations within the FOV.
        """
        up_bound_of_map           = 0
        down_bound_of_map         = self.map_size[0]
        left_bound_of_map         = 0
        right_bound_of_map        = self.map_size[1]

        up_bound_of_fov_in_fov    = 0
        down_bound_of_fov_in_fov  = self.car.fov[0]
        left_bound_of_fov_in_fov  = 0
        right_bound_of_fov_in_fov = self.car.fov[1]

        fov_grids_location = []
        
        # fov_x and fov_y should not be negative if the car is close to the edge of the map
        if self.car.up_bound_fov >= up_bound_of_map:
            fov_x_min_in_map = self.car.up_bound_fov
            fov_x_min_in_fov = up_bound_of_fov_in_fov
        else:
            fov_x_min_in_map = up_bound_of_map
            fov_x_min_in_fov = -self.car.up_bound_fov

        if self.car.down_bound_fov < down_bound_of_map:
            fov_x_max_in_map = self.car.down_bound_fov
            fov_x_max_in_fov = down_bound_of_fov_in_fov
        else:
            fov_x_max_in_map = down_bound_of_map
            fov_x_max_in_fov = down_bound_of_fov_in_fov - (self.car.down_bound_fov - down_bound_of_map)

        if self.car.left_bound_fov >= left_bound_of_map:
            fov_y_min_in_map = self.car.left_bound_fov
            fov_y_min_in_fov = left_bound_of_fov_in_fov
        else:
            fov_y_min_in_map = left_bound_of_map
            fov_y_min_in_fov = -self.car.left_bound_fov

        if self.car.right_bound_fov < right_bound_of_map:
            fov_y_max_in_map = self.car.right_bound_fov
            fov_y_max_in_fov = right_bound_of_fov_in_fov
        else:
            fov_y_max_in_map = right_bound_of_map
            fov_y_max_in_fov = right_bound_of_fov_in_fov - (self.car.right_bound_fov - right_bound_of_map)

        for i in range(fov_x_min_in_map, fov_x_max_in_map): # range(1, 6)= 1, 2, 3, 4, 5
            for j in range(fov_y_min_in_map, fov_y_max_in_map):
                fov_grids_location.append((i, j)) # Append the grid location to fov_grids_location
        return fov_x_min_in_map, fov_x_max_in_map, fov_y_min_in_map, fov_y_max_in_map, fov_x_min_in_fov, fov_x_max_in_fov, fov_y_min_in_fov, fov_y_max_in_fov, fov_grids_location

    def observe(self):
        fov_x_min_in_map, fov_x_max_in_map, fov_y_min_in_map, fov_y_max_in_map, fov_x_min_in_fov, fov_x_max_in_fov, fov_y_min_in_fov, fov_y_max_in_fov, _ = self._get_fov_grids_location()
        fov_map = np.zeros(self.car.fov)
        # Reflect the map in the FOV map
        fov_map[fov_x_min_in_fov:fov_x_max_in_fov,
                 fov_y_min_in_fov:fov_y_max_in_fov] = self.map[fov_x_min_in_map:fov_x_max_in_map, 
                                                               fov_y_min_in_map:fov_y_max_in_map]
        self.fov_map = fov_map
        self.detect_environment(self.fov_map, num_rays=NUM_OF_RAYS)
        # print(self.detect_map)
        return self.visit_count, self.fov_map, self.car.pos
    
    def save_state(self, path):
        np.savez(path, visit_count=self.visit_count, fov_map=self.fov_map, car_pos=self.car.pos)

    def load_state(self, path):
        data = np.load(path)
        self.visit_count = data['visit_count']
        self.fov_map = data['fov_map']
        self.car.pos = tuple(data['car_pos'])
        self.state = self.observe()

    def detect_environment(self, environment, num_rays=NUM_OF_RAYS):
        """
        Simulates a LIDAR scan with adjustable precision.
        
        :param environment: 2D numpy array representing the environment (1: free space, 0: obstacle, 2: LIDAR).
        :param num_rays: Number of rays to cast (higher number means higher resolution, e.g. 360 for 1-degree precision).
        :return: 2D numpy array with detected areas marked as 1, obstacles as 0, and unknown areas as -1.
        """
        def is_valid(r, c, rows, cols):
            """Check if a cell is within the bounds of the matrix."""
            return 0 <= r < rows and 0 <= c < cols
        
        # Get the dimensions of the environment
        rows, cols = self.fov_map.shape
        
        # Initialize the detected environment with unknown areas (-1)
        self.detect_map = np.full((rows, cols), UNKNOWN)
        
        # Find the position of the LIDAR (2)
        car_row, car_col = self.fov_map.shape[0]//2, self.fov_map.shape[1]//2      
        
        # Cast rays in 360 degrees (or a user-defined number of directions)
        for i in range(num_rays):
            # Convert the ray index to an angle in radians
            angle = (i / num_rays) * 2 * math.pi
            
            # Calculate the direction vector for this angle
            dir_r = math.sin(angle)  # Change in row (y-direction)
            dir_c = math.cos(angle)  # Change in column (x-direction)
            
            # Initialize the starting point (LIDAR position)
            r, c = float(car_row), float(car_col)
            
            # Trace the ray step by step
            while True:
                # Move along the ray direction
                r += dir_r
                c += dir_c
                
                # Convert floating point positions to integer grid positions
                int_r, int_c = int(round(r)), int(round(c))
                
                # Stop if the ray goes out of bounds
                if not is_valid(int_r, int_c, rows, cols):
                    break
                
                # Stop if the ray hits an obstacle
                if environment[int_r, int_c] == OBSTACLE:
                    self.detect_map[int_r, int_c] = OBSTACLE
                    break
                
                # Mark the cell as visible in the detected environment
                self.detect_map[int_r, int_c] = UNEXPLORED
        
    # Step function for interacting with the environment
    def step(self, action):
        old_pos = self.car.pos
        self.COLLISION_FLAG = False
        # self.SLIPPERY_FLAG = False

        detect_up_bound = self.car.up_bound
        detect_down_bound = self.car.down_bound
        detect_left_bound = self.car.left_bound
        detect_right_bound = self.car.right_bound

        # Map actions to movement
        if action == 0:  # Up
            movement = "Up"
            new_pos = (self.car.pos[0]-self.car.step_size, self.car.pos[1])
            self.car.pos = new_pos
            # update one of the collision detection bounds
            detect_up_bound = self.car.up_bound
        elif action == 1:  # Down
            movement = "Down"
            new_pos = (self.car.pos[0]+self.car.step_size, self.car.pos[1])
            self.car.pos = new_pos
            detect_down_bound = self.car.down_bound
        elif action == 2:  # Left
            movement = "Left"
            new_pos = (self.car.pos[0], self.car.pos[1]-self.car.step_size)
            self.car.pos = new_pos
            detect_left_bound = self.car.left_bound
        elif action == 3:  # Right
            movement = "Right"
            new_pos = (self.car.pos[0], self.car.pos[1]+self.car.step_size)
            self.car.pos = new_pos
            detect_right_bound = self.car.right_bound
        else:
            raise ValueError(f"Invalid action: {action}")
        
        # Abnormal flags
        if (np.any(self.map[detect_up_bound:detect_down_bound, 
                            detect_left_bound:detect_right_bound] == OBSTACLE) or
            detect_up_bound < 0 or
            detect_down_bound > self.map_size[0] or
            detect_left_bound < 0 or
            detect_right_bound > self.map_size[1]):
            self.COLLISION_FLAG = True
            self.car.pos = old_pos
            # print("Collision! Car stays in the same position: ", self.car.pos)
        else:
            if self.is_slippery:
                if np.random.rand() < 0.2: # Slip
                    self.SLIPPERY_FLAG = True
                    self.car.pos = old_pos
                    # print("Slippery! Car stays in the same position: ", new_pos)
                else:
                    self.SLIPPERY_FLAG = False
                    print(f"Car moves {self.step_size} from {old_pos} to {new_pos}")
            else:
                self.SLIPPERY_FLAG = False
                # print(f"Car moves {movement} {self.car.step_size} units from {old_pos} to {new_pos}")

        reward, done = self.calculate_reward_and_done()

        # Update state
        self.state = self.observe()
        return self.state, reward, done, {}

    def calculate_reward_and_done(self):
        reward = 0
        done = False

        # Abnormal flags handling
        reward = reward + COLLISION_PENALTY if self.COLLISION_FLAG == True else reward + MOVEMENT_PENALTY # Collision and movement penalty

        # Get FOV grid locations and update visit counts
        _, _, _, _, _, _, _, _, fov_grids_location = self._get_fov_grids_location()
        for grid_location in fov_grids_location:
            if self.visit_count[grid_location] == UNEXPLORED:  # New grid
                reward += EXPLORATION_REWARD
            else:
                reward += REVISIT_PENALTY  # Penalty for revisiting
            self.visit_count[grid_location] += 1

        # Check if map is fully explored
        if np.all(self.visit_count != 0):
            reward += FINISH_REWARD  # Big reward for completing exploration
            done = True
        return reward,done

    # Reset environment
    def reset(self):
        self.car = Car()
        self.fov_map = np.zeros(self.car.fov)
        self.visit_count = np.zeros(self.map_size)
        self.state = self.observe()
        return self.state

    # Render the environment
    def render(self, map_type=None):
        if map_type == None: # Render all maps at once
            plt.figure(figsize=FIGURE_SIZE)
            plt.suptitle("Dummy Gym Environment", fontsize=16)
            plt.subplot(2, 2, 1)
            map1 = np.copy(self.map)
            map1[self.car.up_bound:self.car.down_bound, 
                        self.car.left_bound:self.car.right_bound] = CAR
            plt.title("Map")
            plt.imshow(map1, cmap='gray', interpolation='none')
            plt.colorbar()

            plt.subplot(2, 2, 2)
            map2 = np.copy(self.fov_map)
            map2[self.car.up_bound_under_fov:self.car.down_bound_under_fov, 
                        self.car.left_bound_under_fov:self.car.right_bound_under_fov] = CAR
            plt.title("FOV Map")
            plt.imshow(map2, cmap='gray', interpolation='none')
            plt.colorbar()

            plt.subplot(2, 2, 3)
            map3 = np.copy(self.detect_map)
            map3[self.car.up_bound_under_fov:self.car.down_bound_under_fov,
                        self.car.left_bound_under_fov:self.car.right_bound_under_fov] = CAR
            plt.title("Laser detect Map")
            plt.imshow(map3, cmap='gray', interpolation='none')
            plt.colorbar()

            plt.subplot(2, 2, 4)
            map4 = np.copy(self.visit_count)
            map4[self.car.up_bound:self.car.down_bound, 
                        self.car.left_bound:self.car.right_bound] = CAR
            plt.title("Visit Count")
            plt.imshow(map4, cmap='gray', interpolation='none')
            plt.colorbar()

            plt.show()
        else: # Render a specific map
            display_map = {"visit_count": self.visit_count, "fov_map": self.fov_map, "Map": self.map, "detect_map": self.detect_map}[map_type]
            display_map = np.copy(display_map)
            if map_type == "visit_count" or map_type == "Map":
                display_map[self.car.up_bound:self.car.down_bound, 
                            self.car.left_bound:self.car.right_bound] = CAR  # Update car position in map
                # MapBuilder.save_and_show_map(display_map, map_type)
            elif map_type == "fov_map" or map_type == "detect_map":
                display_map[self.car.up_bound_under_fov:self.car.down_bound_under_fov, 
                            self.car.left_bound_under_fov:self.car.right_bound_under_fov] = CAR  # Update car position in FOV map

            plt.title(map_type)
            plt.imshow(display_map, cmap='gray', interpolation='none')
            plt.colorbar()
            plt.show()
        

# Uncomment the following code to perform an example usage:
# env = DummyGym() 
# # env.load_state('test.npz')
# env.render() # render all maps at once
# print(env.action_space.n)

# # Perform a step
# env.step(0) # Move up

# # Render the maps after the step
# env.render()

# # Perform another step and render
# env.step(2) # Move left
# # env.save_state('test.npz')
# env.render()

# env.step(1) # Move right
# env.render()

# env.step(3)
# env.render()