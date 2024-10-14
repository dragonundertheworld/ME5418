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

# Map grid values
OBSTACLE   = 1
UNEXPLORED = 0
CAR        = 2

# Initial values
INIT_POS         = (12, 10)
CAR_SIZE         = (7, 7)
STEP_SIZE        = 5
MAP_SIZE         = (40, 21)
NUM_OF_OBSTACLES = 10
FOV              = (10, 10)

# Rewards and penalties
COLLISION_PENALTY    = -1
EXPLORATION_REWARD   = 0.01
REVISIT_PENALTY      = -0.01
MOVEMENT_PENALTY     = -0.1
STATIONARY_PENALTY   = -0.5
FINISH_REWARD        = 5

class DummyGym(gym.Env):
    """
    DummyGym is a custom OpenAI Gym environment for simulating a car navigating a grid map with obstacles.
    Attributes:
        car_size (tuple): Size of the car in the grid.
        step_size (int): Number of steps the car moves in one action.
        map_size (tuple): Size of the grid map.
        num_of_obstacles (int): Number of obstacles in the map.
        init_pos (tuple): Initial position of the car.
        fov (tuple): Field of view size.
        see_through (bool): Whether obstacles are see-through.
        is_slippery (bool): Whether the environment is slippery.
        car_pos (tuple): Current position of the car.
        map (ndarray): The grid map.
        fov_map (ndarray): The field of view map.
        visit_count (ndarray): Visit counts for each grid cell.
        action_space (gym.spaces.Discrete): Action space (4 discrete actions: Up, Down, Left, Right).
        observation_space (gym.spaces.Box): Observation space (2D map and visit counts).
        state (tuple): Current state of the environment.
    Methods:
        __init__(self, init_pos, car_size, step_size, map_size, num_of_obstacles, fov, see_through, map_file_path, is_slippery):
            Initializes the DummyGym environment.
        _create_map(self, map_file_path):
            Creates the grid map.
        _place_car(self):
            Randomly places the car in the map without overlapping obstacles.
        _get_fov_grids_location(self):
            Gets the grid locations within the field of view.
        _observe(self):
            Observes the current state of the environment.
        step(self, action):
            Executes a step in the environment based on the given action.
        calculate_reward_and_done(self):
            Calculates the reward and checks if the episode is done.
        reset(self):
            Resets the environment to the initial state.
        render(self, map_type):
            Renders the environment.
    """
    def __init__(self, init_pos=(2, 3), car_size=(1,1), step_size=1, map_size=(10, 10), num_of_obstacles=5, fov=(5, 5), see_through=False, map_file_path=None, is_slippery=False):
        super(DummyGym, self).__init__()
        self.car_size         = car_size
        self.step_size        = step_size
        self.map_size         = map_size
        self.num_of_obstacles = num_of_obstacles
        self.init_pos         = init_pos if init_pos else self._place_car()
        self.fov              = fov
        self.see_through      = see_through
        self.is_slippery      = is_slippery
        self.car_pos          = self.init_pos
        
        # Create the map
        self.map = self._create_map(map_file_path)

        # Initialize fov_map
        self.fov_map = np.zeros(self.fov)
        
        # Initialize visit counts for each grid
        self.visit_count = np.zeros(self.map_size)
        
        # Action space: 4 discrete actions (Up, Down, Left, Right)
        self.action_space = spaces.Discrete(4)  # 0: Up, 1: Down, 2: Left, 3: Right
        
        # Observation space: 2D map (grayscale fov), visit counts
        self.observation_space = spaces.Box(low=0, high=1, shape=fov, dtype=np.float32)
        
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
        # Randomly place the car in the map without car_size overlapping with obstacles
        while True:
            car_pos = (np.random.randint(0, self.map_size[0]), np.random.randint(0, self.map_size[1]))
            if np.all(self.map[car_pos[0]:car_pos[0]+self.car_size[0], car_pos[1]:car_pos[1]+self.car_size[1]] == 0):
                self.car_pos = car_pos
                break

    # Get fov grid locations
    def _get_fov_grids_location(self):
        x = self.car_pos[0]
        y = self.car_pos[1]

        up_bound_of_fov_in_map    = x - self.fov[0] // 2
        down_bound_of_fov_in_map  = x + self.fov[0] // 2
        left_bound_of_fov_in_map  = y - self.fov[1] // 2
        right_bound_of_fov_in_map = y + self.fov[1] // 2

        up_bound_of_map           = 0
        down_bound_of_map         = self.map_size[0]
        left_bound_of_map         = 0
        right_bound_of_map        = self.map_size[1]

        up_bound_of_fov_in_fov    = 0
        down_bound_of_fov_in_fov  = self.fov[0]
        left_bound_of_fov_in_fov  = 0
        right_bound_of_fov_in_fov = self.fov[1]

        fov_grids_location = []
        
        # fov_x and fov_y should not be negative if the car is close to the edge of the map
        if up_bound_of_fov_in_map >= up_bound_of_map:
            fov_x_min_in_map = up_bound_of_fov_in_map
            fov_x_min_in_fov = up_bound_of_fov_in_fov
        else:
            fov_x_min_in_map = up_bound_of_map
            fov_x_min_in_fov = -up_bound_of_fov_in_map

        if down_bound_of_fov_in_map < down_bound_of_map:
            fov_x_max_in_map = down_bound_of_fov_in_map
            fov_x_max_in_fov = down_bound_of_fov_in_fov
        else:
            fov_x_max_in_map = down_bound_of_map
            fov_x_max_in_fov = down_bound_of_fov_in_fov - (down_bound_of_fov_in_map - down_bound_of_map)

        if left_bound_of_fov_in_map >= left_bound_of_map:
            fov_y_min_in_map = left_bound_of_fov_in_map
            fov_y_min_in_fov = left_bound_of_fov_in_fov
        else:
            fov_y_min_in_map = left_bound_of_map
            fov_y_min_in_fov = -left_bound_of_fov_in_map

        if right_bound_of_fov_in_map < right_bound_of_map:
            fov_y_max_in_map = right_bound_of_fov_in_map
            fov_y_max_in_fov = right_bound_of_fov_in_fov
        else:
            fov_y_max_in_map = right_bound_of_map
            fov_y_max_in_fov = right_bound_of_fov_in_fov - (right_bound_of_fov_in_map - right_bound_of_map)

        for i in range(fov_x_min_in_map, fov_x_max_in_map): # range(1, 6)= 1, 2, 3, 4, 5
            for j in range(fov_y_min_in_map, fov_y_max_in_map):
                fov_grids_location.append((i, j)) # Append the grid location to fov_grids_location
        return fov_x_min_in_map, fov_x_max_in_map, fov_y_min_in_map, fov_y_max_in_map, fov_x_min_in_fov, fov_x_max_in_fov, fov_y_min_in_fov, fov_y_max_in_fov, fov_grids_location

    def _observe(self):
        fov_x_min_in_map, fov_x_max_in_map, fov_y_min_in_map, fov_y_max_in_map, fov_x_min_in_fov, fov_x_max_in_fov, fov_y_min_in_fov, fov_y_max_in_fov, _ = self._get_fov_grids_location()
        fov_map = np.full((self.fov[0], self.fov[1]), OBSTACLE)  # Initialize FOV map with obstacles

        # Update FOV map with the actual map
        fov_map[fov_x_min_in_fov:fov_x_max_in_fov, fov_y_min_in_fov:fov_y_max_in_fov] = self.map[fov_x_min_in_map:fov_x_max_in_map, fov_y_min_in_map:fov_y_max_in_map]
        fov_map[self.fov[0]//2, self.fov[1]//2] = CAR  # Update car position in FOV map
        self.fov_map = fov_map
        return self.visit_count, self.fov_map, self.car_pos

    # Step function for interacting with the environment
    def step(self, action):
        old_pos = self.car_pos
        self.COLLISION_FLAG = False
        # self.SLIPPERY_FLAG = False

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
        
        self.car_pos = new_pos

        # Abnormal flags
        if self.map[self.car_pos] == OBSTACLE:  # Obstacle collision
            self.COLLISION_FLAG = True
            self.car_pos = old_pos
            print("Collision! Car stays in the same position: ", new_pos)
        else:
            if self.is_slippery:
                if np.random.rand() < 0.2: # Slip
                    self.SLIPPERY_FLAG = True
                    self.car_pos = old_pos
                    print("Slippery! Car stays in the same position: ", new_pos)
                else:
                    self.SLIPPERY_FLAG = False
                    print(f"Car moves {self.step_size} from {old_pos} to {new_pos}")
            else:
                self.SLIPPERY_FLAG = False
                print(f"Car moves {self.step_size} from {old_pos} to {new_pos}")

        reward, done = self.calculate_reward_and_done()

        # Update state
        self.state = self._observe()
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
                reward += REVISIT_PENALTY * self.visit_count[grid_location]  # Penalty for revisiting
            self.visit_count[grid_location] += 1

        # Check if map is fully explored
        if np.all(self.visit_count != 0):
            reward += FINISH_REWARD  # Big reward for completing exploration
            done = True
        return reward,done

    # Reset environment
    def reset(self):
        self.car_pos = self.init_pos
        self.fov_map = np.zeros(self.fov)
        self.visit_count = np.zeros(self.map_size)
        self.state = self._observe()
        return self.state

    # Render the environment
    def render(self, map_type):
        if map_type == 'visit_count':
            display_map = np.copy(self.visit_count)
            display_map[self.car_pos[0], self.car_pos[1]] = CAR
        elif map_type == 'fov_map':
            display_map = np.copy(self.fov_map)
        elif map_type == 'Map':
            display_map = np.copy(self.map)
            display_map[self.car_pos[0], self.car_pos[1]] = CAR
        plt.imshow(display_map, cmap='gray', interpolation='none')
        plt.title(map_type)
        plt.colorbar()
        plt.show()

# Example usage:
env = DummyGym(map_size=(40,21), car_size=(7,7), step_size=5, num_of_obstacles=10, fov=(10,10)) 
env.render('Map')
env.render('fov_map')
env.render('visit_count')
print(env.action_space.n)

# Perform a step
env.step(0)

# Render the maps after the step
env.render('Map')
env.render('fov_map')
env.render('visit_count')

# Perform another step
env.step(1)

# Render the maps after the step
env.render('Map')
env.render('fov_map')
env.render('visit_count')
