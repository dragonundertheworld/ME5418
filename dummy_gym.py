import numpy as np
import MapBuilder

class DummyGym:
    def __init__(self, init_pos=(0, 0), map_size=(10, 10), num_of_obstacles=5, FOV=(5, 5), see_through=False, map_file_path=None, is_slippery=False):
        self.map_size = map_size
        self.num_of_obstacles = num_of_obstacles
        self.init_pos = init_pos if init_pos else (np.random.randint(0, map_size[0]), np.random.randint(0, map_size[1]))
        self.FOV = FOV
        self.see_through = see_through
        self.is_slippery = is_slippery
        self.car_pos = self.init_pos
        self.explored_map = np.zeros(self.map_size)
        self.map = self._create_map(map_file_path)
        self.explored_map[self.car_pos] = 1  # Mark the starting position as explored

    def _create_map(self, map_file_path):
        if map_file_path:
            # Load from file
            return np.loadtxt(map_file_path)
        else:
            # Generate random map
            return MapBuilder.MapBuilder(self.map_size[0], self.map_size[1])

    def action_space(self):
        return ['Up', 'Down', 'Left', 'Right']

    def observation_space(self):
        # Return the car's FOV and position in FOV
        fov_map = self.map[max(0, self.car_pos[0]-self.FOV[0]//2):self.car_pos[0]+self.FOV[0]//2+1,
                           max(0, self.car_pos[1]-self.FOV[1]//2):self.car_pos[1]+self.FOV[1]//2+1]
        car_in_fov = np.zeros_like(fov_map)
        car_in_fov[self.FOV[0]//2, self.FOV[1]//2] = 1
        return fov_map, car_in_fov

    def step(self, action):
        # Handle action execution and update car position
        if action == 'Up':
            self.car_pos = (max(self.car_pos[0]-1, 0), self.car_pos[1])
        elif action == 'Down':
            self.car_pos = (min(self.car_pos[0]+1, self.map_size[0]-1), self.car_pos[1])
        elif action == 'Left':
            self.car_pos = (self.car_pos[0], max(self.car_pos[1]-1, 0))
        elif action == 'Right':
            self.car_pos = (self.car_pos[0], min(self.car_pos[1]+1, self.map_size[1]-1))
        self.explored_map[self.car_pos] = 1  # Update explored map

    def render(self):
        display_map = np.copy(self.map)
        display_map[self.car_pos] = 2  # Mark the car's position
        print(display_map)

# Example Usage
env = DummyGym(init_pos=(2,3), map_size=(30,30), num_of_obstacles=140, FOV=(5,5), see_through=False)
print("Action space: ", env.action_space())
print("Observation space: ", env.observation_space())
env.render()
