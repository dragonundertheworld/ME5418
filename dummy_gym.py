import numpy as np
import matplotlib.pyplot as plt
import os
# from map_generate import create_random_shape_map
import MapBuilder

obstacle = 1
free_space = 0
car = 2
explored = -1

class DummyGym:
    def __init__(self, init_pos=(0, 0), car_size=(1,1), step_size=1, map_size=(10, 10), num_of_obstacles=5, FOV=(5, 5), see_through=False, map_file_path=None, is_slippery=False):
        self.car_size = car_size
        self.step_size = step_size
        self.map_size = map_size
        self.num_of_obstacles = num_of_obstacles
        self.init_pos = init_pos if init_pos else (np.random.randint(0, map_size[0]), np.random.randint(0, map_size[1]))
        self.FOV = FOV
        self.see_through = see_through
        self.is_slippery = is_slippery
        self.car_pos = self.init_pos
        self.explored_map = np.zeros(self.map_size)
        self.map = self._create_map(map_file_path)
        self.explored_map[self.car_pos] = car  # Mark the starting position as explored

    def _create_map(self, map_file_path):
        if map_file_path:
            if not os.path.exists(map_file_path):
                raise ValueError("Invalid map file path!")
            # Load from file
            map = np.loadtxt(map_file_path)
            return map
        else:
            return MapBuilder.MapBuilder(self.map_size[0], self.map_size[1])
            # Generate random-shape map
            # 其实和矩形地图差不多
            # map = create_random_shape_map(self.map_size, self.map_size[0]*self.map_size[1]//2, self.num_of_obstacles)
            # return map
    
    def _place_car(self):
        # Randomly place the car on an open space
        while True:
            self.car_pos = (np.random.randint(0, self.map_size[0] - self.car_size[0] + 1), 
                            np.random.randint(0, self.map_size[1] - self.car_size[1] + 1))
            
            # Check if the car can fit in the space
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
        self.explored_map[self.car_pos] = car

    def action_space(self):
        return ['Up', 'Down', 'Left', 'Right']

    def observation_space(self):
        # Return the car's FOV and position in FOV
        fov_map = self.map[max(0, self.car_pos[0]-self.FOV[0]//2):self.car_pos[0]+self.FOV[0]//2+1,
                           max(0, self.car_pos[1]-self.FOV[1]//2):self.car_pos[1]+self.FOV[1]//2+1]
        car_in_fov = np.zeros_like(fov_map)
        car_in_fov[self.FOV[0]//2, self.FOV[1]//2] = car # Mark the car's position in FOV
        return fov_map, car_in_fov

    def step(self, action):
        # Handle action execution and update car position
        if action == 'Up':
            self.car_pos = (max(self.car_pos[0]-self.step_size, 0), self.car_pos[1])
        elif action == 'Down':
            self.car_pos = (min(self.car_pos[0]+self.step_size, self.map_size[0]-1), self.car_pos[1])
        elif action == 'Left':
            self.car_pos = (self.car_pos[0], max(self.car_pos[1]-self.step_size, 0))
        elif action == 'Right':
            self.car_pos = (self.car_pos[0], min(self.car_pos[1]+self.step_size, self.map_size[1]-1))
        self.explored_map[self.car_pos] = 1  # Update explored map

    def render(self):
        display_map = np.copy(self.map)
        display_map[self.car_pos] = car  # Mark the car's position
        print(display_map)
        plt.imshow(display_map, cmap='gray', interpolation='none')
        plt.colorbar()
        plt.title("Map")
        plt.show()

# Example Usage
env = DummyGym(init_pos=(2,3), map_size=(30,30), num_of_obstacles=140, FOV=(5,5), see_through=False) # num_of_obstacles没有用到，不知道以后要不要用地图大小和障碍物数量创建地图
print("Action space: ", env.action_space())
print("Observation space: ", env.observation_space())
env.render()
