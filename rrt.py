from dummy_gym import *
import matplotlib.pyplot as plt
from MapBuilder import save_and_show_map
from a3c_hypar import *
from a3c_model import *


# processed_states
map = env.map


class RRTExploration:
    def __init__(self, start, map, map_size, step_size, max_samples, processed_states):
        self.start = start
        self.map_size = map_size
        self.step_size = step_size
        self.max_samples = max_samples
        self.tree = [start]  # 初始化树
        self.map = map  # 0: 障碍物, 1: 未知, 2: 已探索区域
        self.car_fov = (3, 3)
        visit_count, fov_map, car_pos = processed_states
    
    def random_sample(self):
        x = np.random.uniform(0, self.map_size[0])
        y = np.random.uniform(0, self.map_size[1])
        return np.array([x, y])
    
    def nearest_node(self, random_point):
        distances = [np.linalg.norm(node - random_point) for node in self.tree]
        nearest_index = np.argmin(distances)
        return self.tree[nearest_index]
    
    def steer(self, from_node, to_node):
        direction = to_node - from_node
        distance = np.linalg.norm(direction)
        if distance > self.step_size:
            direction = direction / distance * self.step_size
        return from_node + direction
    
    def is_valid(self, new_node):
        # 检查是否在地图范围内且无碰撞
        x, y = int(new_node[0]), int(new_node[1])
        if x < 0 or y < 0 or x >= self.map_size[0] or y >= self.map_size[1]:
            return False
        if self.map[x, y] == 0:  
            return False
        return True
    
    def explore(self):
        time = 0
        for _ in range(self.max_samples):
            random_point = self.random_sample()
            nearest = self.nearest_node(random_point)
            new_node = self.steer(nearest, random_point)


            if self.is_valid(new_node):
                time += 1
                self.tree.append(new_node)
                self.update_map(new_node)
                self.show_map(new_node)
                save_and_show_png(map, './rrt_result', f'rrt after {time} steps') if time % 500 == 0 else None
                if self.is_fully_explored():
                    save_and_show_png(map, './rrt_result', f'rrt fully explored after {time} steps')
                    break
        print('time is :', time)
    
    def update_map(self, node):
        x, y = int(node[0]), int(node[1])
        self.map[x, y] = 2  # 标记为已探索区域
        # 更新小车的视角范围
        fov_width, fov_height = self.car_fov
        for i in range(-fov_width // 2, fov_width // 2 + 1):
            for j in range(-fov_height // 2, fov_height // 2 + 1):
                fov_x, fov_y = x + i, y + j
                if 0 <= fov_x < self.map_size[0] and 0 <= fov_y < self.map_size[1]:
                    if self.map[fov_x, fov_y] == 0:
                        pass
                    elif self.map[fov_x, fov_y] == 1:    
                        self.map[fov_x, fov_y] = 2  # 标记视角范围为已探索区域
    
    def is_fully_explored(self):
        return np.all(self.map != 1) 
    
    def show_map(self, node):
        plt.imshow(self.map, cmap='gray', origin='lower')
        plt.colorbar(label='Map Values')
        
        # 绘制小车的视角范围
        x, y = int(node[0]), int(node[1])
        fov_width, fov_height = self.car_fov
        for i in range(-fov_width // 2, fov_width // 2 + 1):
            for j in range(-fov_height // 2, fov_height // 2 + 1):
                fov_x, fov_y = x + i, y + j
                if 0 <= fov_x < self.map_size[0] and 0 <= fov_y < self.map_size[1]:
                    plt.scatter(fov_x, fov_y, color='red', s=10)  # 用红色点表示视角范围

        plt.scatter(x, y, color='blue', s=50)  # 用蓝色点表示小车位置
        plt.title('Map with Car FOV')
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.pause(0.1)  # 暂停以便显示更新
        plt.clf()  # 清除当前图形以便下次绘制

# 初始化参数
start = np.array([0, 0])
map_size = (30, 30)
max_samples = 10000

# 执行探索
rrt = RRTExploration(start, map, map_size, 5, max_samples, processed_states)
rrt.explore()
