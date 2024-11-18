from dummy_gym import *
import matplotlib.pyplot as plt

env = DummyGym()
map = env.map


class RRTExploration:
    def __init__(self, start, map, map_size, step_size, max_samples):
        self.start = start
        self.map_size = map_size
        self.step_size = step_size
        self.max_samples = max_samples
        self.tree = [start]  # 初始化树
        self.map = map  # 0: 障碍物, 1: 未知, 2: 已探索区域
    
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
        for _ in range(self.max_samples):
            random_point = self.random_sample()
            nearest = self.nearest_node(random_point)
            new_node = self.steer(nearest, random_point)
            if self.is_valid(new_node):
                self.tree.append(new_node)
                self.update_map(new_node)
                if self.is_fully_explored():
                    # print(map)
                    break
        # 出图
        print(map)
    
    def update_map(self, node):
        x, y = int(node[0]), int(node[1])
        self.map[x, y] = 2  # 标记为已探索区域
    
    def is_fully_explored(self):
        return np.all(self.map != 1) 

# 初始化参数
start = np.array([0, 0])
map_size = (30, 30)
max_samples = 10000

# 执行探索
rrt = RRTExploration(start, map, map_size, STEP_SIZE, max_samples)
rrt.explore()
