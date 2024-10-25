import numpy as np
import math
import matplotlib.pyplot as plt
from dummy_gym import DummyGym  # 从dummy_gym.py加载环境

class FrontierExplorationAgent:
    def __init__(self, env):
        self.env = env
        self.state = self.env.reset()

    def detect_frontier_points(self, map):
        """检测地图中的前沿点。"""
        frontier_points = []
        for x in range(1, map.shape[0] - 1):
            for y in range(1, map.shape[1] - 1):
                if map[x, y] == 1:  # 未探索的网格
                    neighbors = [(x-1, y), (x+1, y), (x, y-1), (x, y+1)]
                    if any(map[nx, ny] != 1 for nx, ny in neighbors):
                        frontier_points.append((x, y))
        return frontier_points

    def get_action_toward_target(self, target, position):
        """返回前往目标的动作（0: 上, 1: 下, 2: 左, 3: 右）。"""
        dx, dy = target[0] - position[0], target[1] - position[1]
        if abs(dx) > abs(dy):  # 优先水平方向
            return 1 if dx > 0 else 0  # 向下如果 dx > 0，向上如果 dx < 0
        else:
            return 3 if dy > 0 else 2  # 向右如果 dy > 0，向左如果 dy < 0

    def visualize(self):
        """初始化可视化图形。"""
        self.fig, self.ax = plt.subplots()
        self.img = self.ax.imshow(self.env.map, cmap="Blues", vmin=0, vmax=2)
        self.car_marker, = self.ax.plot([], [], "ro")  # 小车标记为红色圆点

    def update_visualization(self):
        """更新可视化图形。"""
        self.img.set_data(self.env.map)
        self.car_marker.set_data(self.env.car.pos[1], self.env.car.pos[0])  # 更新小车位置
        plt.draw()
        plt.pause(0.1)  # 让绘图刷新

    def explore_until_complete(self):
        """探索直到地图完全探索完成。"""
        self.visualize()
        total_reward = 0
        done = False
        self.state = self.env.reset()

        while not done:
            # 检查前沿点并决定下一步动作
            frontier_points = self.detect_frontier_points(self.env.map)
            if frontier_points:
                distances = [math.dist(self.env.car.pos, fp) for fp in frontier_points]
                nearest_fp = frontier_points[np.argmin(distances)]
                action = self.get_action_toward_target(nearest_fp, self.env.car.pos)
            else:
                done = True  # 没有前沿点，地图完全探索
                break

            # 执行动作并更新奖励
            self.state, reward, done, _ = self.env.step(action)
            total_reward += reward

            # 更新可视化
            self.update_visualization()

        print(f"Exploration finished with total reward: {total_reward}")
        plt.show()  # 保持图像

if __name__ == "__main__":
    env = DummyGym()
    agent = FrontierExplorationAgent(env)
    agent.explore_until_complete()
