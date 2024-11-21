import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import numpy as np
import gym
from dummy_gym import DummyGym


# self.visit_count, self.fov_map, self.car.pos

# 定义 Actor-Critic 网络结构
class ActorCriticNet(nn.Module):
    def __init__(self, state_shape, action_size, hidden_dim=256):
        super(ActorCriticNet, self).__init__()
        
        # 从state_shape获取输入维度
        self.visit_count_shape = state_shape[0]  # (30, 30)
        self.fov_map_shape = state_shape[1]      # (3, 3)
        self.car_pos_shape = state_shape[2]      # (2, 1)
        # print(f"visit_count_shape: {self.visit_count_shape}, fov_map_shape: {self.fov_map_shape}, car_pos_shape: {self.car_pos_shape}")
        
        # 处理访问计数地图的卷积网络
        self.visit_count_net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),  # 输出: 16 x 15 x 15
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), # 输出: 32 x 8 x 8
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 128)  # 2048 -> 128
        )
        
        # 处理FOV地图的卷积网络
        self.fov_map_net = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=2, stride=1, padding=0),  # 输出: 8 x 2 x 2
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(8 * 2 * 2, 32)  # 32 -> 32
        )
        
        # 计算合并后的维度
        self.merged_dim = 128 + 32 + self.car_pos_shape[0]  # 128 + 32 + 2 = 162
        
        # Actor网络
        self.actor = nn.Sequential(
            nn.Linear(self.merged_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_size)
        )
        
        # Critic网络
        self.critic = nn.Sequential(
            nn.Linear(self.merged_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, processed_states):
        ''' 
        输入:
        processed_states: 包含三个组件的列表
          - visit_count: 访问计数地图 (30, 30, 1) 
          - fov_map: 视野地图 (3, 3, 1)
          - car_pos: 车辆位置 (2, 1)
        
        输出:
        actor_output: 动作概率分布 (1, action_size)
        critic_output: 状态价值评估 (1, 1)
        从state中获取各个组件 
        ''' 

        visit_count, fov_map, car_pos = processed_states
        
        # 将NumPy数组转换为PyTorch张量并调整维度
        visit_count = torch.from_numpy(visit_count).float()  # (30, 30, 1)
        fov_map = torch.from_numpy(fov_map).float()         # (3, 3, 1)
        car_pos = torch.from_numpy(car_pos).float()         # (2, 1)
        
        # 调整维度顺序为 (batch, channel, height, width)
        visit_count = visit_count.squeeze(-1).unsqueeze(0).unsqueeze(0)  # (30, 30, 1) -> (30, 30) -> (1, 1, 30, 30)
        fov_map = fov_map.squeeze(-1).unsqueeze(0).unsqueeze(0)         # (3, 3, 1) -> (3, 3) -> (1, 1, 3, 3)
        car_pos = car_pos.squeeze(-1).unsqueeze(0)                       # (2, 1) -> (2,) -> (1, 2)
        
        # 处理访问计数地图 (30x30)
        x1 = self.visit_count_net(visit_count)
        x1 = x1.view(x1.size(0), -1)  # 展平
        
        # 处理FOV地图 (3x3)
        x2 = self.fov_map_net(fov_map)
        x2 = x2.view(x2.size(0), -1)  # 展平
        
        # 合并所有特征
        merged = torch.cat([x1, x2, car_pos], dim=1) # 1,162
        # print(merged.shape) 
        
        # 获取actor和critic输出
        actor_output = self.actor(merged)
        critic_output = self.critic(merged)
        
        return actor_output, critic_output
    
    def select_action(self,probs, epsilon):
        if np.random.rand() <= epsilon:
            action = np.random.random_integers(0, 3)
        else:
            action = torch.multinomial(probs, 1).item()
        return action
    
    @property
    def act(self, processed_states):
        logits, _ = self.forward(processed_states) # actor_output
        probs = torch.softmax(logits, dim=-1) # 计算动作概率分布
        # action = torch.multinomial(probs, 1).item()
        action = self.select_action(probs, 0.8) # 选择动作, epsilon=0.8, 80%概率选择最优动作
        print(f"action is {action}")
        return action
    
    def evaluate(self, processed_states, action):
        logits, value = self.forward(processed_states)
        probs = torch.softmax(logits, dim=-1)
        log_prob = torch.log(probs.gather(1, action)) # 计算log_prob 对数概率
        entropy = -(probs * torch.log(probs)).sum(1) # 计算概率分布的熵   
        return log_prob, entropy, value

# 共享本地模型以及世界模型参数
# def ensure_shared_grads(model, shared_model):
#     for param, shared_param in zip(model.parameters(), shared_model.parameters()):
#         if shared_param.grad is None:
#             shared_param._grad = param.grad

# # 主训练函数
# def train(global_model, optimizer, env_name, gamma, max_episodes, num_workers):
#     workers = []
#     for _ in range(num_workers):
#         worker = Worker(global_model, optimizer, env_name, gamma, max_episodes)
#         worker.start()
#         workers.append(worker)
#     for worker in workers:
#         worker.join()

