import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Categorical
import torch.multiprocessing as mp
import matplotlib.pyplot as plt
from a3c_model import ActorCriticNet
from a3c_hypar import *

class Worker(mp.Process):
    def __init__(self, env, name, global_network, optimizer, global_episode):
        super(Worker, self).__init__()
        self.env = env
        self.name = name
        self.global_network = global_network
        self.optimizer = optimizer
        self.global_episode = global_episode
        
        # 创建本地网络
        self.local_network = ActorCriticNet(
            state_shape=state_shape,
            action_size=action_size
        )
        # 初始同步
        self.sync_with_global()
        
        # 超参数
        self.gamma = 0.99  # 折扣因子
        self.max_episodes = 10000
        
    def run(self):
        """Worker的主循环"""
        while self.global_episode.value < self.max_episodes:
            # 收集一个episode的数据
            trajectory = self.collect_trajectory()
            
            if trajectory:  # 确保轨迹不为空
                # 计算优势值和回报
                states, actions, rewards = zip(*trajectory)
                advantages, returns = self.compute_advantages_and_returns(rewards)
                
                # 训练网络
                self.train(states, actions, advantages, returns)
                
                # 同步本地网络与全局网络
                self.sync_with_global()
                
                # 更新全局episode计数
                with self.global_episode.get_lock():
                    self.global_episode.value += 1
    
    def collect_trajectory(self):
        """收集一个episode的轨迹"""
        trajectory = []
        state = self.env.reset()
        processed_states, _ = prepare_state(state)

        done = False
        total_reward = 0
        time_step = 0

        while not done:
            time_step += 1
            # 使用本地网络选择动作
            action = self.local_network.act(processed_states)
            
            # 执行动作
            next_state, reward, done, _ = self.env.step(action)
            if time_step % 100 == 0:
                self.env.render(map_type="visit_count")
                print(f"********************time_step: {time_step}********************")
                print(f"self.env.visit_count: {self.env.visit_count}")
                print(f"self.env.fov_map: {self.env.fov_map}")
                plt.imshow(self.env.fov_map)
                plt.colorbar()
                plt.show()
            
            # 保存transition
            trajectory.append((state, action, reward))
            
            state = next_state
            total_reward += reward
            
        print(f"Worker {self.name}, Episode Reward: {total_reward}")
        return trajectory
    
    def compute_advantages_and_returns(self, rewards):
        """计算优势值和折扣回报"""
        rewards = torch.tensor(rewards, dtype=torch.float32)
        returns = torch.zeros_like(rewards)
        advantages = torch.zeros_like(rewards)
        
        # 计算回报
        R = 0
        for t in reversed(range(len(rewards))):
            R = rewards[t] + self.gamma * R
            returns[t] = R
        
        # 计算优势值
        values = self.local_network.forward(states)[1].detach()  # critic输出
        advantages = returns - values
        
        return advantages, returns
    
    def train(self, states, actions, advantages, returns):
        """更新网络参数"""
        # 转换数据格式
        states = torch.stack([torch.from_numpy(s) for s in states])
        actions = torch.tensor(actions, dtype=torch.long)
        advantages = torch.tensor(advantages, dtype=torch.float32)
        returns = torch.tensor(returns, dtype=torch.float32)
        
        # 前向传播
        logits, values = self.local_network.forward(states)
        
        # 计算策略损失
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        policy_loss = -(log_probs * advantages).mean()
        
        # 计算价值损失
        value_loss = 0.5 * (returns - values.squeeze()).pow(2).mean()
        
        # 总损失
        total_loss = policy_loss + value_loss
        
        # 更新全局网络
        self.optimizer.zero_grad()
        total_loss.backward()
        
        # 将本地梯度应用到全局网络
        for local_param, global_param in zip(
            self.local_network.parameters(),
            self.global_network.parameters()
        ):
            if global_param.grad is not None:
                global_param.grad += local_param.grad
            else:
                global_param.grad = local_param.grad.clone()
        
        self.optimizer.step()
    
    def sync_with_global(self):
        """同步本地网络参数到全局网络"""
        self.local_network.load_state_dict(self.global_network.state_dict()) 