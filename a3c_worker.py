import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Categorical
import torch.multiprocessing as mp
import matplotlib.pyplot as plt
from a3c_model import ActorCriticNet
from a3c_hypar import *

class Worker(mp.Process):
    def __init__(self, env, name, global_network, optimizer, global_episode,gamma,max_episodes):
        super(Worker, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        self.gamma = gamma  # 折扣因子
        self.max_episodes = max_episodes    
        
    def run(self):
        """Worker的主循环"""
        while self.global_episode.value < self.max_episodes:
            time = 0;
            time += 1
            if time % 100 == 0:
                print(f"Worker {self.name}, Episode: {self.global_episode.value}")
                # time.sleep(1)
            # 收集一个episode的数据
            trajectory = self.collect_trajectory()
            
            if trajectory:  # 确保轨迹不为空
                # 计算优势值和回报
                processed_states, actions, rewards = zip(*trajectory)
                processed_states = list(processed_states)
                advantages, returns = self.compute_advantages_and_returns(rewards)
                
                # 训练网络
                self.train(processed_states, actions, advantages, returns) 
                # states,actions为tuple, advantages,returns为tensor
                
                # 同步本地网络与全局网络
                self.sync_with_global()
                
                # 更新全局episode计数
                with self.global_episode.get_lock():
                    self.global_episode.value += 1
    
    def collect_trajectory(self):
        """收集一个episode的轨迹"""
        trajectory = []
        state = self.env.reset() # processed_state(3个参数)
        processed_states, _ = prepare_state(state) # 数据预处理
        print("Length of processed_states:", len(processed_states))
        print(f"type of processed_states: {type(processed_states)}")

        done = False
        total_reward = 0
        time_step = 0

        while not done:
            time_step += 1
            # 使用本地网络选择动作
            if time_step == 1:
                action = self.local_network.act(processed_states)
            else:
                action = self.local_network.act(state)

            # states = processed_states
            
            # 执行动作 next_state是observe()返回的 未进行纬度处理
            state, reward, done, _ = self.env.step(action)
            state, _ = prepare_state(state)

            if time_step % 7500 == 0:
                # self.env.render(map_type="visit_count")
                print(f"********************time_step: {time_step}********************")
                # print(f"self.env.visit_count: {self.env.visit_count}")
                # print(f"self.env.fov_map: {self.env.fov_map}")
                
                # plt.imshow(self.env.visit_count)
                # plt.colorbar()
                # plt.show()
            
            # 保存transition
            trajectory.append((state, action, reward))
            
            # state = next_state
            total_reward += reward
            
        print(f"Worker {self.name}, Episode Reward: {total_reward}")
        return trajectory
    
    def compute_advantages_and_returns(self, rewards):
        """计算优势值和折扣回报"""
        rewards = torch.tensor(rewards, dtype=torch.float32)
        returns = torch.zeros_like(rewards)
        advantages = torch.zeros_like(rewards)
        
        # 计算回报(从后向前计算)
        R = 0
        for t in reversed(range(len(rewards))):
            R = rewards[t] + self.gamma * R    # R = r_t + γ * R_{t+1}
            returns[t] = R
        
        # 计算优势值
        values = self.local_network.forward(processed_states)[1].detach()  # critic输出
        advantages = returns - values
        
        return advantages, returns # 返回优势值和实际回报
    
    def train(self, states, actions, advantages, returns):
        """更新网络参数"""
        # 1. 首先将列表转换为numpy数组
        # states = [np.array(s) if isinstance(s, list) else s for s in states]
        
        # # 2. 然后转换为PyTorch张量
        # states = torch.stack([torch.from_numpy(s) for s in states]).to(self.device)      
        actions = torch.tensor(list(actions), dtype=torch.long).to(self.device)
        advantages = torch.tensor(advantages, dtype=torch.float32).to(self.device)
        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)
        
        print("Length of states:", len(states))
        print(f"type of states: {type(states)}")
        # 2. 通过网络前向传播
        logits, values = self.local_network.forward(states)
        
        # 3. 计算Actor（策略）损失
        dist = Categorical(logits=logits)  # 创建动作概率分布
        log_probs = dist.log_prob(actions)  # 计算所选动作的对数概率
        policy_loss = -(log_probs * advantages).mean()  # 策略梯度损失
        
        # 4. 计算Critic（价值）损失
        value_loss = 0.5 * (returns - values.squeeze()).pow(2).mean()  # MSE损失
        
        # 5. 合并总损失
        total_loss = policy_loss + value_loss
        
        # 6. 反向传播和参数更新
        self.optimizer.zero_grad()  # 清除旧的梯度
        total_loss.backward()  # 计算梯度
        
        # 7. 将本地梯度同步到全局网络
        for local_param, global_param in zip(
            self.local_network.parameters(),
            self.global_network.parameters()
        ):
            if global_param.grad is not None:
                global_param.grad += local_param.grad
            else:
                global_param.grad = local_param.grad.clone()
        
        self.optimizer.step()  # 更新参数
        
        # 8. 清理GPU内存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()    
    
    def sync_with_global(self):
        """同步全局网络参数到本地网络"""
        self.local_network.load_state_dict(self.global_network.state_dict()) 
