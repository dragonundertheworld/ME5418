from a3c_model import ActorCriticNet, train 
import gym
import torch
import torch.optim as optim
import numpy as np

from dummy_gym import DummyGym


# 环境变量导入
env = DummyGym()
env_name = "DummyGym-v0"
original_state = env.observe()  # 状态空间
action_size = env.action_space.n    # 动作空间  
#print(f'action_size is {action_size}') # 4


# 处理原始state
def prepare_state(original_state):
    visit_count, fov_map, car_pos = original_state
    visit_count, fov_map, car_pos = np.array(visit_count), np.array(fov_map), np.array(car_pos)

    # add channel dimension
    visit_count_state = visit_count.reshape((*visit_count.shape, 1))
    fov_map_state = fov_map.reshape((*fov_map.shape, 1))
    car_pos_state = car_pos.reshape((*car_pos.shape, 1))
    
    return [visit_count_state, fov_map_state, car_pos_state], [visit_count_state.shape, fov_map_state.shape, car_pos_state.shape]


processed_states, state_shape = prepare_state(original_state)
# print(state_shape[0], state_shape[1], state_shape[2]) # (30, 30, 1) (3, 3, 1) (2, 1) 三个输入维度