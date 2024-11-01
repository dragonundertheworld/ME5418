from a3c_model import ActorCriticNet, train 
import gym
import torch
import torch.optim as optim
from dummy_gym import DummyGym
from a3c_hypar import *


if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    a3c = ActorCriticNet(state_shape, action_size)
    ctor_output, critic_output = a3c.forward(processed_states)
    # print(actor_output.shape, critic_output.shape)
    # torch.Size([1, 4]) torch.Size([1, 1])

    # env_name = "CartPole-v1"
    # env = gym.make(env_name)
    # input_dim = 3
    # action_space = env.action_space.n
    
    global_model = ActorCriticNet(state_shape, action_size).to(device)
    global_model.share_memory()
    optimizer = optim.Adam(global_model.parameters(), lr=1e-4)
    
    train(global_model, optimizer, env_name, gamma=0.99, max_episodes=1000, num_workers=4)
