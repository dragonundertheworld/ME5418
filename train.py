import torch.multiprocessing as mp

from torch import optim
from a3c_hypar import *
from a3c_worker import Worker

def train():
    # 设置GPU内存分配策略
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.2)  # 每个进程限制使用20%的GPU内存
        torch.cuda.empty_cache()  # 清空GPU缓存
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 创建全局网络
    global_network = ActorCriticNet(
        state_shape=state_shape,
        action_size=action_size
    ).to(device)
    global_network.share_memory()  # 使参数在进程间共享
    
    # 创建优化器
    optimizer = optim.Adam(global_network.parameters(), lr=0.001)
    
    # 创建全局episode计数器(所有进程都可以访问&&修改，用于追踪训练总轮数）
    global_episode = mp.Value('i', 0)
    # print(f"global_episode is {global_episode}") 
    
    # 创建workers
    workers = []
    #num_workers = max(1, mp.cpu_count() // 2)  # 使用CPU核心数的一半作为worker数
    num_workers = 1
    print(f"num_workers is {num_workers}")
    for i in range(num_workers):
        worker = Worker(
            env=env,
            name=f"w{i}",
            global_network=global_network,
            optimizer=optimizer,
            global_episode=global_episode,
            gamma=gamma,
            max_episodes=max_episodes   
        )
        workers.append(worker)
    
    # 启动所有worker
    for worker in workers:
        worker.start()
    
    # 等待所有worker完成
    for worker in workers:
        worker.join()
    
    # 保存模型参数
    torch.save(global_network.state_dict(), 'model_parameters.pth')  # 保存模型参数到文件
    print("Model parameters saved to model_parameters.pth")
    
    print("All workers completed.")

def test():
    # 加载模型参数
    global_network = ActorCriticNet(
        state_shape=state_shape,
        action_size=action_size
    )
    global_network.load_state_dict(torch.load('model_parameters.pth'))  # 从文件加载模型参数
    global_network.eval()  # 设置为评估模式

    # 测试环境
    env = ...  # 初始化测试环境
    state = env.reset()  # 重置环境以获取初始状态
    done = False

    step = 0

    while not done:
        with torch.no_grad():  # 不需要计算梯度
            action = global_network.select_action(state)  # 选择动作
        state, reward, done, _ = env.step(action)  # 执行动作并获取下一个状态和奖励
        step += 1
        if step % 100 == 0:
            print("The step is:", step)
    
    print("The step is:", step)
    print("Testing completed.")

if __name__ == "__main__":
    torch.cuda.empty_cache()
    mp.set_start_method('spawn')  # 设置多进程启动方法
    # train()
    test()  