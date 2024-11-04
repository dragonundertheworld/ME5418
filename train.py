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
    num_workers = max(1, mp.cpu_count() // 2)  # 使用CPU核心数的一半作为worker数
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
    
    print("All workers completed.")


if __name__ == "__main__":
    torch.cuda.empty_cache()
    mp.set_start_method('spawn')  # 设置多进程启动方法
    train() 