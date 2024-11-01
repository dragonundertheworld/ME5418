import torch.multiprocessing as mp

from torch import optim
from a3c_hypar import *
from a3c_worker import Worker

def train():
    # 创建全局网络
    global_network = ActorCriticNet(
        state_shape=state_shape,
        action_size=action_size
    )
    global_network.share_memory()  # 使参数在进程间共享
    
    # 创建优化器
    optimizer = optim.Adam(global_network.parameters(), lr=0.001)
    
    # 创建全局episode计数器
    global_episode = mp.Value('i', 0)
    # print(f"global_episode is {global_episode}") 
    
    # 创建workers
    workers = []
    num_workers = 1
    for i in range(num_workers):
        worker = Worker(
            env=env,
            name=f"w{i}",
            global_network=global_network,
            optimizer=optimizer,
            global_episode=global_episode
        )
        workers.append(worker)
    
    # 启动所有worker
    for worker in workers:
        worker.start()
    
    # 等待所有worker完成
    for worker in workers:
        worker.join()


if __name__ == "__main__":
    mp.set_start_method('spawn')  # 设置多进程启动方法
    train() 