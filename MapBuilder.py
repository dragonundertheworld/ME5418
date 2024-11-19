import random
import numpy as np
import matplotlib.pyplot as plt
import time

size_expansion = 0

def MapBuilder(row, col, ocu, size):
    '''
        This func implement the function of building a map and save it as a txt files
        paras: row: the row of the map
        col: the coloume of the map
        ocu: the occupy of the obstcal
        size: the size of the robot 
    '''
    map = np.ones((row, col), dtype=int)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    # 深度优先搜索 (DFS) 生成迷宫
    def dfs(x, y):
        map[x, y] = 0
        random.shuffle(directions)
        for dx, dy in directions:
            nx, ny = x + 2 * dx, y + 2 * dy
            if 1 <= nx < row - 1 and 1 <= ny < col - 1 and map[nx, ny] == 1:
                map[x + dx, y + dy] = 0  
                dfs(nx, ny)
    
    dfs(1, 1)

    total_cells = (row - 2) * (col - 2) 
    obstacle_count = int(ocu * total_cells)  

    empty_cells = [(i, j) for i in range(1, row - 1) for j in range(1, col - 1) if map[i, j] == 0]
    random.shuffle(empty_cells)

    for i in range(obstacle_count):
        if empty_cells:
            x, y = empty_cells.pop()
            map[x, y] = 1

    # expend the map depending on the robot size 
    expanded_map = np.kron(map, np.ones((size+size_expansion, size+size_expansion), dtype=int))

    return expanded_map

def save_and_show_map(expanded_map, map_type):
    # save map file
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    file_name = f"{map_type}.txt"  
    file_path = f'./test_map/{file_name}'
    with open(file_path, 'w') as file:
        np.savetxt(file, expanded_map, fmt='%d')

    plt.imshow(expanded_map, cmap='gray', interpolation='none')
    plt.colorbar()  
    plt.title(file_name)  
    plt.show()
# 调用MapBuilder并生成扩展后的迷宫
# matrix = MapBuilder(30, 30, 0.3, 1)
# save_and_show_map(matrix, 'smaller_map')

# expanded_map = MapBuilder(30, 30, 0.3, 7)
# save_and_show_map(expanded_map,'test01')