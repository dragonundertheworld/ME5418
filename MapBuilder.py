import random
import numpy as np
import matplotlib.pyplot as plt
import time


def MapBuilder(row, col):
    map = np.ones((row, col), dtype=int)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    def dfs(x, y):
        map[x, y] = 0
        random.shuffle(directions)
        for dx, dy in directions:
            nx, ny = x + 2 * dx, y + 2 * dy
            if 1 <= nx < row - 1 and 1 <= ny < col - 1 and map[nx, ny] == 1:
                map[x + dx, y + dy] = 0  
                dfs(nx, ny)
    
    dfs(1, 1)

    # save map files
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    file_name = f"map_{timestamp}.{'txt'}"  
    file_path = f'ME5418/test_map/{file_name}'
    with open(file_path, 'w') as file:
        file.write(np.array2string(map))

    # show map
    plt.imshow(map, cmap='gray', interpolation='none')
    plt.colorbar()  
    plt.title(file_name)  
    plt.show()


matrix = MapBuilder(30, 30)