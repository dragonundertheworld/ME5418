import numpy as np
from dummy_gym import DummyGym
from dummy_gym import EXPLORED, UNEXPLORED, STEP_SIZE, OBSTACLE
from dummy_gym import save_to_gif
import random

class FrontierExplorer:
    """
    A class for performing frontier exploration in a custom gym environment.
    """

    def __init__(self, env):
        """
        Initialize the FrontierExplorer with a given environment.
        
        Args:
            env: An instance of the DummyGym environment.
        """
        self.env = env

    def identify_frontiers(self):
        """
        Identify frontiers in the current visit count map.
        
        Args:
            visit_count: The visit count of the environment grid.
            
        Returns:
            List of frontier coordinates.
        """
        frontiers = []
        rows, cols = self.env.visit_count.shape
        for row in range(rows):
            for col in range(cols):
                if self.env.visit_count[row, col] == UNEXPLORED:
                    for neighbor in self.get_neighbors(row, col, rows, cols):
                        if self.env.visit_count[neighbor] == EXPLORED and self.env.map[neighbor] != OBSTACLE:
                            # Neighbor is explored
                            frontiers.append((row, col))
                            break
        return frontiers

    @staticmethod
    def get_neighbors(row, col, rows, cols):
        """
        Get valid neighbors of a cell in the grid.
        
        Args:
            row: Row index of the cell.
            col: Column index of the cell.
            rows: Total number of rows in the grid.
            cols: Total number of columns in the grid.
        
        Returns:
            List of valid neighbor coordinates.
        """
        directions = [(-STEP_SIZE, 0), (STEP_SIZE, 0), (0, -STEP_SIZE), (0, STEP_SIZE)]
        neighbor_list = [
            (row + dr, col + dc) # Calculate the neighbor coordinates
            for dr, dc in directions
            if 0 <= row + dr < rows and 0 <= col + dc < cols # Check if the neighbor is within the grid
        ]
        # print(f'neighbor_list is {neighbor_list}')
        return neighbor_list

    @staticmethod
    def select_nearest_frontier(car_pos, frontiers):
        """
        Select the nearest frontier based on Manhattan distance.
        
        Args:
            car_pos: The current position of the car.
            frontiers: A list of frontier coordinates.
        
        Returns:
            The coordinate of the nearest frontier.
        """
        # print('selecting nearest frontier')
        nearest_frontier = min(frontiers, key=lambda x: abs(x[0] - car_pos[0]) + abs(x[1] - car_pos[1]))
        # print(f'going to nearest frontier {nearest_frontier}')
        return nearest_frontier

    def move_towards_target(self, car_pos, target):
        """
        Determine the action to move towards the target.
        
        Args:
            car_pos: The current position of the car.
            target: The target position.
        
        Returns:
            The action to move closer to the target (0: Up, 1: Down, 2: Left, 3: Right).
        """
        if car_pos[0] > target[0] and all(self.env.map[target[0]:car_pos[0], car_pos[1]] != OBSTACLE):
            return 0  # Up
        elif car_pos[0] < target[0] and all(self.env.map[car_pos[0]:target[0], car_pos[1]] != OBSTACLE):
            return 1  # Down
        elif car_pos[1] > target[1] and all(self.env.map[car_pos[0], target[1]:car_pos[1]] != OBSTACLE):
            return 2  # Left
        else:
            return 3

    def explore(self):
        """
        Perform the exploration process in the environment.
        """
        done = False
        time_step = 0
        while not done:
            # Get the current observation

            image_list.append(self.env.visit_count / np.max(self.env.visit_count) * 255)
            save_to_gif(image_list, 'frontier_gif', 'frontier_exploration.gif')
            visit_count, _, car_pos = self.env.observe()
            
            # Identify frontiers
            frontiers = self.identify_frontiers()
            
            if not frontiers:
                # No frontiers left, exploration is complete
                print("No frontiers left to explore.")
                break
            
            # Select the nearest frontier
            print('ready to select nearest frontier')
            target = self.select_nearest_frontier(car_pos, frontiers)
            
            # Move towards the frontier
            print('moving towards frontier')
            action = self.move_towards_target(car_pos, target)
            self.env.render() if action == None else None
            print('taking action')
            state, reward, done, _ = self.env.step(action)
            
            # Render the environment
            self.env.render(map_type='visit_count') if time_step % 50 == 0 else None
            cells_visited = self.env.visit_count[self.env.visit_count == EXPLORED].shape[0] # 计算已经访问的cell数量
            cells = self.env.map_size[0]*self.env.map_size[1]
            print(f"Cells visited: {cells_visited}/{cells} at time step: {time_step}")
            print(f"Action: {action} at time step {time_step}")
            time_step += 1


# Example usage
if __name__ == "__main__":
    # Initialize the environment
    env = DummyGym()
    image_list = []
    image_list.append(env.visit_count)
    env.step(random.choice([0, 1, 2, 3]))

    # Create an instance of FrontierExplorer
    explorer = FrontierExplorer(env)

    # Perform the exploration
    explorer.explore()

