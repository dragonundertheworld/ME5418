import numpy as np
from dummy_gym import DummyGym
from dummy_gym import EXPLORED, UNEXPLORED, STEP_SIZE

env = DummyGym()

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

    def identify_frontiers(self, visit_count):
        """
        Identify frontiers in the current visit count map.
        
        Args:
            visit_count: The visit count of the environment grid.
            
        Returns:
            List of frontier coordinates.
        """
        frontiers = []
        rows, cols = visit_count.shape
        for row in range(rows):
            for col in range(cols):
                if visit_count[row, col] == EXPLORED:
                    for neighbor in self.get_neighbors(row, col, rows, cols):
# ****************************stuck here*****************************************
                        if visit_count[neighbor] == UNEXPLORED:
                            # Neighbor is explored
                            print('find frontier')
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
        print(f'neighbor_list is {neighbor_list}')
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
        print('selecting nearest frontier')
        nearest_frontier = min(frontiers, key=lambda x: abs(x[0] - car_pos[0]) + abs(x[1] - car_pos[1]))
        print(f'going to nearest frontier {nearest_frontier}')
        return nearest_frontier

    @staticmethod
    def move_towards_target(car_pos, target):
        """
        Determine the action to move towards the target.
        
        Args:
            car_pos: The current position of the car.
            target: The target position.
        
        Returns:
            The action to move closer to the target (0: Up, 1: Down, 2: Left, 3: Right).
        """
        if car_pos[0] > target[0]:
            return 0  # Up
        elif car_pos[0] < target[0]:
            return 1  # Down
        elif car_pos[1] > target[1]:
            return 2  # Left
        elif car_pos[1] < target[1]:
            return 3  # Right

    def explore(self):
        """
        Perform the exploration process in the environment.
        """
        done = False
        while not done:
            # Get the current observation
            visit_count, _, car_pos = self.env.observe()
            
            # Identify frontiers
            frontiers = self.identify_frontiers(visit_count)
            
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
            print('taking action')
            state, reward, done, _ = self.env.step(action)
            
            # Render the environment
            self.env.render()
            print(f"Action: {action}, Reward: {reward}")


# Example usage
if __name__ == "__main__":
    # Initialize the environment
    env = DummyGym()

    # Create an instance of FrontierExplorer
    explorer = FrontierExplorer(env)

    # Perform the exploration
    explorer.explore()

