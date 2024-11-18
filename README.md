# Dummy DQN Project

This project implements a Deep Q-Network (DQN) for robotic exploration in a custom environment built using OpenAI's Gym interface. The custom environment simulates a 2D map where a robot (represented as a "car") explores an area, guided by a reinforcement learning-based algorithm.

## Table of Contents
- [Dummy DQN Project](#dummy-dqn-project)
  - [Table of Contents](#table-of-contents)
  - [Project Overview](#project-overview)
  - [Environment Details](#environment-details)
  - [Download and set up environment](#download-and-set-up-environment)
  - [Usage](#usage)
  - [Customization](#customization)
  - [Map Configuration](#map-configuration)
  - [Results](#results)
  - [Contributors](#contributors)

## Project Overview
The goal of this project is to use a reinforcement learning-based DQN algorithm to guide a robot in exploring an unknown environment. The robot must learn to maximize the exploration of new areas while avoiding collisions with obstacles.

The core components of this project include:
- **`dummy_DQN.ipynb`**: A Jupyter Notebook implementing the DQN algorithm.
- **`dummy_gym.py`**: A custom Gym environment script defining the observation space, action space, and reward function for the exploration task.
- **`smaller_map.txt`**: A text file defining the layout of the map with grid-based information.
- **`environment.yml`**: A YAML file for creating a Conda environment with all necessary dependencies.

## Environment Details
The environment is designed using the Gym interface and includes the following features:
- **Observation Space**: The robot's field of view (FOV), position in the map, and a count of visits for each grid cell.
- **Action Space**: The robot can move in four directions—up, down, left, and right.
- **Reward System**:
  - **Penalty for colliding with obstacles**: -10
  - **Reward for exploring a new area**: 0.1 times the visit count
  - **Penalty for revisiting explored areas**: -0.01 times the visit count
  - **Movement Penalty**: -0.01 for each move
  - **Big reward for completing exploration**: +10

## Download and set up environment
To set up this project, follow these steps:

1. **Unzip the project files and change directory**:
   ```bash
   cd group10
   ```

2. **Create a Conda environment** using the `environment.yml` file:
   ```bash
   conda env create -f environment.yml
   conda activate me5418-group10
   ```

## Usage
1. **Set up the custom environment**:
   Ensure that `dummy_gym.py` and `dummy_DQN.ipynb` are in the same directory. The custom environment should automatically recognize the map layout defined in `smaller_map.txt` which is under `test_map` folder.

2. **Run the Jupyter Notebook**:
   Open `dummy_DQN.ipynb` using Jupyter Notebook:
   ```bash
   jupyter notebook dummy_DQN.ipynb
   ```
   There should be already results of this note book. If there's nothing or you want to train the model yourself, just follow the instructions in the notebook to train the DQN agent on the custom environment, trained model will be stored under folder `train_brakpoint`. The notebook includes code for:
   - Defining the DQN model architecture
   - Showing visit_count map while training the DQN agent
   - Plotting results and performance metrics

3. **Evaluate the performance**:
   The notebook contains cells to visualize the agent's exploration path and to analyze training results.

## Customization
You can customize various aspects of the environment, such as:
- **Map Size and Shape**: Modify the map dimensions and obstacle locations in `smaller_map.txt`.
- **Reward Structure**: Change the reward or penalty values to encourage different behaviors.
- **Network Architecture**: Experiment with different DQN architectures or hyperparameters in `dummy_DQN.ipynb`.

## Map Configuration
The map layout is defined in the `smaller_map.txt` file, where different values represent different elements of the grid. For example:
- **0**: Represents an obstacle.
- **1**: Represents unexplored areas.
- **2**: Represents the car's starting position.

You can modify this file to create different scenarios for testing the DQN agent's performance.

## Results
- Car can usually finish 1 epsiode after training around 300 time steps
- With training more and more time steps, Q-values(here we have 4 actions, hence 4 Q-values in each state) in one state are getting closer and closer to each other, which shows our policy is getting more reasonable for our exploration.

## Contributors
- **Cao Zhihan** - In charge of `dummy_gym.py` `dummy_DQN.ipynb` `dummy_gym_unnitest.py` and `README.md`
- **Cheng Yuchao** - In charge of `Frontier_Exploration.py` and report
- **Teng Yu** - In charge of `MapBuilder.py` and part of `dummy_gym_unnitest.py`