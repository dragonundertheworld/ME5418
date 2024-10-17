import unittest
import numpy as np
from unittest.mock import patch, MagicMock
from ME5418.dummy_gym import DummyGym  # Ensure DummyGym is imported from the correct module
from ME5418.dummy_gym import FINISH_REWARD, COLLISION_PENALTY, EXPLORATION_REWARD, REVISIT_PENALTY, MOVEMENT_PENALTY, STATIONARY_PENALTY
from ME5418.dummy_gym import OBSTACLE

class TestDummyGym(unittest.TestCase):
    def setUp(self):
        # Common initialization for each test
        self.env = DummyGym(init_pos=(2, 3), car_size=(1, 1), 
                            step_size=1, map_size=(10, 10), num_of_obstacles=5, fov=(5, 5))
        self.env.map = np.zeros((10, 10))  # No obstacles for simplicity in most tests

    def test_create_map_with_invalid_path(self):
        with self.assertRaises(ValueError):
            self.env._create_map("invalid/path")

    @patch("DummyGym.MapBuilder.MapBuilder")
    def test_create_map_without_file_path(self, mock_map_builder):
        mock_map_builder.return_value = np.zeros((10, 10))
        map_created = self.env._create_map(None)
        mock_map_builder.assert_called_once()
        self.assertEqual(map_created.shape, self.env.map_size)

    def test_place_car_no_collision(self):
        # Assure that car is placed in a free space
        self.env.map[2:3, 3:4] = 0  # Clear area for car placement
        self.env._place_car()
        self.assertNotEqual(self.env.car_pos, (2, 3))  # Since car placement is random

    def test_get_fov_grids_location(self):
        self.env.car_pos = (2, 3)
        _, _, _, _, _, _, _, _, fov_grids_location = self.env._get_fov_grids_location()
        self.assertTrue(all(isinstance(pos, tuple) for pos in fov_grids_location))

    def test_observe_updates_correctly(self):
        visit_count, fov_map, car_pos = self.env._observe()
        self.assertEqual(visit_count.shape, self.env.map_size)
        self.assertEqual(fov_map.shape, self.env.fov)
        self.assertEqual(car_pos, self.env.car_pos)

    def test_step_within_bounds(self):
        # Move the car up within bounds
        self.env.car_pos = (5, 5)
        state, reward, done, _ = self.env.step(0)  # Action 0: Up
        self.assertEqual(self.env.car_pos, (4, 5))

    def test_step_outside_bounds(self):
        # Try to move car outside the upper boundary
        self.env.car_pos = (0, 5)
        state, reward, done, _ = self.env.step(0)  # Action 0: Up
        self.assertEqual(self.env.car_pos, (0, 5))  # Position should not change
        
    def test_step_up(self):
        self.env.car_pos = (5, 5)
        state, reward, done, _ = self.env.step(0)  # Action 0: Up
        self.assertEqual(self.env.car_pos, (4, 5))
        self.assertFalse(done)
        self.assertIsInstance(reward, (int, float))

    def test_step_down(self):
        self.env.car_pos = (5, 5)
        state, reward, done, _ = self.env.step(1)  # Action 1: Down
        self.assertEqual(self.env.car_pos, (6, 5))
        self.assertFalse(done)
        self.assertIsInstance(reward, (int, float))

    def test_step_left(self):
        self.env.car_pos = (5, 5)
        state, reward, done, _ = self.env.step(2)  # Action 2: Left
        self.assertEqual(self.env.car_pos, (5, 4))
        self.assertFalse(done)
        self.assertIsInstance(reward, (int, float))

    def test_step_right(self):
        self.env.car_pos = (5, 5)
        state, reward, done, _ = self.env.step(3)  # Action 3: Right
        self.assertEqual(self.env.car_pos, (5, 6))
        self.assertFalse(done)
        self.assertIsInstance(reward, (int, float))

    def test_step_collision(self):
        self.env.car_pos = (5, 5)
        self.env.map[4, 5] = OBSTACLE  # Place an obstacle above the car
        state, reward, done, _ = self.env.step(0)  # Action 0: Up
        self.assertEqual(self.env.car_pos, (5, 5))  # Position should not change
        self.assertTrue(self.env.COLLISION_FLAG)
        self.assertFalse(done)

    def test_step_slippery(self):
        self.env.is_slippery = True
        self.env.car_pos = (5, 5)
        with patch("numpy.random.rand", return_value=0.1):  # Mock random value to be less than 0.2
            state, reward, done, _ = self.env.step(0)  # Action 0: Up
            self.assertEqual(self.env.car_pos, (5, 5))  # Position should not change due to slip
            self.assertTrue(self.env.SLIPPERY_FLAG)
            self.assertFalse(done)
            self.assertIsInstance(reward, (int, float))

    def test_step_no_slip(self):
        self.env.is_slippery = True
        self.env.car_pos = (5, 5)
        with patch("numpy.random.rand", return_value=0.3):  # Mock random value to be greater than 0.2
            state, reward, done, _ = self.env.step(0)  # Action 0: Up
            self.assertEqual(self.env.car_pos, (4, 5))  # Position should change
            self.assertFalse(self.env.SLIPPERY_FLAG)
            self.assertFalse(done)
            self.assertIsInstance(reward, (int, float))

    def test_reset_resets_state(self):
        self.env.step(1)  # Move car
        self.env.reset()
        self.assertEqual(self.env.car_pos, self.env.init_pos)
        self.assertTrue(np.array_equal(self.env.visit_count, np.zeros(self.env.map_size)))

    @patch("matplotlib.pyplot.show")
    def test_render_displays_correct_map(self, mock_show):
        self.env.render("visit_count")
        mock_show.assert_called_once()

    def test_invalid_action_raises_value_error(self):
        with self.assertRaises(ValueError):
            self.env.step(5)  # Invalid action

# if __name__ == "__main__":
#     unittest.main()


def load_data(filename):
    with open(filename, 'r') as file:
        data = []
        for line in file:
            row = [int(x) for x in line.strip().split()]
            data.append(row)
    #print(data)
    return data

