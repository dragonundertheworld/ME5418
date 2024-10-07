import unittest
from dummy_gym import DummyGym
from dummy_gym import INIT_POS, CAR_SIZE, STEP_SIZE, MAP_SIZE, NUM_OF_OBSTACLES, FOV

class TestDummyGym(unittest.TestCase):

    def setUp(self):
        self.env = DummyGym(map_size=(10, 10), num_of_obstacles=5, FOV=(5, 5))

    def test_initialization(self):
        self.assertEqual(self.env.car_pos, (2, 3))
        self.assertEqual(self.env.map_size, (10, 10))
        self.assertEqual(self.env.num_of_obstacles, 5)
        self.assertEqual(self.env.FOV, (5, 5))
        self.assertEqual(self.env.visit_count.shape, (10, 10))
        self.assertEqual(self.env.action_space.n, 4)
        self.assertEqual(self.env.observation_space.shape, (5, 5))

    def test_step_function(self):
        initial_state = self.env.state
        state, reward, done, _ = self.env.step(0)  # Move up
        self.assertNotEqual(state, initial_state)
        self.assertIsInstance(reward, float)
        self.assertIsInstance(done, bool)

    def test_reset_function(self):
        self.env.step(0)  # Move up
        state_after_step = self.env.state
        reset_state = self.env.reset()
        self.assertNotEqual(state_after_step, reset_state)
        self.assertEqual(reset_state[0].shape, (10, 10))  # visit_count shape
        self.assertEqual(reset_state[1].shape, (5, 5))  # fov_map shape

    def test_render_function(self):
        try:
            self.env.render('Map')
            self.env.render('fov_map')
            self.env.render('visit_count')
        except Exception as e:
            self.fail(f"Render function raised an exception: {e}")

if __name__ == '__main__':
    unittest.main()