import unittest
import numpy as np
from dummy_gym import DummyGym

I_need_render = False

class TestDummyGym(unittest.TestCase):
    def setUp(self):
        # Common initialization for each test
        self.env = DummyGym(map_file_path='./test_map/map_2024-10-17_19-06-07.txt') # car_size=(2, 2),map_size=(40, 21)
    
    def test_step_up_no_col(self):
        self.env.car.pos = (34, 2)
        self.env.render() if I_need_render else None
        state, reward, done, _ = self.env.step(0)  # Action 0: Up
        self.env.render() if I_need_render else None
        self.assertEqual(self.env.car.pos, (29, 2)) # step_size is 5
        self.assertFalse(done)
        self.assertIsInstance(reward, (int, float))
        print('test step up no collision done##############################################')
    
    def test_step_up_with_col(self): # error  do not consider the process of robot movement
        self.env.car.pos = (45, 6)
        self.env.render() if I_need_render else None
        state, reward, done, _ = self.env.step(0)  # Action 0: Up
        self.env.render() if I_need_render else None
        self.assertEqual(self.env.car.pos, (45, 6)) # step_size is 5
        self.assertFalse(done)
        self.assertIsInstance(reward, (int, float))
        print('test step up with collision done##############################################')

    def test_step_up_with_col_boundary(self):
        self.env.car.pos = (3, 2)
        self.env.render() if I_need_render else None
        state, reward, done, _ = self.env.step(0)
        self.env.render() if I_need_render else None
        self.assertEqual(self.env.car.pos, (3, 2))
        self.assertFalse(done)
        self.assertIsInstance(reward, (int, float))
        print('test step up with collision boundary done##############################################')

    def test_step_down_no_col(self):
        self.env.car.pos = (5, 5)
        self.env.render() if I_need_render else None
        state, reward, done, _ = self.env.step(1)
        self.env.render() if I_need_render else None
        self.assertEqual(self.env.car.pos, (10, 5))
        self.assertFalse(done)
        self.assertIsInstance(reward, (int, float))
        print('test step down no collision done##############################################')
    
    def test_step_down_with_col(self):
        self.env.car.pos = (1, 9)
        self.env.render() if I_need_render else None
        state, reward, done, _ = self.env.step(1)
        self.env.render() if I_need_render else None
        self.assertEqual(self.env.car.pos, (1, 9))
        self.assertFalse(done)
        self.assertIsInstance(reward, (int, float))
        print('test step down with collision done##############################################')

    def test_step_down_with_col_boundary(self):
        self.env.car.pos = (self.env.map_size[0]-3, 2)
        self.env.render() if I_need_render else None
        state, reward, done, _ = self.env.step(1)
        self.env.render() if I_need_render else None
        self.assertEqual(self.env.car.pos, (self.env.map_size[0]-3, 2))
        self.assertFalse(done)
        self.assertIsInstance(reward, (int, float))
        print('test step down with collision boundary done##############################################')
    
    def test_step_left_no_col(self):
        self.env.car.pos = (10, 7)
        self.env.render() if I_need_render else None
        state, reward, done, _ = self.env.step(2)
        self.env.render() if I_need_render else None
        self.assertEqual(self.env.car.pos, (10, 2))
        self.assertFalse(done)
        self.assertIsInstance(reward, (int, float))
        print('test step left no collision done##############################################')
    
    def test_step_left_with_col(self):
        self.env.car.pos = (5, 14)
        self.env.render() if I_need_render else None
        state, reward, done, _ = self.env.step(2)
        self.env.render() if I_need_render else None
        self.assertEqual(self.env.car.pos, (5, 14))
        self.assertFalse(done)
        self.assertIsInstance(reward, (int, float))
        print('test step left with collision done##############################################')
    
    def test_step_left_with_col_boundary(self):
        self.env.car.pos = (3, 3)
        self.env.render() if I_need_render else None
        state, reward, done, _ = self.env.step(2)
        self.env.render() if I_need_render else None
        self.assertEqual(self.env.car.pos, (3, 3))
        self.assertFalse(done)
        self.assertIsInstance(reward, (int, float))
        print('test step left with collision boundary done##############################################')
    
    def test_step_right_no_col(self):
        self.env.car.pos = (2, 2)
        self.env.render() if I_need_render else None
        state, reward, done, _ = self.env.step(3)
        self.env.render() if I_need_render else None
        self.assertEqual(self.env.car.pos, (2, 7))
        self.assertFalse(done)
        self.assertIsInstance(reward, (int, float))
        print('test step right no collision done##############################################')
    
    def test_step_right_with_col(self):
        self.env.car.pos = (4, 4)
        self.env.render() if I_need_render else None
        state, reward, done, _ = self.env.step(3)
        self.env.render() if I_need_render else None
        self.assertEqual(self.env.car.pos, (4, 4))
        self.assertFalse(done)
        self.assertIsInstance(reward, (int, float))
        print('test step right with collision done##############################################')
    
    def test_step_right_with_col_boundary(self):
        self.env.car.pos = (2, self.env.map_size[1]-2)
        self.env.render() if I_need_render else None
        state, reward, done, _ = self.env.step(3)
        self.env.render() if I_need_render else None
        self.assertEqual(self.env.car.pos, (2, self.env.map_size[1]-2))
        self.assertFalse(done)
        self.assertIsInstance(reward, (int, float))
        print('test step right with collision boundary done##############################################')
    
    def test_reset(self):
        self.env.car.pos = (37, 7)
        self.env.render() if I_need_render else None
        state = self.env.reset()
        self.env.render() if I_need_render else None
        self.assertEqual(self.env.car.pos, (12, 10))
        print('test reset done##############################################')


if __name__ == '__main__':
    unittest.main()
    # print(map)
    
