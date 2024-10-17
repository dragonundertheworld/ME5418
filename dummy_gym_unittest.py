import unittest
import numpy as np
from ME5418.dummy_gym import DummyGym



class TestDummyGym(unittest.TestCase):
    def setUp(self):
        # Common initialization for each test
        self.env = DummyGym() # car_size=(2, 2),map_size=(40, 21)
        data = load_data('ME5418/test_map/map_2024-10-17_19-06-07.txt')
        arr_flat = np.array(data).flatten()
        arr_cropped = arr_flat[:40*21]
        self.map  = np.array(arr_cropped).reshape(40, 21)    
        # self.env.map = np.zeros((10, 10))  # No obstacles for simplicity in most tests
    
    def test_step_up_no_col(self):
        self.env.car.pos = (37, 2)
        state, reward, done, _ = self.env.step(0)  # Action 0: Up
        self.assertEqual(self.env.car.pos, (32, 2)) # step_size is 5
        self.assertFalse(done)
        self.assertIsInstance(reward, (int, float))
    
    def test_step_up_with_col(self): # error  do not consider the process of robot movement
        self.env.car.pos = (39, 6)
        state, reward, done, _ = self.env.step(0)  # Action 0: Up
        self.assertEqual(self.env.car.pos, (34, 6)) # step_size is 5
        self.assertFalse(done)
        self.assertIsInstance(reward, (int, float))

    # def test_step_up(self):
    #     self.env.car_pos = (5, 5)
    #     state, reward, done, _ = self.env.step(0)  # Action 0: Up
    #     self.assertEqual(self.env.car_pos, (4, 5))
    #     self.assertFalse(done)
    #     self.assertIsInstance(reward, (int, float))





def load_data(filename):
    with open(filename, 'r') as file:
        data = []
        for line in file:
            row = [int(x) for x in line.strip().split()]
            data.append(row)
    #print(data)
    return data

if __name__ == '__main__':
    #data = load_data('ME5418/test_map/map_2024-10-17_19-06-07.txt')
    unittest.main()

    data = load_data('ME5418/test_map/map_2024-10-17_19-06-07.txt')
    arr_flat = np.array(data).flatten()
    arr_cropped = arr_flat[:40*21]
    map = np.array(arr_cropped).reshape(40, 21)
    # print(map)
    
