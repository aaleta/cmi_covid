import unittest
import numpy as np
import pandas as pd
from utils import extract_wave, extract_target, lag_data, get_generation_time
from pandas.testing import assert_frame_equal
from numpy.testing import assert_array_equal, assert_array_almost_equal


class TestExtractData(unittest.TestCase):
    @classmethod
    def setUp(self):
        self.max_backwards = 3
        self.data = pd.read_csv('data/test_data.csv')

    def test_wave_1(self):
        obtained = extract_wave(self.data, 1, self.max_backwards)
        expected = pd.DataFrame({'0-9': [0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                                 '10-19': [0, 0, 0, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11]
                                 })
        assert_frame_equal(obtained, expected)

    def test_wave_2(self):
        obtained = extract_wave(self.data, 2, self.max_backwards)
        expected = pd.DataFrame({'0-9': [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
                                 '10-19': [13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
                                 })
        assert_frame_equal(obtained, expected)


class TestExtractTarget(unittest.TestCase):
    def setUp(self):
        self.max_backwards = 3
        self.data = pd.read_csv('data/test_data.csv')

    def test_extraction(self):
        data = extract_wave(self.data, 1, self.max_backwards)
        obtained = extract_target(data, '0-9', self.max_backwards)
        expected = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]])
        assert_array_equal(obtained, expected)


class TestLagData(unittest.TestCase):
    def setUp(self):
        self.col = '0-9'
        self.max_backwards = 3
        self.data = pd.read_csv('data/test_data.csv')

    def test_no_gt(self):
        w_gt = None
        data = extract_wave(self.data, 1, self.max_backwards)
        obtained = lag_data(data, self.col, self.max_backwards, w_gt)
        expected = np.array([[0, 0, 0], [1, 0, 0], [2, 1, 0], [3, 2, 1], [4, 3, 2], [5, 4, 3],
                             [6, 5, 4], [7, 6, 5], [8, 7, 6], [9, 8, 7]])
        assert_array_equal(obtained, expected)

    def test_gt_exp(self):
        w_gt_theo = [0.6652, 0.2447, 0.0900]
        _, w_gt_obs = get_generation_time('exponential', [1], self.max_backwards)
        assert_array_almost_equal(w_gt_theo, w_gt_obs, 3)

    def test_gt_gamma(self):
        w_gt_theo = [0.2557, 0.3568, 0.3875]
        _, w_gt_obs = get_generation_time('gamma', [1.87, 1.0/0.27], self.max_backwards)
        assert_array_almost_equal(w_gt_theo, w_gt_obs, 3)

    def test_gt(self):
        w_gt = [1, 0.37, 0.14] # exp rate 1
        data = extract_wave(self.data, 1, self.max_backwards)
        obtained = lag_data(data, self.col, self.max_backwards, w_gt)
        expected = np.array([[np.dot([0, 0, 0], w_gt)], [np.dot([1, 0, 0], w_gt)],
                                [np.dot([2, 1, 0], w_gt)], [np.dot([3, 2, 1], w_gt)],
                                [np.dot([4, 3, 2], w_gt)], [np.dot([5, 4, 3], w_gt)],
                                [np.dot([6, 5, 4], w_gt)], [np.dot([7, 6, 5], w_gt)],
                                [np.dot([8, 7, 6], w_gt)], [np.dot([9, 8, 7], w_gt)]])
        assert_array_equal(obtained, expected)


if __name__ == '__main__':
    unittest.main()
