import unittest
from preprocess import PreProcess
import numpy as np
test_class = PreProcess()


class TestOutlierDetection(unittest.TestCase):
    def test_detect_outlier(self):

        data = [1, 2, 3, 4, 100, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
        outliers = test_class.detect_outliers(data)
        self.assertEqual(outliers, np.array([4]))

        data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 1000]
        outliers = test_class.detect_outliers(data)
        self.assertEqual(outliers, np.array([20]))

# Many such cases can be written to test function/method level working


if __name__ == '__main__':
    unittest.main()
