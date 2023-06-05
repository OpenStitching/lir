import os
import unittest

import numpy as np

from .context import lir, pt1, pt2

TEST_DIR = os.path.abspath(os.path.dirname(__file__))

GRID = np.array(
    [
        [0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 1, 1, 0, 0, 0],
        [0, 0, 1, 1, 1, 1, 1, 0, 0],
        [0, 0, 1, 1, 1, 1, 1, 1, 0],
        [0, 0, 1, 1, 1, 1, 1, 1, 0],
        [0, 1, 1, 1, 1, 1, 1, 0, 0],
        [0, 0, 1, 1, 1, 1, 0, 0, 0],
        [0, 0, 1, 1, 1, 1, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 0, 0, 0],
        [1, 1, 0, 0, 0, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
    ],
    "bool",
)


class TestLIR(unittest.TestCase):
    def test_lir_polygon(self):
        polygon = np.array(
            [[[10, 10], [150, 10], [100, 100], [-40, 100]]], dtype=np.int32
        )

        rect = lir(polygon)
        np.testing.assert_array_equal(rect, np.array([10, 10, 91, 91]))

    def test_lir_binary_mask(self):
        rect = lir(GRID)
        np.testing.assert_array_equal(rect, np.array([2, 2, 4, 7]))

    def test_lir_binary_mask_with_contour(self):
        contour = np.array(
            [
                [2, 0],
                [2, 1],
                [2, 2],
                [2, 3],
                [2, 4],
                [1, 5],
                [2, 6],
                [2, 7],
                [1, 8],
                [0, 8],
                [0, 9],
                [1, 9],
                [2, 8],
                [3, 8],
                [4, 8],
                [5, 9],
                [6, 9],
                [7, 9],
                [8, 9],
                [7, 9],
                [6, 9],
                [5, 8],
                [5, 7],
                [5, 6],
                [6, 5],
                [7, 4],
                [7, 3],
                [6, 2],
                [5, 1],
                [4, 1],
                [3, 2],
                [2, 1],
            ],
            dtype=np.int32,
        )

        rect = lir(GRID, contour)
        np.testing.assert_array_equal(rect, np.array([2, 2, 4, 7]))

    def test_rectangle_pts(self):
        rect = np.array([10, 10, 91, 91])
        self.assertEqual(pt1(rect), (10, 10))
        self.assertEqual(pt2(rect), (100, 100))


def starttest():
    unittest.main()


if __name__ == "__main__":
    starttest()
