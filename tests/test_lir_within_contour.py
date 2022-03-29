import unittest
import os

import numpy as np
import cv2 as cv

from .context import lir_within_contour as lir

TEST_DIR = os.path.abspath(os.path.dirname(__file__))


class TestLIRwithinOutlines(unittest.TestCase):

    def test_grid(self):
        grid = np.array([[0, 0, 0, 1, 0, 1, 0, 0, 0],
                         [0, 0, 1, 1, 0, 1, 1, 0, 0],
                         [0, 1, 1, 1, 1, 1, 1, 1, 0],
                         [1, 1, 1, 1, 1, 1, 1, 1, 1],
                         [1, 1, 1, 1, 1, 1, 0, 0, 0],
                         [0, 0, 1, 1, 1, 1, 1, 0, 0],
                         [1, 1, 1, 1, 1, 1, 1, 1, 1],
                         [1, 1, 1, 1, 1, 1, 1, 1, 1],
                         [0, 1, 1, 1, 1, 1, 1, 1, 0],
                         [0, 0, 1, 1, 0, 1, 1, 0, 0],
                         [0, 0, 0, 1, 0, 1, 0, 0, 0]])

        grid = np.uint8(grid * 255)

        contours, _ = \
            cv.findContours(grid, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
        contour = contours[0][:, 0, :]

        grid = grid > 0

        rect = lir.largest_interior_rectangle(grid, contour)

        np.testing.assert_array_equal(rect, np.array([2, 2, 4, 7]))

    def test_img(self):
        grid = cv.imread(os.path.join(TEST_DIR, "testdata", "mask.png"), 0)

        contours, _ = \
            cv.findContours(grid, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
        contour = contours[0][:, 0, :]

        grid = grid > 0

        rect = lir.largest_interior_rectangle(grid, contour)

        np.testing.assert_array_equal(rect, np.array([4, 20, 834, 213]))


class TestLIRwithinMultipleOutlines(unittest.TestCase):

    def test_multiple_shapes(self):
        grid = cv.imread(os.path.join(TEST_DIR, "testdata", "two_shapes.png"),
                         0)

        contours, _ = \
            cv.findContours(grid, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
        contour1 = contours[0][:, 0, :]
        contour2 = contours[1][:, 0, :]

        grid = grid > 0

        rect1 = lir.largest_interior_rectangle(grid, contour1)
        rect2 = lir.largest_interior_rectangle(grid, contour2)

        np.testing.assert_array_equal(rect1, np.array([162, 62, 43, 44]))
        np.testing.assert_array_equal(rect2, np.array([95, 62, 43, 44]))


def starttest():
    unittest.main()


if __name__ == "__main__":
    starttest()
