import unittest
import os
import sys

import numpy as np
import cv2 as cv

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                '..')))

import lir_within_outline as lir

# %%


class TestLIR(unittest.TestCase):

    def test_multiple_shapes(self):
        cells = cv.imread("two_shapes.png", 0)

        outlines = list(lir.get_outlines(cells))
        rect1 = lir.largest_interior_rectangle(cells, outlines[0])
        rect2 = lir.largest_interior_rectangle(cells, outlines[1])

        np.testing.assert_array_equal(rect1, np.array([162, 62, 43, 44]))
        np.testing.assert_array_equal(rect2, np.array([95, 62, 43, 44]))

        # # PLOT
        # cells = cv.cvtColor(cells, cv.COLOR_GRAY2RGB)
        # start_point = tuple(rect1[:2])
        # end_point = tuple(rect1[:2] + rect1[2:] - 1)
        # image = cv.rectangle(cells, start_point, end_point, (0, 0, 255), 1)
        # start_point = tuple(rect2[:2])
        # end_point = tuple(rect2[:2] + rect2[2:] - 1)
        # image = cv.rectangle(image, start_point, end_point, (0, 0, 255), 1)
        # cv.imwrite("rect.png", image)


def starttest():
    unittest.main()


if __name__ == "__main__":
    starttest()
