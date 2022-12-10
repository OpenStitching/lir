import unittest
import os

import numpy as np
import cv2 as cv

from .context import lir_within_polygon as lir

TEST_DIR = os.path.abspath(os.path.dirname(__file__))


class TestLIRwithinPolygon(unittest.TestCase):
    
    def test_create_mask_from_polygon(self):
        polygon = np.array([[
            [10,10],
            [150,10],
            [100,100],
            [-40,100]]
            ], dtype=np.int32)
        
        origin, mask = lir.create_mask_from_polygon(polygon)
        
        self.assertEqual(origin, (-40, 10))
        self.assertEqual(mask.shape, (91, 191))
        self.assertEqual(np.count_nonzero(mask == 255), 12831)
            
    def test_polygon(self):
        polygon = np.array([[
            [10,10],
            [150,10],
            [100,100],
            [-40,100]]
            ], dtype=np.int32 )

        rect = lir.largest_interior_rectangle(polygon)
        np.testing.assert_array_equal(rect, np.array([10, 10, 91, 91]))
        
    def test_polygon2(self):
        polygon = np.array([[
                [9,-7],
                [12,-6],
                [8,3],
                [10,6],
                [12,7],
                [1,9],
                [-8,7],
                [-6,6],
                [-4,6],
                [-6,2],
                [-6,0],
                [-7,-5],
                [-2,-7],
                [1,-3],
                [5,-7],
                [8,-4],
            ]], dtype=np.int32 )

        rect = lir.largest_interior_rectangle(polygon)
        np.testing.assert_array_equal(rect, np.array([-5, -3, 14, 12]))
    
    
    def test_img(self):
        grid = cv.imread(os.path.join(TEST_DIR, "testdata", "two_shapes.png"), 0)

        contours, _ = \
            cv.findContours(grid, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
        polygon = np.array([contours[0][:, 0, :]])

        rect = lir.largest_interior_rectangle(polygon)
        np.testing.assert_array_equal(rect, np.array([162,  62,  43,  44]))
        

def starttest():
    unittest.main()


if __name__ == "__main__":
    starttest()
