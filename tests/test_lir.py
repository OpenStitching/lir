import unittest
import os

import numpy as np
import cv2 as cv

from .context import lir

TEST_DIR = os.path.abspath(os.path.dirname(__file__))


class TestLIR(unittest.TestCase):

    def test_lir(self):

        cells = np.array([[0, 0, 1, 0, 0, 0, 0, 0, 0],
                          [0, 0, 1, 0, 1, 1, 0, 0, 0],
                          [0, 0, 1, 1, 1, 1, 1, 0, 0],
                          [0, 0, 1, 1, 1, 1, 1, 1, 0],
                          [0, 0, 1, 1, 1, 1, 1, 1, 0],
                          [0, 1, 1, 1, 1, 1, 1, 0, 0],
                          [0, 0, 1, 1, 1, 1, 0, 0, 0],
                          [0, 0, 1, 1, 1, 1, 0, 0, 0],
                          [1, 1, 1, 1, 1, 1, 0, 0, 0],
                          [1, 1, 0, 0, 0, 1, 1, 1, 1],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0]])
        cells = cells > 0

        h = lir.horizontal_adjacency(cells)
        v = lir.vertical_adjacency(cells)
        span_map = lir.span_map(cells, h, v)
        rect = lir.biggest_span_in_span_map(span_map)
        rect2 = lir.largest_interior_rectangle(cells)

        np.testing.assert_array_equal(rect, np.array([2, 2, 4, 7]))
        np.testing.assert_array_equal(rect, rect2)

    def test_spans(self):
        cells = np.array([[1, 1, 1],
                          [1, 1, 0],
                          [1, 0, 0],
                          [1, 0, 0],
                          [1, 0, 0],
                          [1, 1, 1]])
        cells = cells > 0

        h = lir.horizontal_adjacency(cells)
        v = lir.vertical_adjacency(cells)
        v_vector = lir.v_vector(v, 0, 0)
        h_vector = lir.h_vector(h, 0, 0)
        spans = lir.spans(h_vector, v_vector)

        np.testing.assert_array_equal(v_vector, np.array([6, 2, 1]))
        np.testing.assert_array_equal(h_vector, np.array([3, 2, 1]))
        np.testing.assert_array_equal(spans, np.array([[3, 1],
                                                       [2, 2],
                                                       [1, 6]]))

    def test_vector_size(self):
        t0 = np.array([1, 1, 1, 1], dtype=np.uint32)
        t1 = np.array([1, 1, 1, 0], dtype=np.uint32)
        t2 = np.array([1, 1, 0, 1, 1, 0], dtype=np.uint32)
        t3 = np.array([0, 0, 0, 0], dtype=np.uint32)
        t4 = np.array([0, 1, 1, 1], dtype=np.uint32)
        t5 = np.array([], dtype=np.uint32)

        self.assertEqual(lir.predict_vector_size(t0), 4)
        self.assertEqual(lir.predict_vector_size(t1), 3)
        self.assertEqual(lir.predict_vector_size(t2), 2)
        self.assertEqual(lir.predict_vector_size(t3), 0)
        self.assertEqual(lir.predict_vector_size(t4), 0)
        self.assertEqual(lir.predict_vector_size(t5), 0)

    def test_img(self):
        cells = cv.imread(os.path.join(TEST_DIR, "testdata", "mask.png"), 0)
        cells = cells > 0
        rect = lir.largest_interior_rectangle(cells)
        np.testing.assert_array_equal(rect, np.array([4, 20, 834, 213]))


def starttest():
    unittest.main()


if __name__ == "__main__":
    starttest()


# README PLOTS

# from matplotlib import pyplot as plt

# data = np.flip(cells, 0)

# # The normal figure
# fig = plt.figure(figsize=(16, 12))
# ax = fig.add_subplot(111)
# ax.axes.xaxis.set_visible(False)
# ax.axes.yaxis.set_visible(False)
# im = ax.imshow(data, origin='lower', interpolation='None', cmap='Greys_r')
# fig.colorbar(im)

# data = np.flip(h, 0)

# # The normal figure
# fig = plt.figure(figsize=(16, 12))
# ax = fig.add_subplot(111)
# ax.axes.xaxis.set_visible(False)
# ax.axes.yaxis.set_visible(False)
# im = ax.imshow(data, origin='lower', interpolation='None', cmap='viridis')
# fig.colorbar(im)

# data = np.flip(v, 0)

# # The normal figure
# fig = plt.figure(figsize=(16, 12))
# ax = fig.add_subplot(111)
# ax.axes.xaxis.set_visible(False)
# ax.axes.yaxis.set_visible(False)
# im = ax.imshow(data, origin='lower', interpolation='None', cmap='viridis')
# fig.colorbar(im)

# data = np.flip(v*h, 0)

# h_vec = lir.h_vector(h, 2, 2)
# v_vec = lir.v_vector(v, 2, 2)
# spans = lir.spans(h_vec, v_vec)

# data = np.flip(span_map[:, :, 0], 0)

# # The normal figure
# fig = plt.figure(figsize=(16, 12))
# ax = fig.add_subplot(111)
# ax.axes.xaxis.set_visible(False)
# ax.axes.yaxis.set_visible(False)
# im = ax.imshow(data, origin='lower', interpolation='None', cmap='viridis')
# fig.colorbar(im)


# data = np.flip(span_map[:, :, 1], 0)

# # The normal figure
# fig = plt.figure(figsize=(16, 12))
# ax = fig.add_subplot(111)
# ax.axes.xaxis.set_visible(False)
# ax.axes.yaxis.set_visible(False)
# im = ax.imshow(data, origin='lower', interpolation='None', cmap='viridis')
# fig.colorbar(im)


# data = np.flip(span_map[:, :, 0] * span_map[:, :, 1], 0)

# # The normal figure
# fig = plt.figure(figsize=(16, 12))
# ax = fig.add_subplot(111)
# ax.axes.xaxis.set_visible(False)
# ax.axes.yaxis.set_visible(False)
# im = ax.imshow(data, origin='lower', interpolation='None', cmap='viridis')
# fig.colorbar(im)
