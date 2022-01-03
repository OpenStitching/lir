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

    def test_cells(self):
        cells = np.array([[0, 0, 0, 1, 0, 1, 0, 0, 0],
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
        cells = np.uint8(cells * 255)

        rect = lir.largest_interior_rectangle(cells)
        np.testing.assert_array_equal(rect, np.array([2, 2, 4, 7]))

    def test_img(self):
        cells = cv.imread("mask.png", 0)
        rect = lir.largest_interior_rectangle(cells)
        np.testing.assert_array_equal(rect, np.array([4, 20, 834, 213]))

        # # PLOT
        # cells = cv.cvtColor(cells, cv.COLOR_GRAY2RGB)
        # start_point = tuple(rect[:2])
        # end_point = tuple(rect[:2] + rect[2:] - 1)
        # image = cv.rectangle(cells, start_point, end_point, (0, 0, 255), 1)
        # cv.imwrite("rect.png", image)


def starttest():
    unittest.main()


if __name__ == "__main__":
    starttest()


# # README PLOTS

# from matplotlib import pyplot as plt

# outline = lir.get_outline(cells)
# adjacencies = lir.adjacencies_all_directions(cells)
# s_map, d_map, saddle_candidates_map = lir.create_maps(outline, adjacencies)

# data = np.flip(cells, 0)

# # The normal figure
# fig = plt.figure(figsize=(16, 12))
# ax = fig.add_subplot(111)
# ax.axes.xaxis.set_visible(False)
# ax.axes.yaxis.set_visible(False)
# im = ax.imshow(data, origin='lower', interpolation='None', cmap='Greys_r')
# fig.colorbar(im)


# data = d_map

# cmap = plt.get_cmap('viridis')
# rgba = cmap(data)
# # Fill in colours for the out of range data:
# rgba[data==0, :] = [1, 1, 1, 1]
# rgba[data==1, :] = [0, 0, 1, 1]
# rgba[data==2, :] = [0, 1, 0, 1]
# rgba[data==3, :] = [1, 0, 0, 1]

# fig = plt.figure(figsize=(16, 12))
# ax = fig.add_subplot(111)
# ax.axes.xaxis.set_visible(False)
# ax.axes.yaxis.set_visible(False)
# im = ax.imshow(rgba)
# fig.colorbar(im)


# data = cells-saddle_candidates_map/2

# cmap = plt.get_cmap('viridis')
# rgba = cmap(data)
# # Fill in colours for the out of range data:
# rgba[data==0, :] = [0, 0, 0, 1]
# rgba[data==255, :] = [1, 1, 1, 1]
# rgba[data==127.5, :] = [1, 0, 0, 1]


# # The normal figure
# fig = plt.figure(figsize=(16, 12))
# ax = fig.add_subplot(111)
# ax.axes.xaxis.set_visible(False)
# ax.axes.yaxis.set_visible(False)
# im = ax.imshow(rgba)
# fig.colorbar(im)


# data = np.flip(s_map[:, :, 0], 0)

# # The normal figure
# fig = plt.figure(figsize=(16, 12))
# ax = fig.add_subplot(111)
# ax.axes.xaxis.set_visible(False)
# ax.axes.yaxis.set_visible(False)
# im = ax.imshow(data, origin='lower', interpolation='None', cmap='viridis')
# fig.colorbar(im)


# data = np.flip(s_map[:, :, 1], 0)

# # The normal figure
# fig = plt.figure(figsize=(16, 12))
# ax = fig.add_subplot(111)
# ax.axes.xaxis.set_visible(False)
# ax.axes.yaxis.set_visible(False)
# im = ax.imshow(data, origin='lower', interpolation='None', cmap='viridis')
# fig.colorbar(im)


# data = np.flip(s_map[:, :, 0] * s_map[:, :, 1], 0)

# # The normal figure
# fig = plt.figure(figsize=(16, 12))
# ax = fig.add_subplot(111)
# ax.axes.xaxis.set_visible(False)
# ax.axes.yaxis.set_visible(False)
# im = ax.imshow(data, origin='lower', interpolation='None', cmap='viridis')
# fig.colorbar(im)


# span_map = lir.span_map(adjacencies[0], adjacencies[2], lir.cells_of_interest(saddle_candidates_map))

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
