import numpy as np
import numba as nb

from .lir_basis import horizontal_adjacency as horizontal_adjacency_left2right
from .lir_basis import vertical_adjacency as vertical_adjacency_top2bottom
from .lir_basis import h_vector as h_vector_top2bottom
from .lir_basis import v_vector as v_vector_left2right
from .lir_basis import (predict_vector_size, spans, span_map,
                        biggest_span_in_span_map)


def largest_interior_rectangle(grid, contour):
    adjacencies = adjacencies_all_directions(grid)
    contour = contour.astype("uint32", order="C")

    s_map, _, saddle_candidates_map = create_maps(adjacencies, contour)
    lir1 = biggest_span_in_span_map(s_map)

    s_map = span_map(saddle_candidates_map, adjacencies[0], adjacencies[2])
    lir2 = biggest_span_in_span_map(s_map)

    lir = biggest_rectangle(lir1, lir2)
    return lir


@nb.njit('uint32[:,::1](boolean[:,::1])', parallel=True, cache=True)
def horizontal_adjacency_right2left(grid):
    result = np.zeros(grid.shape, dtype=np.uint32)
    for y in nb.prange(grid.shape[0]):
        span = 0
        for x in range(grid.shape[1]):
            if grid[y, x]:
                span += 1
            else:
                span = 0
            result[y, x] = span
    return result


@nb.njit('uint32[:,::1](boolean[:,::1])', parallel=True, cache=True)
def vertical_adjacency_bottom2top(grid):
    result = np.zeros(grid.shape, dtype=np.uint32)
    for x in nb.prange(grid.shape[1]):
        span = 0
        for y in range(grid.shape[0]):
            if grid[y, x]:
                span += 1
            else:
                span = 0
            result[y, x] = span
    return result


@nb.njit(cache=True)
def adjacencies_all_directions(grid):
    h_left2right = horizontal_adjacency_left2right(grid)
    h_right2left = horizontal_adjacency_right2left(grid)
    v_top2bottom = vertical_adjacency_top2bottom(grid)
    v_bottom2top = vertical_adjacency_bottom2top(grid)
    return h_left2right, h_right2left, v_top2bottom, v_bottom2top


@nb.njit('uint32[:](uint32[:,::1], uint32, uint32)', cache=True)
def h_vector_bottom2top(h_adjacency, x, y):
    vector_size = predict_vector_size(np.flip(h_adjacency[:y+1, x]))
    h_vector = np.zeros(vector_size, dtype=np.uint32)
    h = np.Inf
    for p in range(vector_size):
        h = np.minimum(h_adjacency[y-p, x], h)
        h_vector[p] = h
    h_vector = np.unique(h_vector)[::-1]
    return h_vector


@nb.njit(cache=True)
def h_vectors_all_directions(h_left2right, h_right2left, x, y):
    h_l2r_t2b = h_vector_top2bottom(h_left2right, x, y)
    h_r2l_t2b = h_vector_top2bottom(h_right2left, x, y)
    h_l2r_b2t = h_vector_bottom2top(h_left2right, x, y)
    h_r2l_b2t = h_vector_bottom2top(h_right2left, x, y)
    return h_l2r_t2b, h_r2l_t2b, h_l2r_b2t, h_r2l_b2t


@nb.njit('uint32[:](uint32[:,::1], uint32, uint32)', cache=True)
def v_vector_right2left(v_adjacency, x, y):
    vector_size = predict_vector_size(np.flip(v_adjacency[y, :x+1]))
    v_vector = np.zeros(vector_size, dtype=np.uint32)
    v = np.Inf
    for q in range(vector_size):
        v = np.minimum(v_adjacency[y, x-q], v)
        v_vector[q] = v
    v_vector = np.unique(v_vector)[::-1]
    return v_vector


@nb.njit(cache=True)
def v_vectors_all_directions(v_top2bottom, v_bottom2top, x, y):
    v_l2r_t2b = v_vector_left2right(v_top2bottom, x, y)
    v_r2l_t2b = v_vector_right2left(v_top2bottom, x, y)
    v_l2r_b2t = v_vector_left2right(v_bottom2top, x, y)
    v_r2l_b2t = v_vector_right2left(v_bottom2top, x, y)
    return v_l2r_t2b, v_r2l_t2b, v_l2r_b2t, v_r2l_b2t


@nb.njit(cache=True)
def spans_all_directions(h_vectors, v_vectors):
    span_l2r_t2b = spans(h_vectors[0], v_vectors[0])
    span_r2l_t2b = spans(h_vectors[1], v_vectors[1])
    span_l2r_b2t = spans(h_vectors[2], v_vectors[2])
    span_r2l_b2t = spans(h_vectors[3], v_vectors[3])
    return span_l2r_t2b, span_r2l_t2b, span_l2r_b2t, span_r2l_b2t


@nb.njit(cache=True)
def get_n_directions(spans_all_directions):
    n_directions = 1
    for direction_spans in spans_all_directions:
        all_x_1 = np.all(direction_spans[:, 0] == 1)
        all_y_1 = np.all(direction_spans[:, 1] == 1)
        if not all_x_1 and not all_y_1:
            n_directions += 1
    return n_directions


@nb.njit(cache=True)
def get_xy_array(x, y, spans, mode=0):
    """0 - flip none, 1 - flip x, 2 - flip y, 3 - flip both"""
    xy = spans.copy()
    xy[:, 0] = x
    xy[:, 1] = y
    if mode == 1:
        xy[:, 0] = xy[:, 0] - spans[:, 0] + 1
    if mode == 2:
        xy[:, 1] = xy[:, 1] - spans[:, 1] + 1
    if mode == 3:
        xy[:, 0] = xy[:, 0] - spans[:, 0] + 1
        xy[:, 1] = xy[:, 1] - spans[:, 1] + 1
    return xy


@nb.njit(cache=True)
def get_xy_arrays(x, y, spans_all_directions):
    xy_l2r_t2b = get_xy_array(x, y, spans_all_directions[0], 0)
    xy_r2l_t2b = get_xy_array(x, y, spans_all_directions[1], 1)
    xy_l2r_b2t = get_xy_array(x, y, spans_all_directions[2], 2)
    xy_r2l_b2t = get_xy_array(x, y, spans_all_directions[3], 3)
    return xy_l2r_t2b, xy_r2l_t2b, xy_l2r_b2t, xy_r2l_b2t


@nb.njit(cache=True)
def cell_on_contour(x, y, contour):
    x_true = contour[:, 0] == x
    y_true = contour[:, 1] == y
    both_true = np.logical_and(x_true, y_true)
    return np.any(both_true)


@nb.njit('Tuple((uint32[:,:,::1], uint8[:,::1], boolean[:,::1]))'
         '(UniTuple(uint32[:,::1], 4), uint32[:,::1])',
         parallel=True, cache=True)
def create_maps(adjacencies, contour):
    h_left2right, h_right2left, v_top2bottom, v_bottom2top = adjacencies

    shape = h_left2right.shape
    span_map = np.zeros(shape + (2,), "uint32")
    direction_map = np.zeros(shape, "uint8")
    saddle_candidates_map = np.zeros(shape, "bool_")

    for idx in nb.prange(len(contour)):
        x, y = contour[idx, 0], contour[idx, 1]
        h_vectors = h_vectors_all_directions(h_left2right, h_right2left, x, y)
        v_vectors = v_vectors_all_directions(v_top2bottom, v_bottom2top, x, y)
        span_arrays = spans_all_directions(h_vectors, v_vectors)
        n = get_n_directions(span_arrays)
        direction_map[y, x] = n
        xy_arrays = get_xy_arrays(x, y, span_arrays)
        for direction_idx in range(4):
            xy_array = xy_arrays[direction_idx]
            span_array = span_arrays[direction_idx]
            for span_idx in range(span_array.shape[0]):
                x, y = xy_array[span_idx][0], xy_array[span_idx][1]
                w, h = span_array[span_idx][0], span_array[span_idx][1]
                if w*h > span_map[y, x, 0] * span_map[y, x, 1]:
                    span_map[y, x, :] = np.array([w, h], "uint32")
                if n == 3 and not cell_on_contour(x, y, contour):
                    saddle_candidates_map[y, x] = True

    return span_map, direction_map, saddle_candidates_map


def biggest_rectangle(*args):
    biggest_rect = np.array([0, 0, 0, 0], dtype=np.uint32)
    for rect in args:
        if rect[2] * rect[3] > biggest_rect[2] * biggest_rect[3]:
            biggest_rect = rect
    return biggest_rect
