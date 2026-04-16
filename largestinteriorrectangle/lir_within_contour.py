import numba as nb
import numpy as np

from .lir_basis import (
    biggest_span_in_span_map,
    biggest_span_in_span_map_closest_to_center,
)
from .lir_basis import h_vector as h_vector_top2bottom
from .lir_basis import horizontal_adjacency as horizontal_adjacency_left2right
from .lir_basis import predict_vector_size, span_map, spans
from .lir_basis import v_vector as v_vector_left2right
from .lir_basis import vertical_adjacency as vertical_adjacency_top2bottom


def largest_interior_rectangle(
    grid, contour, target_ratio=None, target_center=None, tolerance=None
):
    adjacencies = adjacencies_all_directions(grid)
    contour = contour.astype("uint32", order="C")

    target_ratio = float(target_ratio) if target_ratio is not None else 0

    s_map1, _, saddle_candidates_map = create_maps(adjacencies, contour, target_ratio)

    s_map2 = span_map(
        saddle_candidates_map, adjacencies[0], adjacencies[2], target_ratio
    )

    areas1 = s_map1[:, :, 0] * s_map1[:, :, 1]
    areas2 = s_map2[:, :, 0] * s_map2[:, :, 1]
    mask = areas2 > areas1

    s_map_combined = s_map1.copy()
    s_map_combined[mask] = s_map2[mask]

    if target_center is not None:
        lir = biggest_span_in_span_map_closest_to_center(
            s_map_combined, target_center, tolerance
        )
    else:
        lir = biggest_span_in_span_map(s_map_combined)

    return lir


@nb.njit("uint32[:,::1](boolean[:,::1])", parallel=True, cache=True)
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


@nb.njit("uint32[:,::1](boolean[:,::1])", parallel=True, cache=True)
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


@nb.njit("uint32[:](uint32[:,::1], uint32, uint32)", cache=True)
def h_vector_bottom2top(h_adjacency, x, y):
    vector_size = predict_vector_size(np.flip(h_adjacency[: y + 1, x]))
    h_vector = np.zeros(vector_size, dtype=np.uint32)
    h = np.inf
    for p in range(vector_size):
        h = np.minimum(h_adjacency[y - p, x], h)
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


@nb.njit("uint32[:](uint32[:,::1], uint32, uint32)", cache=True)
def v_vector_right2left(v_adjacency, x, y):
    vector_size = predict_vector_size(np.flip(v_adjacency[y, : x + 1]))
    v_vector = np.zeros(vector_size, dtype=np.uint32)
    v = np.inf
    for q in range(vector_size):
        v = np.minimum(v_adjacency[y, x - q], v)
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
def get_top_left(x, y, w, h, direction_idx):
    """0: none, 1: flip x, 2: flip y, 3: flip both"""
    tx = x
    ty = y
    if direction_idx == 1:
        tx = x - w + 1
    elif direction_idx == 2:
        ty = y - h + 1
    elif direction_idx == 3:
        tx = x - w + 1
        ty = y - h + 1
    return tx, ty


@nb.njit(cache=True)
def cell_on_contour(x, y, contour):
    x_true = contour[:, 0] == x
    y_true = contour[:, 1] == y
    both_true = np.logical_and(x_true, y_true)
    return np.any(both_true)


@nb.njit(
    "Tuple((uint32[:,:,::1], uint8[:,::1], boolean[:,::1]))"
    "(UniTuple(uint32[:,::1], 4), uint32[:,::1], float64)",
    parallel=True,
    cache=True,
)
def create_maps(adjacencies, contour, target_ratio):
    h_left2right, h_right2left, v_top2bottom, v_bottom2top = adjacencies

    shape = h_left2right.shape
    span_map = np.zeros(shape + (2,), "uint32")
    direction_map = np.zeros(shape, "uint8")
    saddle_candidates_map = np.zeros(shape, "bool_")
    constrained = target_ratio > 0

    contour_grid = np.zeros(shape, "bool_")
    for i in range(len(contour)):
        contour_grid[contour[i, 1], contour[i, 0]] = True

    for idx in nb.prange(len(contour)):
        x, y = contour[idx, 0], contour[idx, 1]
        h_vectors = h_vectors_all_directions(h_left2right, h_right2left, x, y)
        v_vectors = v_vectors_all_directions(v_top2bottom, v_bottom2top, x, y)
        span_arrays = spans_all_directions(h_vectors, v_vectors)
        n = get_n_directions(span_arrays)
        direction_map[y, x] = n

        for direction_idx in range(4):
            span_array = span_arrays[direction_idx]
            for span_idx in range(span_array.shape[0]):
                w_max, h_max = span_array[span_idx][0], span_array[span_idx][1]

                if constrained:
                    w1 = w_max
                    h1 = int(w1 / target_ratio)
                    if h1 > h_max:
                        h1 = h_max
                        w1 = int(h1 * target_ratio)

                    h2 = h_max
                    w2 = int(h2 * target_ratio)
                    if w2 > w_max:
                        w2 = w_max
                        h2 = int(w2 / target_ratio)

                    tx1, ty1 = get_top_left(x, y, w1, h1, direction_idx)
                    if w1 * h1 > span_map[ty1, tx1, 0] * span_map[ty1, tx1, 1]:
                        span_map[ty1, tx1, :] = np.array([w1, h1], "uint32")

                    tx2, ty2 = get_top_left(x, y, w2, h2, direction_idx)
                    if w2 * h2 > span_map[ty2, tx2, 0] * span_map[ty2, tx2, 1]:
                        span_map[ty2, tx2, :] = np.array([w2, h2], "uint32")
                else:
                    w, h = w_max, h_max
                    tx, ty = get_top_left(x, y, w, h, direction_idx)
                    if w * h > span_map[ty, tx, 0] * span_map[ty, tx, 1]:
                        span_map[ty, tx, :] = np.array([w, h], "uint32")

            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < shape[0] and 0 <= nx < shape[1]:
                        if h_left2right[ny, nx] > 0 and not contour_grid[ny, nx]:
                            saddle_candidates_map[ny, nx] = True

    return span_map, direction_map, saddle_candidates_map


def biggest_rectangle(*args):
    biggest_rect = np.array([0, 0, 0, 0], dtype=np.uint32)
    for rect in args:
        if rect[2] * rect[3] > biggest_rect[2] * biggest_rect[3]:
            biggest_rect = rect
    return biggest_rect


def rectangle_center_distance(rect, target_center):
    rcx = float(rect[0]) + (float(rect[2]) - 1.0) / 2.0
    rcy = float(rect[1]) + (float(rect[3]) - 1.0) / 2.0
    dx = rcx - float(target_center[0])
    dy = rcy - float(target_center[1])
    return dx * dx + dy * dy


def choose_rectangle(rect1, rect2, target_center=None):
    if target_center is None:
        return biggest_rectangle(rect1, rect2)

    d1 = rectangle_center_distance(rect1, target_center)
    d2 = rectangle_center_distance(rect2, target_center)
    if d1 < d2:
        return rect1
    if d2 < d1:
        return rect2

    a1 = rect1[2] * rect1[3]
    a2 = rect2[2] * rect2[3]
    if a1 >= a2:
        return rect1
    return rect2
