import math

import numpy as np


def _trim_grid(grid: np.array, center, target_ratio=0.00):
    assert target_ratio >= 0, "target_ratio cannot be negative!"
    rows, cols = grid.shape
    center_x, center_y = center

    left_edge = center_x
    for x in range(center_x, -1, -1):
        if not grid[center_y, x]:
            break
        left_edge = x

    right_edge = center_x
    for x in range(center_x, cols):
        if not grid[center_y, x]:
            break
        right_edge = x

    up_edge = center_y
    for y in range(center_y, -1, -1):
        if not grid[y, center_x]:
            break
        up_edge = y

    down_edge = center_y
    for y in range(center_y, rows):
        if not grid[y, center_x]:
            break
        down_edge = y

    half_trimmed_x_range = min(right_edge - center_x, center_x - left_edge)
    half_trimmed_y_range = min(down_edge - center_y, center_y - up_edge)
    x_min = max(center_x - half_trimmed_x_range, 0)
    x_max = min(center_x + half_trimmed_x_range, cols - 1)
    y_min = max(center_y - half_trimmed_y_range, 0)
    y_max = min(center_y + half_trimmed_y_range, rows - 1)

    if target_ratio:
        w = x_max - x_min
        h = y_max - y_min
        if w > h:
            w = h * target_ratio
            x_min = int(center_x - w // 2)
            x_max = int(center_x + w // 2)
        else:
            h = w / target_ratio
            y_min = int(center_y - h // 2)
            y_max = int(center_y + h // 2)

    trimmed_grid = np.zeros_like(grid)
    trimmed_grid[y_min : y_max + 1, x_min : x_max + 1] = grid[
        y_min : y_max + 1, x_min : x_max + 1
    ]
    return trimmed_grid


def _get_step(target_ratio, step_max=20):
    scale_factor = 100000  # Scale up to avoid floating point precision issues
    x_step = y_step = step_max + 1
    while (x_step > step_max or y_step > step_max) and scale_factor:
        width = target_ratio
        height = int(scale_factor)
        width = int(width * scale_factor)
        width = width + 1 if width % 2 else width

        gcd = math.gcd(width, height)  # Compute the GCD

        x_step = width // gcd
        y_step = height // gcd
        scale_factor /= 10

    return x_step, y_step


def largest_interior_rectangle(
    grid: np.array, target_ratio=0.00, ratio_tolerance=0.01, target_center_coord=None
) -> np.array:
    rows, cols = grid.shape
    max_area = 0
    max_rect = (0, 0, 0, 0)  # (x1, y1, x2, y2)

    if target_center_coord:
        # Check if center_coord is in the grid
        center_x, center_y = target_center_coord
        if (
            0 <= center_x < cols
            and 0 <= center_y < rows
            and grid[center_y, center_x] == 1
        ):
            grid = _trim_grid(grid, target_center_coord, target_ratio=target_ratio)
            x_step, y_step = _get_step(target_ratio)
            x_extend = y_extend = 0
            # TODO: simplify this ugly while and that ugly if in it
            while (
                center_y + y_extend < rows
                and center_y - y_extend >= 0
                and center_x + x_extend < cols
                and center_x - x_extend >= 0
                and grid[center_y + y_extend, center_x]
                and grid[center_y - y_extend, center_x]
                and grid[center_y, center_x + x_extend]
                and grid[center_y, center_x - x_extend]
            ):
                y_extend += y_step
                x_extend += x_step
                if (
                    center_y - y_extend >= 0
                    and center_y + y_extend < rows
                    and center_x - x_extend >= 0
                    and center_x + x_extend < cols
                    and grid[
                        center_y - y_extend : center_y + y_extend + 1,
                        center_x - x_extend : center_x + x_extend + 1,
                    ].all()
                ):
                    max_area = max(max_area, (x_extend * 2) * (y_extend * 2))
                    max_rect = (
                        center_x - x_extend,
                        center_y - y_extend,
                        center_x + x_extend,
                        center_y + y_extend,
                    )
        else:
            print(f"⚠️Center coordinate is not in the grid! ({grid.shape = })")

    else:  # auto mode
        # ref: LeetCode 84. Largest Rectangle in Histogram
        heights = np.zeros(cols)
        for y2 in range(rows):
            if (
                rows - y2
            ) * cols < max_area:  # The remaining area is smaller than the max area
                break

            heights = (heights + 1) * grid[y2, :]  # compute the heights (histogram)
            stack = []  # reset stack at every row
            for x2 in range(cols + 1):
                if (rows - y2) * (
                    cols + 1 - x2
                ) < max_area:  # The remaining area is smaller than the max area
                    break

                while stack and (x2 == cols or heights[stack[-1]] > heights[x2]):
                    height = heights[stack.pop()]
                    width = x2 if not stack else x2 - stack[-1] - 1
                    curr_ratio = (
                        width / height if height > 0 else 0
                    )  # prevent ZeroDivisionError
                    if (
                        target_ratio * (1 - ratio_tolerance)
                        <= curr_ratio
                        <= target_ratio * (1 + ratio_tolerance)
                        or target_ratio == 0
                    ):
                        area = width * height
                        x1 = 0 if not stack else stack[-1] + 1
                        rect = (x1, y2 - height + 1, x2 - 1, y2)
                        if area > max_area:
                            max_area = area
                            max_rect = rect

                stack.append(x2)

    x1, y1, x2, y2 = max_rect
    return np.array([x1, y1, x2 - x1, y2 - y1], dtype="int")
