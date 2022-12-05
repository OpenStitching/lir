import numpy as np
import cv2 as cv

from .lir_within_contour import largest_interior_rectangle as lir_contour


def largest_interior_rectangle(polygon):
    origin, mask = create_mask_from_polygon(polygon)
    contours, _ = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    contour = contours[0][:, 0, :]
    mask = mask > 0
    lir = lir_contour(mask, contour)
    lir = lir.astype(np.int32)
    lir[0:2] = lir[0:2] + origin
    return lir


def create_mask_from_polygon(polygon):
    assert polygon.shape[0] == 1
    assert polygon.shape[1] > 2
    assert polygon.shape[2] == 2
    bbox = cv.boundingRect(polygon)
    mask = np.zeros([bbox[3], bbox[2]], dtype=np.uint8)
    zero_centered_x = polygon[:, :, 0] - bbox[0]
    zero_centered_y = polygon[:, :, 1] - bbox[1]
    polygon = np.dstack((zero_centered_x, zero_centered_y))
    cv.fillPoly(mask, polygon, 255)
    origin = bbox[0:2]
    return origin, mask
