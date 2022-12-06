import numpy as np

from .lir_within_contour import largest_interior_rectangle as lir_contour

cv = None  # as an optional dependency opencv will only be imported if needed


def largest_interior_rectangle(polygon):
    check_for_opencv()
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
    check_for_opencv()
    bbox = cv.boundingRect(polygon)
    mask = np.zeros([bbox[3], bbox[2]], dtype=np.uint8)
    zero_centered_x = polygon[:, :, 0] - bbox[0]
    zero_centered_y = polygon[:, :, 1] - bbox[1]
    polygon = np.dstack((zero_centered_x, zero_centered_y))
    cv.fillPoly(mask, polygon, 255)
    origin = bbox[0:2]
    return origin, mask


def check_for_opencv():
    global cv
    if cv is None:
        try:
            import cv2
            cv = cv2
        except Exception:
            raise ImportError('Missing optional dependency \'opencv-python\' to compute lir based on polygon. Use pip or conda to install it.')
