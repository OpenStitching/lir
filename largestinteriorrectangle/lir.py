from .lir_basis import largest_interior_rectangle as lir_basis
from .lir_within_contour import largest_interior_rectangle as lir_within_contour
from .lir_within_polygon import largest_interior_rectangle as lir_within_polygon


def lir(data, contour=None):
    """
    Computes the Largest Interior Rectangle.
    :param data: Can be
        1. a 2D ndarray with shape (n, m) of type boolean.
        The lir is found within all True cells
        2. a 3D ndarray with shape (1, n, 2) with integer xy coordinates of a
        polygon in which the lir should be found
    :param contour: (optional) 2D ndarray with shape (n, 2) containing xy
    values of a specific contour where the rectangle could start (in all directions).
    Only needed for case 1.
    :return: 1D ndarray with lir specification: x, y, width, height
    :rtype: ndarray
    """
    if len(data.shape) == 3:
        return lir_within_polygon(data)
    if contour is None:
        return lir_basis(data)
    else:
        return lir_within_contour(data, contour)


def pt1(lir):
    """
    Helper function to compute pt1 of OpenCVs rectangle() from a lir
    """
    assert lir.shape == (4,)
    return (lir[0], lir[1])


def pt2(lir):
    """
    Helper function to compute pt2 of OpenCVs rectangle() from a lir
    """
    assert lir.shape == (4,)
    return (lir[0] + lir[2] - 1, lir[1] + lir[3] - 1)
