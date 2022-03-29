from .lir_basis import largest_interior_rectangle as lir_basis
from .lir_within_contour import largest_interior_rectangle \
    as lir_within_contour


def lir(grid, contour=None):
    """
    Returns the Largest Interior Rectangle of a binary grid.
    :param grid: 2D ndarray containing data with `bool` type.
    :param contour: (optional) 2D ndarray with shape (n, 2) containing
        xy values of a specific contour where the rectangle could start
        (in all directions).
    :return: 1D ndarray with lir specification: x, y, width, height
    :rtype: ndarray
    """
    if contour is None:
        return lir_basis(grid)
    else:
        return lir_within_contour(grid, contour)
