import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from largestinteriorrectangle import (  # noqa: F401, E402
    lir,
    lir_basis,
    lir_within_contour,
    lir_within_polygon,
    pt1,
    pt2,
)
