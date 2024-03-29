import os
import sys
from timeit import Timer

import cv2 as cv
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import lir  # noqa: F401, E402
import lir_within_outline  # noqa: F401, E402

# %%


c1 = cv.imread("mask_200w.png", 0)
c2 = cv.imread("mask_3500w.png", 0)
c3 = cv.imread("mask_5000w.png", 0)

t = Timer(lambda: lir.largest_interior_rectangle(c1))
lir1_1 = t.timeit(number=1)

t = Timer(lambda: lir_within_outline.largest_interior_rectangle(c1))
lir2_1 = t.timeit(number=1)

t = Timer(lambda: lir.largest_interior_rectangle(c2))
lir1_2 = t.timeit(number=1)

t = Timer(lambda: lir_within_outline.largest_interior_rectangle(c2))
lir2_2 = t.timeit(number=1)

t = Timer(lambda: lir.largest_interior_rectangle(c3))
lir1_3 = t.timeit(number=1)

t = Timer(lambda: lir_within_outline.largest_interior_rectangle(c3))
lir2_3 = t.timeit(number=1)


# create data
x = [200 * 68 / 1000000, 3500 * 1189 / 1000000, 5000 * 1700 / 1000000]
y1 = [lir1_1, lir1_2, lir1_3]
y2 = [lir2_1, lir2_2, lir2_3]

# plot lines
plt.plot(x, y1, label="lir")
plt.plot(x, y2, label="lir_within_outline")
plt.legend()
plt.xlabel("Megapixel")
plt.ylabel("time (s)")
plt.yscale("log")
plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: "{:g}".format(y)))
plt.savefig("performance_comparison.svg")
