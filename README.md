# lir
Largest Interior Rectangle implementation in Python. 

:rocket: Through Numba it is compiled to machine code “just-in-time” for execution at native machine code speed! 

### Acknowledgements

Thanks to [Tim Swan](https://www.linkedin.com/in/tim-swan-14b1b/) for making his Largest Interior Rectangle implementation in C# [open source](https://github.com/Evryway/lir) and did a great [blog post](https://www.evryway.com/largest-interior/) about it. The first part was mainly reimplementing his solution in Python.

The used Algorithm was described 2019 in [Algorithm for finding the largest inscribed rectangle in polygon](https://journals.ut.ac.ir/article_71280_2a21de484e568a9e396458a5930ca06a.pdf) by [Zahraa Marzeh, Maryam Tahmasbi and Narges Mireh](https://journals.ut.ac.ir/article_71280.html).

Thanks also to [Mark Setchell](https://stackoverflow.com/users/2836621/mark-setchell) and [joni](https://stackoverflow.com/users/4745529/joni) who greatly helped optimizing the performance using cpython/numba in [this SO querstion](https://stackoverflow.com/questions/69854335/optimize-the-calculation-of-horizontal-and-vertical-adjacency-using-numpy)

### How it works

For a cell grid:

<img width="200" src="https://github.com/lukasalexanderweber/lir/blob/readme/readme_imgs/cells.png">

We can specify for each cell how far one can go to the right and to the bottom:

Horizontal Adjacency             |  Vertical Adjacency
:-------------------------:|:-------------------------:
<img width="300" src="https://github.com/lukasalexanderweber/lir/blob/readme/readme_imgs/h_adjacency.png" /> |  <img width="300" src="https://github.com/lukasalexanderweber/lir/blob/readme/readme_imgs/v_adjacency.png" />

Now the goal is to find the possible rectangles for each cell. For that, we can specify a Horizontal Vector based on the Horizontal Adjacency and Vertical Vector based on the Vertical Adjacency:

Horizontal Vector (2,2)             |  Vertical Vector (2,2)
:-------------------------:|:-------------------------:
<img width="300" src="https://github.com/lukasalexanderweber/lir/blob/readme/readme_imgs/h_vector.png" /> |  <img width="300" src="https://github.com/lukasalexanderweber/lir/blob/readme/readme_imgs/v_vector.png" />

So at the given cell the Horizontal Vector is (5,4) and the Vertical Vector is (7,4).

Reversing either vector lets you create the spans by zipping the vectors, so for example reversing the Vertical Vector to (4,7) gives a set of spans of (5 by 4) and (4 by 7).

Since `4*7=28 > 5*4=20` a rectangle with width 4 and height 7 is the biggest possible rectangle for cell (2,2).
The width and height is stored in a span map, where the widths and heights of the maximum rectangles are stored for all cells.
Using the area we can identify the biggest rectangle at (2, 2) with width 4 and height 7. 

Widths             |  Heights             |  Areas
:-------------------------:|:-------------------------:|:-------------------------:
<img width="300" src="https://github.com/lukasalexanderweber/lir/blob/readme/readme_imgs/span_map_widths.png" /> |  <img width="300" src="https://github.com/lukasalexanderweber/lir/blob/readme/readme_imgs/span_map_heights.png" /> |  <img width="300" src="https://github.com/lukasalexanderweber/lir/blob/readme/readme_imgs/span_map_areas.png" />

### LIR based on outline

Especially for bigger grids the functionality can be further optimized by only analysing the outline. For a medium mask, resulting from an [image stitching process](https://github.com/lukasalexanderweber/opencv_stitching_tutorial/blob/master/Stitching%20Tutorial.ipynb) with 839 x 285 = ~240.000 cells identifying the LIR takes 1.6s (without compilation time) on my system. Using the outline approach this time is cutted by half to 0.8s.

<img width="500" src="https://github.com/lukasalexanderweber/lir/blob/readme/test_data/mask.png" />

Here how it works:

<img width="200" src="https://github.com/lukasalexanderweber/lir/blob/readme/readme_imgs/outline_approach/cells2.png">

<img width="300" src="https://github.com/lukasalexanderweber/lir/blob/readme/readme_imgs/outline_approach/directions.png">
