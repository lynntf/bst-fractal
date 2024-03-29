# Construction of the Exceptional Set for the BST PWI fractal
*By Thomas Lynn, Northwestern University*
***
Developed in conjunction with the following papers:
* P. P. Park, T. F. Lynn, P. B. Umbanhowar, J. M. Ottino, and R. M. Lueptow.
*Mixing and the fractal geometry of piecewise isometries*. Phys Rev E, 95:042208,
Apr. 2017. doi:[10.1103/PhysRevE.95.042208](https://doi.org/10.1103/PhysRevE.95.042208).
* T. F. Lynn, L. D. Smith, J. M. Ottino, P. B. Umbanhowar, and R. M. Lueptow.
*Cutting and shuffling a hemisphere: Nonorthogonal axes*. Phys Rev E, 99:032204,
Mar. 2019. doi:[10.1103/PhysRevE.99.032204](https://doi.org/10.1103/PhysRevE.99.032204).
* T. F. Lynn, J. M. Ottino, P. B. Umbanhowar, and R. M. Lueptow.
*Identifying invariant ergodic subsets and barriers to mixing by cutting and shuffling: Study in a birotated hemisphere*.
Phys Rev E, 101:012204, Jan. 2020.
doi:[10.1103/PhysRevE.101.012204](https://doi.org/10.1103/PhysRevE.101.012204).

## Purpose of the program:
Fractal generation for the cut-and-shuffle model (piecewise isometry, PWI) of
the half-filled bi-axial spherical tumbler (BST). The program seeds points in a
2D space and rotates them according to the bi-axial spherical tumbler (BST)
piecewise isometry (PWI) protocol specified in the input file. As points are
rotated, a variety of measurements can be made although the program currently
only supports output of one measurement at a time. Since each point is
independent, the system is embarrassingly parallelizable and the use of GPU
technology provides significant speed increases over CPUs.

## Prerequisites:
* NVIDIA graphics card with CUDA
* CUDA Library
* NVCC compiler

## Compiling:
Compile with NVCC, no special libraries or links outside of CUDA are required.
### Example makefile:
```makefile
coverage: coverage.o
    nvcc -o coverage coverage.o
coverage.o: coverage.cu
    nvcc -c coverage.cu
```


## Note on the BST PWI:
The BST PWI acts on a unit hemispherical shell where *y* &lt; 0. The BST PWI is a
series of rotations about two horizontal axes (in the *xz*-plane), executed in
the code as:
1) A rotation about the *z*-axis by angle &alpha;
2) A rotation about *y* by -&gamma; to put the second axis aligned with the
    *z*-axis
3) A rotation about the *z*-axis by angle &beta;
4) A rotation about *y* by &gamma; to put the first axis aligned with the
    *z*-axis

The first axis is always aligned with the *z* direction.


## Basic program structure:
1. Receive inputs from file
2. Pass information to the GPU
3. Create arrays containing seed point locations in the 2D plane
4. Project seed points to the hemisphere (discard points outside of the range
that can be projected). Parameters regarding the domain center, size, and
resolution are used here.
5. During rotation procedure, record information into arrays. Parameters
regarding the rotational protocol and desired information are used here. Do this
at multiple times throughout the process depending on the desired information:
    1. Before any rotation
    2. After the first axis rotation
    3. After the second axis rotation
    4. After all rotations
6. Retrieve data from the GPU and save


## Input file structure:
A sample input file has been included.

1. ```"[Number of Iterations] "int```<br />
Indicates the number of iterations of the PWI will be executed.
2. ```"[Alpha] " dbl```<br />
Angle &alpha; in degrees. &alpha; is the rotation angle about the first axis
in the counter-clockwise direction.
3. ```"[Beta] " dbl```<br />
Angle &beta; in degrees. &beta; is the rotation angle about the second axis in
the counter-clockwise direction.
4. ```"[Gamma] " dbl```<br />
Angle &gamma; in degrees. &gamma; is the angular separation in the *xz*-plane
between the first and second axes. Orthogonal axes use &gamma; = 90&deg;.
5. ```"[Output Resolution] "int```<br />
The output resolution in pixels. The program currently only supports output of
square domains so this resolution is the number of pixels in each direction.
Changing the program for different resolutions in the x and z direction is not
particularly challenging, but is not implemented here.
6. ```"[Projection: 0 - Orthographic, 1 - Stereographic, 2 - Lambert EA, 3 - Gnomonic, 4 - Square Lambert] " int```<br />
The program supports various projections from the seeded 2D plane to the
hemisphere. The available projections are:

    * 0: [*Orthographic*](https://en.wikipedia.org/wiki/Orthographic_projection):
    This is a straight on projection from *-y*. Points on the hemisphere have
    their *y* coordinate ignored and are projected onto the *xz*-plane. This is
    the perspective from *y* = -&infin;.
    * 1: [*Stereographic*](https://en.wikipedia.org/wiki/Stereographic_projection):
    This is a projection to the *xz*-plane from the perspective of (*x*,*y*,*z*) =
    (0,1,0). This conformal projection preserves circles on the hemisphere but
    distorts area radially (area increases radially).
    * 2: [*Lambert azimuthal
    equal-area*](https://en.wikipedia.org/wiki/Lambert_azimuthal_equal-area_projection):
    This projection preserves area (the program scales the area by a factor of
    &radic;2). This projection should be used if area is being measured or the
    entire hemisphere should be shown without distorting area. This projection
    distorts more radially.
    * 3: [*Gnomonic*](https://en.wikipedia.org/wiki/Gnomonic_projection): Neither
    area-preserving nor conformal, this projection cannot show the entire
    hemisphere but has the property that great circle arcs become straight
    lines. Since all cutting lines in the PWI are great circle arcs, this can be
    a nice feature.
    * 4: *Square Lambert*: Applies a transformation from the
    square to the unit circle ([Shirley
    1997](https://doi.org/10.1080/10867651.1997.10487479)) before applying the
    Lambert EA projection. Intended to utilize the entire rectangular grid.
    Mostly used for computing fractional coverage &Phi;.
7. ```"[Spread] " dbl```<br />
The spread specifies the half-width of the domain to be seeded.<br />
```Domain = [x_center - spread, x_center + spread] by [z_center - spread, z_center + spread]```
8. ```"[x Center] " dbl```<br />
The x center is the x coordinate of the center of the domain to be seeded.
9. ```"[z Center] " dbl```<br />
The z center is the z coordinate of the center of the domain to be seeded. The
philosophy for specifying the center and spread of the domain is to mimic that
of traditional fractal software and allow the user to specify a point and change
the spread to zoom in on the given point.
10. ```"[Line Thickness] " dbl```<br />
The line thickness is &epsilon; for some measurements on the exceptional
set. This determines how close points need to be to be considered "cut", or
"returned".
11. ```"[Output Data: 0 - Initial cuts, 1 - Stacked cuts, 2 - First return iter, 3 - Boundary location of initial cut, 4 - Stacked distance to boundary, 5 - Stacked location, 6 - Final location, 8 - Just Coverage, 9 - Minimum distance to cut] " int```<br />
The program can output various metrics (some unlisted because they are not
useful) on the exceptional set as it is calculated:

    * 0: The program records the iteration number at which a seed point first
    passes within &epsilon; of a the equator *y* = 0 (which generates
    cutting lines). Depending on which rotation this occurs on, the iteration
    number is stored in one of two arrays such that the first encounter with
    each cutting line is stored separately.
    * 1: The program counts the number of encounters (passing within
    &epsilon; of the equator) with each cutting line and stores them in two
    arrays, one for each cutting line. This is how cutting line density is
    computed.
    * 2: The program records the iteration number at which the seeded point
    passes within &epsilon; of its initial position in the first array. The
    second array currently stores the iteration number at which a point returns
    to its initial position after only one of the two axial rotations (half
    iteration), but this currently has no meaningful use.
    * 3: The program records the largest *z* value in the *xz*-plane at which a
    point first encounters the equator (post processed to an angular position
    using arcsin(2*z* -1); cutting lines are not sided here). The goal of this
    measurement is to reveal invariant sets along the equator, but this data is
    not refined.
    * 4: The program calculates the distance to the equator after each cut and
    accumulates this distance. The two cuts are again stored in different
    arrays. This measurement, when divided by the number of iterations will
    provide the average distance from each of the cutting lines for points
    throughout the domain. This can  potentially be used to show invariant sets
    and mixing.
    * 5: The program accumulates the *x* and *z* coordinates. When divided by
    the number of iterations, this gives and average x and z coordinate for the
    seed point. This is only done after each full iteration.
    * 6: The program only records the final *x* and *z* coordinates of the
    seeded point. This produces a map between initial and final locations that
    can be used to mix an initial condition.
    * 7: *unlisted* The program records how many times the point has passed
    through the infinitely thin flowing layer (ITFL; periodic equator) after
    each rotation. This reveals almost no information at all since, on average,
    every point passes through the ITFL the same amount of times.
    * 8: Just returns the value of coverage, &Phi;. Does not utilize the two
    register variables and reduces file size. Coverage can be computed from
    other outputs so only use for data savings.
    * 9: Minimum distance to each of the two cutting lines over the course of a
    trajectory. For points in the exceptional set, this tends to zero as more
    iterations are computed. Values are initialized at 10, so points falling
    outside the range of the unit circle will have minimum distance of 10.
    * 10: *unlisted* Records the last iteration that the point returns to the
    cutting line. Does not reveal anything particularly interesting.
    * 11: *unlisted* Counts the number of returns within &epsilon; of the
    initial location. Again, returns for the half iteration are computed, but
    are currently meaningless.
    * 12: *unlisted* Accumulation of distances to the initial point. Does not
    reveal much of anything.

## Output file structure:
The output file is binary data of the following structure in order:

|Type | Size | Description|
|:--        | --:       | :--|
| ```int``` |         1 | *N*, the number of seed points created, equal to the resolution squared|
| ```int``` |         1 | *T*, the number of iteration|
| ```int``` |         1 | ```res```, the pixel resolution in the *x* and *z* directions|
| ```dbl``` |         1 | &beta;, the angle of rotation about axis 2 in radians (note the order of &alpha; and &beta;)|
| ```dbl``` |         1 | &alpha;, the angle of rotation about axis 1 in radians|
| ```dbl``` |         1 | &gamma;, the angle between axis 1 and axis 2 in radians|
| ```dbl``` |         1 | *x* center, *x* coordinate of center of seeded points|
| ```dbl``` |         1 | *z* center, *z* coordinate of center of seeded points|
| ```dbl``` |         1 | ```spread```, half-width of seeded points|
| ```int``` |         1 | output data form (see available outputs above)|
| ```dbl``` |         1 | &epsilon;|
| ```dbl``` |       *N* | data 1, one of two arrays storing output data (often after 1st cut) [not written for output (8)]|
| ```dbl``` |       *N* | data 2, one of two arrays storing output data (often after 2nd cut) [not written for output (8)]|
| ```dbl``` | ```res``` | ```xspace```, *x* axis values for seed grid|
| ```dbl``` | ```res``` | ```zspace```, *z* axis values for seed grid|
| ```int``` |         1 | number of seed points in the estimate of the exceptional set (for computing &Phi;)|
| ```int``` |         1 | number of seed points that fell outside of the hemisphere (for computing &Phi;)|

 Total file size is approximately ```8*(2*res*res + 2*res + 13)``` bytes
 * For a 1,000 &times; 1,000 grid, this is 15.6 MB
 * For a 2,000 &times; 2,000 grid, this is 62.5 MB
 * For a 4,000 &times; 4,000 grid, this is 250.0 MB


## Output post processing:

MATLAB scripts are used to process raw data.

## Support

The development of the *bst-fractal* program was supported by the [US National
Science Foundation](http://www.nsf.gov/) under grant
[CMMI-1435065](https://www.nsf.gov/awardsearch/showAward?AWD_ID=1435065).
