**************************************************
* Construction of the Exceptional Set for BST PWI*
*                        By                      *
*                   Thomas Lynn                  *
*            Northwestern University             *
**************************************************

Program name:	coverage
Compiled from:	coverage.cu [Language: CUDA C]

Purpose of the program:
    The program seeds points in a 2D space and rotates them according to the BST
    PWI protocol specified in the input file. As points are rotated, a variety 
    of measurements can be made although the program currently only supports 
    output of one measurement at a time. Since each point is independent, the
    system is embarassingly parallelizable and the use of GPU technology
    provides significant speed increases over CPUs.

Prerequisites:
    NVIDIA graphics card with CUDA
    CUDA Library
    NVCC compiler

Compiling:
    Compile with NVCC, no special libraries or links outside of CUDA are
    required.

Note on the BST PWI:
    The BST PWI acts on a unit hemispherical shell where y < 0. The BST PWI is a 
    series of rotations about two horizontal axes (in the xz-plane), executed in
    the code as:
		(1) A rotation about the z-axis by angle alpha
		(2) A rotation about y by -gamma to put the second axis aligned with the
		z-axis
		(3) A rotation about the z-axis by angle beta
		(4) A rotation about y by gamma to put the first axis aligned with the
		z-axis
    The first axis is always aligned with the z direction.



Input file structure:
A sample input file has been included.
"[Number of Iterations] "int
    Indicates 20,000 iterations of the PWI will be executed.
"[Alpha] "dbl
    Angle alpha in degrees. Alpha is the rotation angle about the first axis in 
    the counter-clockwise direction.
"[Beta] "dbl
    Angle beta in degrees. Beta is the rotation angle about the second axis in 
    the counter-clockwise direction.
"[Gamma] "dbl
    Angle gamma in degrees. Gamma is the angular separation in the xz-plane 
    between the first and second axes.
"[Output Resolution] "int
    The output resolution in pixels. The program currently only supports output 
    of square domains so this resolution is the number of pixels in each 
    direction.
"[Projection: 0 - Orthographic, 1 - Stereographic, 2 - Lambert EA,
3 - Gnomonic] "int
    The program supports various projections from the seeded 2D plane to the 
    hemisphere. The available projections are:
    	(0) Orthographic: This is a straight on projection from -y. Points on 
    	the hemisphere have their y coordinate ignored and are projected onto
    	the xz-plane. This is the perspective from y = -infinty.
    	(1) Stereographic: This is a projection to the xz-plane from the 
    	persepctive of (x,y,z) = (0,1,0). This conformal projection preserves 
    	circles on the hemisphere but distorts area radially (area increases 
    	radially).
    	(2) Lambert azimuthal equal-area: This projection preserves area (the 
    	program scales the area by a factor of sqrt(2)). This projection should 
    	be used if area is being measured or the entire hemisphere should be 
    	shown without distorting area. This projection distorts more radially.
    	(3) Gnomonic: Neither area-preserving nor conformal, this projection 
    	cannot show the entire hemisphere but has the property that great circle
    	arcs become straight lines. Since all cutting lines in the PWI are great
    	circle arcs, this can be a nice feature.
"[Spread] "dbl
	The spread specifies the half-width of the domain to be seeded.
"[x Center] "dbl
	The x center is the x coordinate of the center of the domain to be seeded.
"[z Center] "dbl
	The z center is the z coordinate of the center of the domain to be seeded.
	The philosophy for specifying the center and spread of the domain is to
	mimic that of traditional fractal software and allow the user to specify a
	point and change the spread to zoom in on the given point.
"[Line Thickness] "dbl
	The line thickness is how epsilon is specified for some measurements on the
	exceptional set. The line thickness is the pixel width given to the cutting
	lines of the PWI, not the actual width. The actual width can be calculated
	using this value and the spread and resolution values.
	epsilon = (line_thickness / resolution) * (2 * spread)
"[Output Data: 0 - Initial cuts, 1 - Stacked cuts, 2 - First return iter,
3 - Boundary location of initial cut, 4 - Stacked distance to boundary,
5 - Stacked location, 6 - Final location] "int
	The program can output various metrics on the exceptional set as it is
	calculated:
		(0) The program records the iteration number at which a seed point first
		passes within epsilon of a the equator y = 0 (which generates cutting
		lines). Depending on which rotation this occurs on, the iteration number
		is stored in one of two arrays such that the first encounter with each
		cutting line is stored separately.
		(1) The program counts the number of encounters (passing withini epsilon
		of the equator) with each cutting line and stores them in two arrays,
		one for each cutting line.
		(2) The program records the iteration number at which the seeded point
		passes within epsilon of its initial position in the first array. The
		second array currently stores the iteration number at which a point
		returns to its initial position after only one of the two axial 
		rotations, but this currently has no use.
		(3) The program records the angle in the xz-plane at which a point first
		encounters the eqautor. The goal of this measurement is to reveal
		invariant sets along the equator, but this data is not refined.
		(4) The program calculates the distance to the equator after each cut
		and incrememnts the seeded point's indices by the distance. The two
		cuts are again stored in different arrays. This measurment, when divided
		by the number of iterations will provide the average distance from each
		of the cutting lines for points throughout the domain. This can be used
		to show invariant sets and mixing, but it has not been looked into
		extensively.
		(5) The program increments the seeded point's indices by the x and z
		coordinates respectively. When divided by the number of iterations, this
		gives and average x and z coordinate for the seed point.
		(6) The program only records the final x and z coordinates of the seeded
		point. This is used to show mixing of an initial condition since it
		provides a simple mapping between the initial and final positions.



Output file structure:
The output file is binary data of the following structure in order:
Type | Size | Description
-----+------+-------------------------------------------------------------------
 int |    1 | N, the number of seed points created, equal to the resolution
     |      | 	squared
 int |    1 | T, the number of iteration
 int |    1 | res, the pixel resolution in the x and z directions
 dbl |    1 | beta, the angle of rotation about axis 2 in radians
 dbl |    1 | alpha, the angle of rotation about axis 1 in radians
 dbl |    1 | gamma, the angle between axis 1 and axis 2 in radians
 dbl |    1 | x center, x coord of center of seeded points 
 dbl |    1 | z center, z coord of center of seeded points
 dbl |    1 | spread, half-width of seeded points
 int |    1 | output data form, see available outputs above
 dbl |    N | data 1, one of two arrays storing output data (often after 1st
     |      | 	cut)
 dbl |    N | data 2, one of two arrays storing output data (often after 2nd
     |      | 	cut)
 dbl |  res | xspace, x axis values for seed grid
 dbl |  res | zspace, z axis values for seed grid
--------------------------------------------------------------------------------
 Total file size: 50 + 2*res*(res + 1) bytes
 For a 1,000x1,000 grid, this is roughly 2MB.
 
Output post processing:
Output has been processed in MATLAB to generate plots and add color.
