/*
  Bi-axial Spherical Tumbler Peice-wise Isometry
    with Point Tracing and Tunable Axes
    Zooming in on the exceptional set
  
  Program written by Thomas Lynn

  Inputs:  file name of input file
           file name of output file

  Outputs: file containing header data and output image

  Editing History:
    06/14/2016: Initial draft
    06/22/2016: Updated to cartesian, commented code
    07/05/2016: Changed to a 'histogram' image type output (looks pretty)
    07/25/2016: Reduced number of working blocks
                Drastically reduced working memory by writing to array more often
    08/17/2016: Changed to seeding points in current view and marking them when they hit the border.
                This allows deep zooms, but resolution isn't great since you have to rotate
		so many points, and the system still takes a long time to evolve small detail.
    07/06/2017: Several edits have happened. Rotation kernel function has been refined to do the
                proper sequence of rotations (previously was incorrect). Various code cleanups.
    07/07/2017: Add in a Danckwerts feature so that the two codes don't need to be separate. This
                entails changing markers to doubles and adding a new switch.

  Notes:
    [ ] Add in a randomization element to the angles rotated (should be fine to generate on the host)
    [ ] Make sure you are making back-up copies of code

 */

// Program requires access to math.h and cuda.h
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda.h>
#include <curand.h>

#ifndef M_PI
#define M_PI 3.1415926535897932384626433832795
#endif

__device__ static int    dev_N;
__device__ static int    dev_T;
__device__ static int    dev_res;
__device__ static int    dev_histogram;
//__device__ static int    dev_mixing;
//__device__ static int    dev_antialias;
__device__ static double dev_xcenter;
__device__ static double dev_zcenter;
__device__ static double dev_thresh;
__device__ static double dev_spread;
__device__ static double dev_Sx;
__device__ static double dev_Cx;
__device__ static double dev_Sz;
__device__ static double dev_Cz;
__device__ static double dev_Saxis;
__device__ static double dev_Caxis;
//__device__ static int dev_phi;
__device__ static int    dev_half_iter;

/*
void rotate(int t, double* x, double* y, double* z, double theta, int tnext)

This function rotates points for the biaxial spherical tumbler PWI in reverse order.

Inputs:
        int     t            The iteration steps to take.
	double* x            The array where x values are stored.
	double* z            The array where z values are stored.
	int*    marker1      Storage for color 1.
	int*    marker2      Storage for color 2.
	int     doprojection Label for the type of projection to do.

Outputs:
        Alters *marker1      The output array stores the information in marker1 at the x and z indices.
        Alters *marker2      The output array stores the information in marker2 at the x and z indices.
 */

__global__ void rotate(int t, double* x, double* z, double* marker1, double* marker2, int doprojection, int* phi, int* used) {
  // Get the global index
  int gi = threadIdx.x + blockIdx.x * blockDim.x;

  // Protect uninitialized regions of the data
  if (gi < dev_res*dev_res) {
    
    // Store local copies of z and y
    double xl = x[gi/dev_res];
    double zl = z[gi%dev_res];
    int done1 = 0;
    int done2 = 0;
    int flip = 0;

    double marker1_l = 0;
    double marker2_l = 0;


    //////////////////////////////////
    // Anti aliasing
    //////////////////////////////////
    // double samples = 1;
    // double marker1_a = 0;
    // double marker2_a = 0;
    // double xl_init = xl;
    // double zl_init = zl;

    // if (antialias == 1) {
    //   samples =4;
    // }

    //////////////////////////////////
    // Loop over anti aliasing samples
    //////////////////////////////////
    // int jj = 0;
    // for (jj = 0; (jj < samples); jj++){

    // if(antialias == 1) {
    // 	xl = xl_init + ( 1.0*(jj%2) - 0.5  ) * dev_spread/(dev_res * 2.0);
    // 	zl = zl_init + ( 1.0*(jj/2) - 0.5  ) * dev_spread/(dev_res * 2.0);
    // }

    // Remove points that are outside the domain
    int outside = 0;
    if (xl*xl + zl*zl >1) {
      done1 = 1;
      done2 = 1;
      outside = 1;
      if (dev_histogram == 8) { // Count the number of grid points inside
	      atomicAdd(used,1);
      }
    }

    ///////////////////////////////////////////////
    // Project from the flat grid to the hemisphere
    ///////////////////////////////////////////////
    double temp = xl;
    if (doprojection == 1){
      // Stereographic projection
      xl = 2*xl/(1+ xl*xl+ zl*zl);
      zl = 2*zl/(1+ temp*temp+ zl*zl);
    } else if (doprojection == 2){
      // Lambert equal-area
      xl = sqrt(1.0 - (xl*xl + zl*zl) /2.0 )*xl*sqrt(2.0);
      zl = sqrt(1.0 - (temp*temp + zl*zl) /2.0 )*zl*sqrt(2.0);
    } else if (doprojection == 3){
      // Gnomonic
      xl = xl*5;
      zl = zl*5;
      double rho = sqrt(xl*xl + zl*zl);
      double c = atan(rho);
      double phi = asin(cos(c));//asin(zl*sin(c)/rho);
      double lamb = atan2(xl*sin(c),(-zl*sin(c)));//atan2(xl*sin(c),(rho*cos(c)));
      xl = cos(phi)*cos(lamb);
      zl = cos(phi)*sin(lamb);
    }
    
    // Calculate the initial y-value
    double yl = -sqrt(abs(1 - xl*xl - zl*zl));

    // Initial positions
    double xl0 = xl;
    double zl0 = zl;
    double yl0 = yl;

    // Initialize two counters
    int count1 = 0;
    int count2 = 0;

    /////////////////////////////////////////////////////////////
    // What information to output (initial values for some cases)
    /////////////////////////////////////////////////////////////
    switch (dev_histogram) {
      case 3 : // 
        if (done1==0) {
          marker1_l = 10;
          marker2_l = 10;
        }
        break;
      case 1 : // Add one to count of times near the boundary
        if (abs(yl) < dev_thresh & !outside) {
          marker2_l++;
        }
        break;
      
      case 0 : // Mark points near the boundary as done
        if (abs(yl) < dev_thresh & !done2) {
          marker2_l = 1;
          done2 = 1;
        }
        break;
      case 9 : // min dist to a cut
        marker2_l = 10;
        marker1_l = 10;
    }



    ////////////////////////
    // Do several iterations
    ////////////////////////
    int k = 0;
    // k goes from 0 to t-1 as long as one color isn't done
    for(k = 0; (k < t) & (!done1 | !done2); k++) {


      temp = xl;
      // Rotate the axes back
      xl =   xl * dev_Caxis  -  zl * dev_Saxis;
      zl = temp * dev_Saxis  +  zl * dev_Caxis;
      

      if (k > 0 || dev_half_iter == 0) {  // Skip this rotation for k==0 when a half-iteration is specified (otherwise execute)
        // Store an extra copy of y so that it is not overwritten during computation
        temp = yl;
        
        // Do the backwards rotation about x
        yl =   yl * dev_Cx  -  xl * dev_Sx;
        xl = temp * dev_Sx  +  xl * dev_Cx;
        
        // Flip the point over if it went past the edge (periodic boundary)
        if (yl*temp < 0) {
          yl = -yl;
          xl = -xl;
          flip = 1;
        }


        /////////////////////////////
        // What information to output
        /////////////////////////////
        switch (dev_histogram) {
          case 8 : // Just coverage
            if (abs(yl) < dev_thresh & !done1) {
              // int* temp = &dev_phi;
              atomicAdd(phi, 1);
              done1 = 1;
              done2 = 1;
            }
            break;

          case 7: // Number of flips
            marker1_l += flip;
            flip = 0;
            break;

          // case 5 : // Stacked x and z locations
          //   marker2_l += zl;
          //   marker1_l += xl;
          //   break;

          case 4: // Distance to boundary
            marker1_l+= asin(abs(yl));
            break;

          case 3 : // Anglular position of interaction with boundary
            if (abs(yl) < dev_thresh & !done1) { // Check if near the discontinuity
              temp = 0.5*(zl/(sqrt(xl*xl + zl*zl)) + 1); // This is an angular position? Doesn't look like it. If there was an acos or atan2, then it would maybe make sense? Must need post-processing
              marker1_l = (temp > marker1_l) ? marker1_l : temp;
              // count1 += 1;
              // done1 = 1;
            }
            break;

          case 2 : // Iteration where point returns to initial position
            if (sqrt((xl-xl0)*(xl-xl0) + (zl-zl0)*(zl-zl0) + (yl-yl0)*(yl-yl0)) < dev_thresh & !done1) { // If Euclidean distance is less than the threshold value, consider returned
              marker1_l = k+1;
              done1 = 1;
            }
            break;

          case 1 : // Number of times passing close to boundary
            if (abs(yl) < dev_thresh) {
              marker1_l++;
            }
            break;
            
          case 0 : // First iteration passing close to boundary
            if (abs(yl) < dev_thresh & !done1) {
              marker1_l = k+1;
              done1 = 1;
              done2 = 1;
            }
            break;
            
          case 10 : // Last iteration passing close to the boundary
            if (abs(yl) < dev_thresh & !done1) {
              marker1_l = k+1;
            }
            break;
          case 9 : // min dist to cutting line
            marker1_l = (asin(abs(yl)) < marker1_l) ? asin(abs(yl)) : marker1_l;
            break;
          case 11 : // Count of returns to initial position
            // if (sqrt((xl-xl0)*(xl-xl0) + (zl-zl0)*(zl-zl0) + (yl-yl0)*(yl-yl0)) < dev_thresh & !done1) { // If Euclidean distance is less than the threshold value, consider returned
            //   marker1_l++;
            // }
            marker1_l = marker1_l + sqrt((xl-xl0)*(xl-xl0) + (zl-zl0)*(zl-zl0) + (yl-yl0)*(yl-yl0));
            break;
        }
      }


      // Rotate the axes forward
      temp = xl;
      xl =   xl * dev_Caxis  +  zl * dev_Saxis;
      zl =-temp * dev_Saxis  +  zl * dev_Caxis;
      
      // Do the backwards rotation about z
      temp = yl;
      yl =   yl * dev_Cz  -  xl * dev_Sz;
      xl = temp * dev_Sz  +  xl * dev_Cz;
	
      // Flip the point over if it went past the edge (periodic boundary)
      if (yl*temp < 0) {
        yl = -yl;
        xl = -xl;
        flip = 1;
      }

      /////////////////////////////
      // What information to output
      /////////////////////////////
      switch (dev_histogram) {
        case 8 : // Just coverage
          if (abs(yl) < dev_thresh & !done1) {
            // int* temp = &dev_phi;
            atomicAdd(phi, 1);
            done1 = 1;
          }
          break;

        case 7: // Number of flips
          marker2_l += flip;
          flip = 0;
          break;

        case 5 : // Stacked x and z locations
          marker2_l += zl;
          marker1_l += xl;
          break;

        case 4 : // Distance to boundary
          marker2_l+= asin(abs(yl));
          break;

        case 3 : // Anglular position of interaction with boundary
          if (abs(yl) < dev_thresh & !done2) {
            temp = 0.5*(zl/(sqrt(xl*xl + zl*zl)) + 1);
            marker2_l = (temp > marker2_l) ? marker2_l : temp;
            // marker2_l += 0.5*(sin(atan2(zl,xl)) + 1);
            // count2 += 1;
            // done2 = 1;
          }
          break;

        case 2 : // Iteration where point returns to initial position
          if (sqrt((xl-xl0)*(xl-xl0) + (zl-zl0)*(zl-zl0) + (yl-yl0)*(yl-yl0)) < dev_thresh & !done2) {
            marker2_l = k+1;
            done2 = 1;
            //done1 = 1;
          }
          break;

        case 1 : // Number of times passing close to the boundary
          if (abs(yl) < dev_thresh) {
            marker2_l++;
          }
          break;
            
        case 0 : // First iteration passing close to boundary
          if (abs(yl) < dev_thresh & !done2) {
            marker2_l = k+1;
            done2 = 1;
          }
          break;

        case 10 : // Last iteration passing close to the boundary
          if (abs(yl) < dev_thresh & !done1) {
            marker2_l = k+1;
          }
          break;
        case 9 : // min dist to cutting line
          marker2_l = (asin(abs(yl)) < marker2_l) ? asin(abs(yl)) : marker2_l;
          break;
        case 11 : // Count of returns to initial position
            // if (sqrt((xl-xl0)*(xl-xl0) + (zl-zl0)*(zl-zl0) + (yl-yl0)*(yl-yl0)) < dev_thresh & !done1) { // If Euclidean distance is less than the threshold value, consider returned
            //   marker2_l++;
            // }
            marker2_l = marker2_l + sqrt((xl-xl0)*(xl-xl0) + (zl-zl0)*(zl-zl0) + (yl-yl0)*(yl-yl0));
            break;
      }
    }
    //  }

    // if (dev_histogram == 3) {
    //   if (count1 != 0) {marker1_l = marker1_l/((double)(count1));}
    //   if (count2 != 0) {marker2_l = marker2_l/((double)(count2));}
    // }

    //////////////////////////////////
    // Store data in the shared memory
    //////////////////////////////////
    switch (dev_histogram) {
      case 6 : // Final location
        if (!outside) {
          marker1[gi] = xl;
          marker2[gi] = zl;
        } else {
          marker1[gi] = dev_xcenter - dev_spread;
          marker2[gi] = dev_zcenter - dev_spread;
        }
        break;

      default : // Store marker values in shared memory
        marker1[gi] = marker1_l;
        marker2[gi] = marker2_l;
    }
  }
}


/*
void init_data(int t, double*x, double* y, double* z)

This function puts the initial conditions into the global memory blocks being used.
Points are seeded around the equator/boundary of the hemisphere.

Inputs:
        int t                  Indicates whether or not it is the first or second half iteration.
        double* x, y, z        Arrays for initial conditions storage

Outputs:
	Alters *x, *y, *z      Initial conditions are stored.
 */

__global__ void init_new(double* x, double* z) {
  // Get the global index
  int gi = threadIdx.x + blockIdx.x * blockDim.x;

  if (gi < dev_res){
    int xi = gi;
    int zi = gi;
    // Starting at the left edge, add half a pixel and then the rest to get into position
    double xl = dev_xcenter - dev_spread + xi*(2*dev_spread/dev_res) + (dev_spread/dev_res);
    double zl = dev_zcenter - dev_spread + zi*(2*dev_spread/dev_res) + (dev_spread/dev_res);
   
    // Set the initial conditions
    x[gi] = xl;
    z[gi] = zl;
  }
}



/*
int main(int argc, char* argv[])

The main program executes a point rotation over a hemisphere with periodic boundary conditions.

Inputs:
       input filename   The input file for heat which contains all of the input data
       output filename  The output file containing N, T, x, y, and z

Outputs:
       int              The main program returns 0 if sucessful
       output file      The output file contains int N, int T, and double *x, *y, *z.

Console Outputs:
       N  =  ... points
       T  =  ... iterations
       thetax =  ... degrees
       thetaz =  ... degrees
       Execution time = ... seconds
       
 */


int main(int argc, char* argv[])
{
  ////////////////////////////////////////////////////////////////////////////
  // Setup
  ////////////////////////////////////////////////////////////////////////////
  // Begin timing the computation
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start,0);

  // Choose the GPU card (on ESAM Kepler or Tesla)
  cudaDeviceProp prop;
  int dev;
  memset(&prop, 0, sizeof(cudaDeviceProp));
  prop.multiProcessorCount = 13;
  cudaChooseDevice(&dev, &prop);
  cudaSetDevice(dev);

  // Get the maximum thread count for the device
  cudaGetDeviceProperties(&prop, dev);
  int num_threads = prop.maxThreadsPerBlock;

  ////////////////////////////////////////////////////////////////////////////
  // Input
  ////////////////////////////////////////////////////////////////////////////
  // Open the input file on the host
  // Check if the inputs to the program are correct
  if (argc > 3 ) {
    printf ("Incorrect usage: only enter the input data file name and output data file name\n") ;
    return 0;
  }
  
  // Open the input file for reading
  FILE* inputfile = fopen(argv[1],"r");
  // Return an error if the input file couldn't be opened
  if (!inputfile) {
    printf("Unable to open input file\n");
    return 0;
  }
  // N points, T iterations, Do a projection or not, Do a halfiteration or not
  int N, T, res, projection, histogram, half_iter;
  // Initalize thetax, thetaz, thetaaxis, deltatheta doubles
  double thetax, thetaz, thetaaxis, spread, xcenter, zcenter, line;
  spread = 0;
  line = 1;
  // Read in N, T, thetax, thetaz from the input file
  fscanf(inputfile,
   "[Number of Iterations] %d"
   " [Alpha] %lf"
   " [Beta] %lf"
   " [Gamma] %lf"
   " [Output Resolution] %d"
   " [Projection: 0 - Orthographic, 1 - Stereographic, 2 - Lambert EA, 3 - Gnomonic, 4 - Square Lambert] %d"
   " [Spread] %lf"
   " [x Center] %lf"
   " [z Center] %lf"
   " [Line Thickness] %lf"
   " [Output Data: 0 - Initial cuts, 1 - Stacked cuts, 2 - First return iter, 3 - Boundary location of initial cut, 4 - Stacked distance to boundary, 5 - Stacked location, 6 - Final location, 8 - Just Coverage, 9 - Minimum distance to cut] %d"
   " [Half iter] %d",
   &T, &thetaz, &thetax, &thetaaxis, &res, &projection, &spread, &xcenter,
   &zcenter, &line, &histogram, &half_iter);

  
  // fscanf(inputfile,"[Number of Iterations] %d", &T);
    printf("[Number of Iterations] %d \n", T);
  // fscanf(inputfile,"[Alpha] %lf", &thetaz);
    printf("[Alpha] %lf  \n", thetaz);
  // fscanf(inputfile,"[Beta] %lf", &thetax);
    printf("[Beta] %g  \n", thetax);
  // fscanf(inputfile,"[Gamma] %lf", &thetaaxis);
    printf("[Gamma] %g  \n", thetaaxis);
  // fscanf(inputfile,"[Output Resolution] %d", &res);
    printf("[Output Resolution] %d  \n", res);
  // fscanf(inputfile,"[Projection: 0 - Orthographic, 1 - Stereographic, 2 - Lambert EA, 3 - Gnomonic, 4 - Square Lambert] %d", &projection);
    printf("[Projection: 0 - Orthographic, 1 - Stereographic, 2 - Lambert EA, 3 - Gnomonic, 4 - Square Lambert] %d  \n", projection);
  // fscanf(inputfile,"[Spread] %lf", &spread);
    printf("[Spread] %g \n", spread);
  // fscanf(inputfile,"[x Center] %lf", &xcenter);
    printf("[x Center] %g \n", xcenter);
  // fscanf(inputfile,"[z Center] %lf", &zcenter);
    printf("[z Center] %lf \n", zcenter);
  // fscanf(inputfile,"[Line Thickness] %lf", &line);
    printf("[Line Thickness] %lf \n", line);
  // fscanf(inputfile,"[Output Data: 0 - Initial cuts, 1 - Stacked cuts, 2 - First return iter, 3 - Boundary location of initial cut, 4 - Stacked distance to boundary, 5 - Stacked location, 6 - Final location, 8 - Just Coverage, 9 - minimum distance to cut] %d", &histogram);
    printf("[Output Data: 0 - Initial cuts, 1 - Stacked cuts, 2 - First return iter, 3 - Boundary location of initial cut, 4 - Stacked distance to boundary, 5 - Stacked location, 6 - Final location, 8 - Just Coverage, 9 - minimum distance to cut] %d \n", histogram);
  // fscanf(inputfile,"[Half iter] %d", &half_iter);


  thetax = thetax/180*M_PI;        // Convert to radians
  thetaz = thetaz/180*M_PI;        // Convert to radians
  thetaaxis = thetaaxis/180*M_PI;  // Convert to radians

  // printf("%f \n",spread);

  // Close the input file
  fclose(inputfile);


  // int res_start = 1;
  // int Nres = 200;
  // int res_arr[Nres], outside_arr[Nres], phi_arr[Nres];
  // int k = 0;
  // for (k = 0; (k < Nres); ++k) {
  //   res = k+res_start;
  //   res_arr[k] = res;


  N = res*res;
  //double thresh = line*spread / res;
  double thresh = line;
  // For a full Lambert equal area azimuthal projection, the line threshold should
  // be (5/4)*sqrt(2) to accomodate for the distortion at the border and allow for
  // continuous lines

  // Calculate the required number of blocks for initialization
  int num_blocks = (res)/num_threads + ((res)%num_threads ? 1 : 0);
  // Number of blocks to rotate all pixels
  int num_bigblocks = (N)/num_threads + ((N)%num_threads ? 1 : 0);

  // Reduce thread calls if possible
  if (N < num_threads) {
    num_threads = N;
  }

  double Sx = sin(thetax);
  double Cx = cos(thetax);
  double Sz = sin(thetaz);
  double Cz = cos(thetaz);
  double Saxis = sin(thetaaxis);
  double Caxis = cos(thetaaxis);
  int phi = 0;
  int outside = 0;
  //int half_iter = 1;

  //////////////////////////////////////////////////////////////////////////////
  // Memory
  //////////////////////////////////////////////////////////////////////////////
  // Copy constants into constant memory
  cudaMemcpyToSymbol(dev_N,         &N,         sizeof(int));
  cudaMemcpyToSymbol(dev_T,         &T,         sizeof(int));
  cudaMemcpyToSymbol(dev_res,       &res,       sizeof(int));
  cudaMemcpyToSymbol(dev_histogram, &histogram, sizeof(int));
  cudaMemcpyToSymbol(dev_half_iter, &half_iter, sizeof(int));
  cudaMemcpyToSymbol(dev_xcenter,   &xcenter,   sizeof(double));
  cudaMemcpyToSymbol(dev_zcenter,   &zcenter,   sizeof(double));
  cudaMemcpyToSymbol(dev_thresh,    &thresh,    sizeof(double));
  cudaMemcpyToSymbol(dev_spread,    &spread,    sizeof(double));
  cudaMemcpyToSymbol(dev_Sx,        &Sx,        sizeof(double));
  cudaMemcpyToSymbol(dev_Cx,        &Cx,        sizeof(double));
  cudaMemcpyToSymbol(dev_Sz,        &Sz,        sizeof(double));
  cudaMemcpyToSymbol(dev_Cz,        &Cz,        sizeof(double));
  cudaMemcpyToSymbol(dev_Saxis,     &Saxis,     sizeof(double));
  cudaMemcpyToSymbol(dev_Caxis,     &Caxis,     sizeof(double));

  // Allocate memory on the host
  double* x = (double*)malloc(res * sizeof(double));
  double* z = (double*)malloc(res * sizeof(double));
  double* marker1 = (double*)malloc(res * res * sizeof(double));         // One color
  double* marker2 = (double*)malloc(res * res * sizeof(double));         // One color

  // Allocate memory on the device
  double* dev_x;
  double* dev_z;
  double* dev_marker1;
  double* dev_marker2;
  int* dev_phi;
  int* dev_outside;

  // Make sure to check that allocation has no errors
  int err = cudaMalloc((void**)&dev_x,       res*sizeof(double));
  err =     cudaMalloc((void**)&dev_z,       res*sizeof(double));
  err =     cudaMalloc((void**)&dev_marker1, res*res*sizeof(double));
  err =     cudaMalloc((void**)&dev_marker2, res*res*sizeof(double));
  err =     cudaMalloc(&dev_phi, sizeof(int));
  err =     cudaMalloc(&dev_outside, sizeof(int));
  if (err != cudaSuccess) {
    printf("%s\n", cudaGetErrorString((cudaError_t)err));
    exit(1);
  }

  // Initialize to zero to clear out junk data
  cudaMemset(dev_x, 0, res*sizeof(double));
  cudaMemset(dev_z, 0, res*sizeof(double));
  cudaMemset(dev_marker1, 0, res*res*sizeof(double));
  cudaMemset(dev_marker2, 0, res*res*sizeof(double));
  cudaMemset(dev_phi,    0, sizeof(int));
  cudaMemset(dev_outside,0, sizeof(int));


  //////////////////////////////////////////////////////////////////////////////
  // Initial Conditions (x and z grid values)
  //////////////////////////////////////////////////////////////////////////////
  init_new<<<num_blocks,num_threads>>>(dev_x,dev_z);
  cudaError_t err2 = cudaGetLastError();
  if (err2 != cudaSuccess) 
    printf("Error: %s\n", cudaGetErrorString(err2));

  //////////////////////////////////////////////////////////////////////////////
  // Rotation
  //////////////////////////////////////////////////////////////////////////////
  rotate<<<num_bigblocks,num_threads>>>(T,dev_x,dev_z, dev_marker1, dev_marker2, projection, dev_phi, dev_outside);
  err2 = cudaGetLastError();
  if (err2 != cudaSuccess) 
    printf("Error: %s\n", cudaGetErrorString(err2));


  //////////////////////////////////////////////////////////////////////////////
  // Collect data on host and clean up
  //////////////////////////////////////////////////////////////////////////////
  // Copy memory from the device to the host
  cudaMemcpy(x,       dev_x,       res*sizeof(double),     cudaMemcpyDeviceToHost);
  cudaMemcpy(z,       dev_z,       res*sizeof(double),     cudaMemcpyDeviceToHost);
  cudaMemcpy(marker1, dev_marker1, res*res*sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(marker2, dev_marker2, res*res*sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(&phi,    dev_phi,     sizeof(int),            cudaMemcpyDeviceToHost);
  cudaMemcpy(&outside,dev_outside, sizeof(int),            cudaMemcpyDeviceToHost);

  cudaMemset(dev_marker1,0,res*res*sizeof(double));
  cudaMemset(dev_marker2,0,res*res*sizeof(double));

  // Open the output file on the host
  FILE* outputfile = fopen(argv[2],"w");

  // Write information to the outputfile
  fwrite(&N,         sizeof(int),    1,       outputfile);
  fwrite(&T,         sizeof(int),    1,       outputfile);
  fwrite(&res,       sizeof(int),    1,       outputfile);
  fwrite(&thetax,    sizeof(double), 1,       outputfile);
  fwrite(&thetaz,    sizeof(double), 1,       outputfile);
  fwrite(&thetaaxis, sizeof(double), 1,       outputfile);
  fwrite(&xcenter,   sizeof(double), 1,       outputfile);
  fwrite(&zcenter,   sizeof(double), 1,       outputfile);
  fwrite(&spread,    sizeof(double), 1,       outputfile);
  fwrite(&histogram, sizeof(int),    1,       outputfile);
  fwrite(&thresh,    sizeof(double), 1,       outputfile);
  if (histogram != 8) { // Histogram == 8 implies we only want the value of the coverage, not the actual output
    fwrite(marker1,    sizeof(double), res*res, outputfile);
    fwrite(marker2,    sizeof(double), res*res, outputfile);
    fwrite(x,          sizeof(double), res,     outputfile);
    fwrite(z,          sizeof(double), res,     outputfile);
  }
  fwrite(&phi,       sizeof(int),    1,       outputfile);
  fwrite(&outside,   sizeof(int),    1,       outputfile);

  // Close the output file
  fclose(outputfile);
  
  // Print to console information relevant to the problem
  printf("N =   %d points\n",N);
  printf("T =   %d iterations\n",T);
  printf("num_threads =  %d\n", num_threads);
  printf("num_blocks  =  %d\n", num_blocks);
  printf("thetax =  %g degrees\n",thetax/M_PI*180);
  printf("thetaz =  %g degrees\n",thetaz/M_PI*180);
  printf("thetaaxis = %g degrees\n", thetaaxis/M_PI*180);
  printf("res =   %d pixels\n",res);
  printf("spread =   %g\n",spread);
  printf("thresh =   %g\n",thresh);
  printf("phi = %d\n",phi);
  printf("inside = %d\n",N - outside);

  // Free dynamic memory
  cudaFree(dev_x);
  cudaFree(dev_z);
  cudaFree(dev_marker1);
  cudaFree(dev_marker2);
  cudaFree(dev_phi);
  cudaFree(dev_outside);

  free(x);
  free(z);
  free(marker1);
  free(marker2);

  // // Print data to console
  // printf("%d, ",res);
  // printf("%d, ",N - outside);
  // printf("%d\n",phi);

  //   outside_arr[k] = N - outside;
  //   phi_arr[k] = phi;
  // }

  // // Open the output file on the host
  // FILE* outputfile = fopen(argv[2],"w");
  // fwrite(&Nres,      sizeof(int),    1,       outputfile);
  // fwrite(&T,         sizeof(int),    1,       outputfile);
  // fwrite(&spread,    sizeof(double), 1,       outputfile);
  // fwrite(res_arr,    sizeof(int)*Nres, 1,       outputfile);
  // fwrite(outside_arr,    sizeof(int)*Nres, 1,       outputfile);
  // fwrite(phi_arr,    sizeof(int)*Nres, 1,       outputfile);

  // // Close the output file
  // fclose(outputfile);

  // End the timing
  cudaEventRecord(stop,0);
  cudaEventSynchronize(stop);
  float elapsedTime;
  cudaEventElapsedTime(&elapsedTime, start, stop);

  // // Print the elapsed time
  // printf("Execution time =   %le seconds", elapsedTime/1000.);

  // End the program
  return 0;
}
