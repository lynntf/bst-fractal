/*
    Bi-axial Spherical Tumbler (BST) Peicewise Isometry Fractal
    
    Program written by Thomas Lynn

    Inputs: 
        file name of input file
        file name of output file

    Outputs: 
        file containing header data and output image
 */

// Program requires access to math.h and cuda.h
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda.h>

#ifndef M_PI
#define M_PI 3.1415926535897932384626433832795
#endif

__device__ static int    dev_N;
__device__ static int    dev_T;
__device__ static int    dev_res;
__device__ static int    dev_histogram;
__device__ static double dev_xcenter;
__device__ static double dev_zcenter;
__device__ static double dev_thresh;
__device__ static double dev_spread;
__device__ static double dev_Sb;
__device__ static double dev_Cb;
__device__ static double dev_Sa;
__device__ static double dev_Ca;
__device__ static double dev_Sg;
__device__ static double dev_Cg;
__device__ static int    dev_half_iter;

#include "init_new.h"
#include "project.h"

/*
void rotate(int t, double* x, double* y, double* z,
            double* marker1, double* marker2,
            int doprojection, int* phi, int* used)

This function rotates points according to the inverse BST PWI.

Inputs:
    int     t            The number of iterations.
    double* x            Vector of x locations of seed points.
    double* z            Vector of z locations of seed points.
    int*    marker1      Register 1.
    int*    marker2      Register 2.
    int     doprojection Type of projection from grid to sphere.
    int*    phi          Coverage (fraction of used points). Only used if 
                         returning just the coverage.
    int*    used         Counts the number of used points for computing
                         the coverage.

Outputs:
    Alters *marker1      The output array stores the information in
                         marker1 at the x and z indices.
    Alters *marker2      The output array stores the information in
                         marker2 at the x and z indices.
    Alters *phi          Outputs the coverage if just returning the
                         coverage.
    Alters *used         Outputs which pixels have been used for
                         computing the coverage.
 */

__global__ void rotate(int t, double* x, double* z, double* marker1,
                        double* marker2, int doprojection, int* phi,
                        int* used) {
    // Get the global index of the GPU core
    int gi = threadIdx.x + blockIdx.x * blockDim.x;
    // Protect un-initialized regions of the data
    if (gi < dev_res*dev_res) {
        // Store local copies of z and y
        double xl = x[gi/dev_res];
        double zl = z[gi%dev_res];
        double yl;
        double temp;
        int done1 = 0;
        int done2 = 0;
        int flip = 0;

        double marker1_l = 0;
        double marker2_l = 0;
        
        // Remove points that are outside the domain
        int ysign = -1;

        ///////////////////////////////////////////////
        // Project from the flat grid to the hemisphere
        ///////////////////////////////////////////////
        project(&xl, &zl, &yl, doprojection, &ysign);
        
        if (ysign > 0) {
            done1 = 1;
            done2 = 1;
            if (dev_histogram == 8) { // Count the number of grid
                                      // points inside unit circle
                atomicAdd(used,1);
            }
        }
        
        // Calculate the initial y-value
        yl = ysign*sqrt(abs(1 - xl*xl - zl*zl));
        // Initial positions
        double xl0 = xl;
        double zl0 = zl;
        double yl0 = yl;

        /////////////////////////////////////////////////////////////
        // What information to output (initial values for some cases)
        /////////////////////////////////////////////////////////////
        switch (dev_histogram) {
            case 0 : // Mark points near the boundary as done
                if ((abs(yl) < dev_thresh) && (!done2)) {
                    marker2_l = 1;
                    done2 = 1;
                }
                break;
            case 1 : // Add one to count of times near the boundary
                if ((abs(yl) < dev_thresh) && (ysign < 0)) {
                    marker2_l++;
                }
                break;
            case 3 : // 
                if (done1 == 0) {
                    marker1_l = 10;
                    marker2_l = 10;
                }
                break;
            case 9 : // min dist to a cut
                marker2_l = 10;
                marker1_l = 10;
                break;
        }
        ////////////////////////
        // Do several iterations
        ////////////////////////
        int k = 0;
        // k goes from 0 to t-1 as long as one color isn't done
        for(k = 0; (k < t) && (!done1 | !done2); k++) {
            temp = xl;
            // Rotate the axes back
            xl =   xl * dev_Cg  -  zl * dev_Sg;
            zl = temp * dev_Sg  +  zl * dev_Cg;
            
            // Skip this rotation for k==0 when a half-iteration is
            // specified (otherwise execute).
            if (k > 0 || dev_half_iter == 0) {  
                temp = yl;
                // Do the backwards rotation about x
                yl =   yl * dev_Cb  -  xl * dev_Sb;
                xl = temp * dev_Sb  +  xl * dev_Cb;
                
                // Flip the point over if it went past the edge (periodic
                // boundary)
                if (yl * temp < 0) {
                    yl = -yl;
                    xl = -xl;
                    flip = 1;
                }


                /////////////////////////////
                // What information to output
                /////////////////////////////
                switch (dev_histogram) {
                    case 0 : // First iteration passing close to boundary
                        if ((abs(yl) < dev_thresh) && (!done1)) {
                            marker1_l = k+1;
                            done1 = 1;
                            done2 = 1;
                        }
                        break;
                    case 1 : // Number of times passing close to boundary
                        if (abs(yl) < dev_thresh) {
                            marker1_l++;
                        }
                        break;
                    case 2 : // Iteration where point returns to initial
                             // position
                        if ((sqrt((xl-xl0)*(xl-xl0) + (zl-zl0)*(zl-zl0) +
                            (yl-yl0)*(yl-yl0)) < dev_thresh) && (!done1)) {
                            // If Euclidean distance is less than
                            // the threshold value, consider
                            // returned
                            marker1_l = k+1;
                            done1 = 1;
                        }
                        break;
                    case 3 : // Anglular position of interaction with
                             // boundary
                        if ((abs(yl) < dev_thresh) && (!done1)) {
                            // Check if near the discontinuity
                            temp = 0.5*(zl/(sqrt(xl*xl + zl*zl)) + 1);
                            // Rescale z -> [0,1] and post process
                            // to get angular position
                            marker1_l = (temp > marker1_l) ?
                                             marker1_l : temp;
                        }
                        break;
                    case 4: // Distance to boundary
                        marker1_l += asin(abs(yl));
                        break;
                    case 7: // Number of flips
                        marker1_l += flip;
                        flip = 0;
                        break;
                    case 8 : // Just coverage
                        if ((abs(yl) < dev_thresh) && (!done1)) {
                            atomicAdd(phi, 1);
                            done1 = 1;
                            done2 = 1;
                        }
                        break;
                    case 9 : // min dist to cutting line
                        marker1_l = (asin(abs(yl)) < marker1_l) ?
                                        asin(abs(yl)) : marker1_l;
                        break;

                    case 10 : // Last iteration passing close to 
                              // the boundary
                        if ((abs(yl) < dev_thresh) && !done1) {
                            marker1_l = k+1;
                        }
                        break;
                
                    case 11 : // Count of returns to initial position
                        if ((sqrt((xl-xl0)*(xl-xl0) +
                                 (zl-zl0)*(zl-zl0) +
                                 (yl-yl0)*(yl-yl0)) <
                            dev_thresh) && (!done1)) {
                            // If Euclidean distance is less than
                            // the threshold value, consider
                            // returned
                            marker1_l++;
                        }
                        break;
                    case 12 : // Sum up the distance to the initial point
                        marker1_l = marker1_l + 
                                    sqrt((xl-xl0)*(xl-xl0) +
                                         (zl-zl0)*(zl-zl0) +
                                         (yl-yl0)*(yl-yl0));
                        break;
                }
            }


            // Rotate the axes forward
            temp = xl;
            xl =   xl * dev_Cg  +  zl * dev_Sg;
            zl =-temp * dev_Sg  +  zl * dev_Cg;
            
            // Do the backwards rotation about z
            temp = yl;
            yl =   yl * dev_Ca  -  xl * dev_Sa;
            xl = temp * dev_Sa  +  xl * dev_Ca;
            
            // Flip the point over if it went past the edge
            // (periodic boundary)
            if (yl*temp < 0) {
                yl = -yl;
                xl = -xl;
                flip = 1;
            }

            /////////////////////////////
            // What information to output
            /////////////////////////////
            switch (dev_histogram) {
                case 0 : // First iteration passing close to boundary
                    if ((abs(yl) < dev_thresh) && (!done2)) {
                        marker2_l = k+1;
                        done2 = 1;
                    }
                    break;
                case 1 : // Number of times passing close to the boundary
                    if (abs(yl) < dev_thresh) {
                        marker2_l++;
                    }
                    break;
                case 2 : // Iteration where point returns to 
                         // initial position
                    if ((sqrt((xl-xl0)*(xl-xl0) + 
                             (zl-zl0)*(zl-zl0) + 
                             (yl-yl0)*(yl-yl0)) < 
                        dev_thresh) && (!done2)) {
                        marker2_l = k+1;
                        done2 = 1;
                    }
                    break;
                case 3 : // Anglular position of interaction with
                         // boundary
                    if ((abs(yl) < dev_thresh) && (!done2)) {
                        temp = 0.5*(zl/(sqrt(xl*xl + zl*zl)) + 1);
                        marker2_l = (temp > marker2_l) ?
                                    marker2_l : temp;
                    }
                    break;
                case 4 : // Distance to boundary
                    marker2_l += asin(abs(yl));
                    break;
                case 5 : // Stacked x and z locations
                    marker2_l += zl;
                    marker1_l += xl;
                    break;
                case 7: // Number of flips
                    marker2_l += flip;
                    flip = 0;
                    break;
                case 8 : // Just coverage
                    if ((abs(yl) < dev_thresh) && (!done1)) {
                        // int* temp = &dev_phi;
                        atomicAdd(phi, 1);
                        done1 = 1;
                    }
                    break;
                case 9 : // min dist to cutting line
                    marker2_l = (asin(abs(yl)) < marker2_l) ?
                                 asin(abs(yl)) : marker2_l;
                    break;
                case 10 : // Last iteration passing close to the
                          // boundary
                    if ((abs(yl) < dev_thresh) && (!done1)) {
                        marker2_l = k+1;
                    }
                    break;
                case 11 : // Count of returns to initial position
                    if ((sqrt((xl-xl0)*(xl-xl0) +
                                (zl-zl0)*(zl-zl0) +
                                (yl-yl0)*(yl-yl0)) <
                        dev_thresh) && (!done1)) {
                        // If Euclidean distance is less than
                        // the threshold value, consider
                        // returned
                        marker2_l++;
                    }
                    break;
                case 12 : // Sum up the distance to the initial point
                    marker2_l = marker2_l + 
                                sqrt((xl-xl0)*(xl-xl0) +
                                        (zl-zl0)*(zl-zl0) +
                                        (yl-yl0)*(yl-yl0));
                    break;
            }
        }

        //////////////////////////////////
        // Store data in the shared memory
        //////////////////////////////////
        switch (dev_histogram) {
        case 6 : // Final location
            if (ysign < 0) {
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
int main(int argc, char* argv[])

The main program executes a point rotation over a hemisphere with
periodic boundary conditions.

Inputs:
    input filename   The input file for heat which contains all of
                     the input data
    output filename  The output file containing N, T, x, y, and z

Outputs:
    int              The main program returns 0 if sucessful
    output file      The output file contains information about the
                     run and requested data

Console Outputs:
    N  =  ... points
    T  =  ... iterations
    beta =  ... degrees
    alpha =  ... degrees
    gamma = ... degrees
    res = ... pixels
    spread = ...
    thresh = ...
    phi = ...
    inside = ...
 */


int main(int argc, char* argv[]) {
    //////////////////////////////////
    // Setup
    //////////////////////////////////
    // Begin timing the computation
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start,0);

    // Choose the GPU card (on ESAM Kepler or Tesla)
    cudaDeviceProp prop;
    int dev = 0;
    // memset(&prop, 0, sizeof(cudaDeviceProp));
    // prop.multiProcessorCount = 13;
    // cudaChooseDevice(&dev, &prop);
    cudaSetDevice(dev);

    // Get the maximum thread count for the device
    cudaGetDeviceProperties(&prop, dev);
    int num_threads = prop.maxThreadsPerBlock;

    //////////////////////////////////
    // Input
    //////////////////////////////////
    // Open the input file on the host
    // Check if the inputs to the program are correct
    if (argc > 3 ) {
        printf ("Incorrect usage: only enter the input "
                "data file name and output data file "
                "name\n") ;
        return 0;
    }
    
    // Open the input file for reading
    FILE* inputfile = fopen(argv[1],"r");
    // Return an error if the input file couldn't be opened
    if (!inputfile) {
        printf("Unable to open input file\n");
        return 0;
    }
    // N points, T iterations, Do a projection or not, Do a
    // halfiteration or not
    int N, T, res, projection, histogram, half_iter;
    // Initalize beta, alpha, gamma, deltatheta doubles
    double beta, alpha, gamma, spread, xcenter, zcenter, line;
    spread = 0;
    line = 1;
    // Read in N, T, beta, alpha from the input file
    fscanf(inputfile,
    "[Number of Iterations] %d"
    " [Alpha] %lf"
    " [Beta] %lf"
    " [Gamma] %lf"
    " [Output Resolution] %d"
    " [Projection: 0 - Orthographic,"
    " 1 - Stereographic, 2 - Lambert EA,"
    " 3 - Gnomonic, 4 - Square Lambert] %d"
    " [Spread] %lf"
    " [x Center] %lf"
    " [z Center] %lf"
    " [Line Thickness] %lf"
    " [Output Data: 0 - Initial cuts,"
    " 1 - Stacked cuts, 2 - First return iter,"
    " 3 - Boundary location of initial cut,"
    " 4 - Stacked distance to boundary,"
    " 5 - Stacked location, 6 - Final location,"
    " 8 - Just Coverage, 9 - Minimum distance to cut] %d"
    " [Half iter] %d",
    &T, &alpha, &beta, &gamma, &res, &projection, &spread, &xcenter,
    &zcenter, &line, &histogram, &half_iter);

    // Print information to console for debugging, should be
    // identical to input file
    printf("[Number of Iterations] %d \n", T);
    printf("[Alpha] %lf  \n", alpha);
    printf("[Beta] %g  \n", beta);
    printf("[Gamma] %g  \n", gamma);
    printf("[Output Resolution] %d  \n", res);
    printf("[Projection: 0 - Orthographic,"
           " 1 - Stereographic, 2 - Lambert EA,"
           " 3 - Gnomonic,"
           " 4 - Square Lambert] %d  \n", projection);
    printf("[Spread] %g \n", spread);
    printf("[x Center] %g \n", xcenter);
    printf("[z Center] %lf \n", zcenter);
    printf("[Line Thickness] %lf \n", line);
    printf("[Output Data: 0 - Initial cuts,"
           " 1 - Stacked cuts, 2 - First return iter,"
           " 3 - Boundary location of initial cut,"
           " 4 - Stacked distance to boundary,"
           " 5 - Stacked location, 6 - Final location,"
           " 8 - Just Coverage,"
           " 9 - minimum distance to cut] %d \n", histogram);
    printf("[Half iter] %d \n", half_iter);

    beta = beta/180*M_PI;        // Convert to radians
    alpha = alpha/180*M_PI;        // Convert to radians
    gamma = gamma/180*M_PI;  // Convert to radians

    // Close the input file
    fclose(inputfile);


    N = res*res;
    double thresh = line;
    //double thresh = line*spread / res;
    // For a full Lambert equal area azimuthal projection, the line
    // threshold should be (5/4)*sqrt(2) to accomodate for the
    // distortion at the border and allow for continuous lines

    // Calculate the required number of blocks for initialization
    int num_blocks = (res)/num_threads + ((res)%num_threads ? 1 : 0);
    // Number of blocks to rotate all pixels
    int num_bigblocks = (N)/num_threads + ((N)%num_threads ? 1 : 0);

    // Reduce thread calls if possible
    if (N < num_threads) {
        num_threads = N;
    }

    double Sb = sin(beta);
    double Cb = cos(beta);
    double Sa = sin(alpha);
    double Ca = cos(alpha);
    double Sg = sin(gamma);
    double Cg = cos(gamma);
    int phi = 0;
    int outside = 0;
    //int half_iter = 1;

    ////////////////////////////
    // Memory
    ////////////////////////////
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
    cudaMemcpyToSymbol(dev_Sb,        &Sb,        sizeof(double));
    cudaMemcpyToSymbol(dev_Cb,        &Cb,        sizeof(double));
    cudaMemcpyToSymbol(dev_Sa,        &Sa,        sizeof(double));
    cudaMemcpyToSymbol(dev_Ca,        &Ca,        sizeof(double));
    cudaMemcpyToSymbol(dev_Sg,        &Sg,        sizeof(double));
    cudaMemcpyToSymbol(dev_Cg,        &Cg,        sizeof(double));

    // Allocate memory on the host
    double* x = (double*)malloc(res * sizeof(double));
    double* z = (double*)malloc(res * sizeof(double));
    // Two registers, one for each color (blue, red)
    double* marker1 = (double*)malloc(res * res * sizeof(double));
    double* marker2 = (double*)malloc(res * res * sizeof(double));

    // Allocate memory on the device
    double* dev_x;
    double* dev_z;
    double* dev_marker1;
    double* dev_marker2;
    int* dev_phi;     // Coverage (integer number)
    int* dev_outside; // Number of points in grid outside unit circle

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
    cudaMemset(dev_x,       0, res*sizeof(double));
    cudaMemset(dev_z,       0, res*sizeof(double));
    cudaMemset(dev_marker1, 0, res*res*sizeof(double));
    cudaMemset(dev_marker2, 0, res*res*sizeof(double));
    cudaMemset(dev_phi,     0, sizeof(int));
    cudaMemset(dev_outside, 0, sizeof(int));


    ///////////////////////////////////////////
    // Initial Conditions (x and z grid values)
    ///////////////////////////////////////////
    init_new<<<num_blocks,num_threads>>>(dev_x,dev_z);
    cudaError_t err2 = cudaGetLastError();
    if (err2 != cudaSuccess) 
        printf("Error: %s\n", cudaGetErrorString(err2));

    ///////////
    // Rotation
    ///////////
    rotate<<<num_bigblocks,num_threads>>>(T,dev_x,dev_z,
                                          dev_marker1, dev_marker2,
                                          projection, dev_phi,
                                          dev_outside);
    err2 = cudaGetLastError();
    if (err2 != cudaSuccess) 
        printf("Error: %s\n", cudaGetErrorString(err2));


    ////////////////////////////////////
    // Collect data on host and clean up
    ////////////////////////////////////
    // Copy memory from the device to the host
    cudaMemcpy(x,       dev_x,       res*sizeof(double),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(z,       dev_z,       res*sizeof(double),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(marker1, dev_marker1, res*res*sizeof(double),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(marker2, dev_marker2, res*res*sizeof(double),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(&phi,    dev_phi,     sizeof(int),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(&outside,dev_outside, sizeof(int),
               cudaMemcpyDeviceToHost);

    // Zero out data on the device
    cudaMemset(dev_marker1,0,res*res*sizeof(double));
    cudaMemset(dev_marker2,0,res*res*sizeof(double));

    // Open the output file on the host
    FILE* outputfile = fopen(argv[2],"wb");

    // Write information to the outputfile
    fwrite(&N,         sizeof(int),    1,       outputfile);
    fwrite(&T,         sizeof(int),    1,       outputfile);
    fwrite(&res,       sizeof(int),    1,       outputfile);
    fwrite(&beta,      sizeof(double), 1,       outputfile);
    fwrite(&alpha,     sizeof(double), 1,       outputfile);
    fwrite(&gamma,     sizeof(double), 1,       outputfile);
    fwrite(&xcenter,   sizeof(double), 1,       outputfile);
    fwrite(&zcenter,   sizeof(double), 1,       outputfile);
    fwrite(&spread,    sizeof(double), 1,       outputfile);
    fwrite(&histogram, sizeof(int),    1,       outputfile);
    fwrite(&thresh,    sizeof(double), 1,       outputfile);
    if (histogram != 8) {
        // Histogram == 8 implies we only want the value of the
        // coverage
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
    printf("beta =  %g degrees\n",beta/M_PI*180);
    printf("alpha =  %g degrees\n",alpha/M_PI*180);
    printf("gamma = %g degrees\n", gamma/M_PI*180);
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

    // End the timing
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);

    // Print the elapsed time
    printf("Execution time =   %le seconds", elapsedTime/1000.);

    // End the program
    return 0;
}