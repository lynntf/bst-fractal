/*
void init_new(double*x, double* z)

This function puts the initial positions into the global memory
blocks being used. Points are seeded as two vectors, values are
picked from these when they need to be accessed in the rotation
function.

Inputs:
    double* x, z        Arrays for initial conditions storage

Outputs:
    Alters *x, *z       Initial conditions are stored.
 */

__global__ void init_new(double* x, double* z) {
  // Get the global index
  int gi = threadIdx.x + blockIdx.x * blockDim.x;

  if (gi < dev_res){ // Protect memory outside of
                     // initialization
    // Starting at the left edge, add half a pixel and then the
    // rest to get into position
    x[gi] = dev_xcenter - dev_spread +
            (dev_spread/dev_res) + gi*(2*dev_spread/dev_res);
    z[gi] = dev_zcenter - dev_spread +
            (dev_spread/dev_res) + gi*(2*dev_spread/dev_res);
  }
}