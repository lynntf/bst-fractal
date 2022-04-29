/*
Projects grid points x,z to 3D points on the sphere (x,y,z) according
to the specified projection.
*/

__device__ void project(double* x, double* z, double* y,
                            int projection, int* ysign) {
    double xl = *x;
    double yl = *y;
    double zl = *z;
    int ys = *ysign;
    double temp = xl;
    double rho, c, phi_g, lamb, r, phi;
    switch (projection){
        case 1 :
            // Stereographic projection
            if (xl*xl + zl*zl > 1) {
                ys = 1;
            }
            xl = 2*xl/(1+ xl*xl+ zl*zl);
            zl = 2*zl/(1+ temp*temp+ zl*zl);
            break;
        case 2 :
            // Lambert equal-area projection
            if (xl*xl + zl*zl > 1) {
                ys = 1;
            }
            xl = sqrt(1.0 - (xl*xl + zl*zl) /2.0 )*xl*sqrt(2.0);
            zl = sqrt(1.0 - (temp*temp + zl*zl) /2.0 )*zl*sqrt(2.0);
            break;
        case 3 :
            // Gnomonic projection
            xl = xl*5;
            zl = zl*5;
            rho = sqrt(xl*xl + zl*zl);
            c = atan(rho);
            phi_g = asin(cos(c));
            lamb = atan2(xl*sin(c),(-zl*sin(c)));
            xl = cos(phi_g)*cos(lamb);
            zl = cos(phi_g)*sin(lamb);
            if (xl*xl + zl*zl > 1) {
                ys = 1;
            }
            break;
        case 4 :
            // Square to disk projection due to Shirely (1997)
            r = -zl;
            phi = 0;
            if ((xl > -zl) && (xl > zl)) { // Region 1
                r = xl;
                phi = M_PI/4.0 * (zl/xl);
            } else if ((xl > -zl) && (xl <= zl)) { // Region 2
                r = zl;
                phi = M_PI/4.0 * (2 - (xl/zl));
            } else if ((xl <= -zl) && (xl < zl)) { // Region 3
                r = -xl;
                phi = M_PI/4.0 * (4 + (zl/xl));
            } else if ((xl <= -zl) && (xl >= zl) && (zl != 0)) { // Region 4
                r = -zl;
                phi = M_PI/4.0 * (6 - (xl/zl));
            }
            xl = r * cos(phi);
            zl = r * sin(phi);
            // Remove points that are outside the domain
            if (xl*xl + zl*zl <= 1) {
                ys = 0;
            }
            // Lambert equal-area projection
            xl = sqrt(1.0 - (xl*xl + zl*zl) /2.0 )*xl*sqrt(2.0);
            zl = sqrt(1.0 - (temp*temp + zl*zl) /2.0 )*zl*sqrt(2.0);
            break;
    }
    *x = xl;
    *y = yl;
    *z = zl;
    *ysign = ys;
}