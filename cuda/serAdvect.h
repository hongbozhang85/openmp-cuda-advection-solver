// serial 2D advection solver module
// written by Peter Strazdins, Mar 17 for COMP4300/8300 Assignment 1 
// updated (slightly) Apr 17 for COMP4300/8300 Assignment 2
// v1.0 28 Apr
#include <string>   //std::string

#define HANDLE_ERROR( err ) (cudaHandleError( err, __FILE__, __LINE__ ))
void cudaHandleError(cudaError_t err, const char *file, int line);

// number of FLOPs to update a single element in the advection function 
extern const int AdvFLOPsPerElt; 

// parameters needed for advection solvers
extern const double Velx, Vely; //advection velocity                
extern double dt;               //time for 1 step
extern double deltax, deltay;   //grid spacing

// initializes the advection paamters for a global M x N field 
void initAdvectParams(int M, int N);

// access element (i,j) of array u with leading dimension ldu
#define V(u, i, j) u[(i) + (ld##u)*(j)]

// initialize (non-halo elements) of a m x n local advection field (u, ldu)
//    local element [0,0] is element [M0,N0] in the global field
void initAdvectField(int M0, int N0, int m, int n, double *u, int ldu);

// sum errors in an m x n local advection field (u, ldu) after r timesteps 
//    local element [0,0] is element [M0,N0] in the global field 
double errAdvectField(int r, int M0, int N0, int m, int n, double *u, int ldu);

// print out  m x n local advection field (u, ldu) 
void printAdvectField(std::string label, int m, int n, double *u, int ldu);

// evolve advection on host over r timesteps, with (u,ldu) storing the field
void hostAdvectSerial(int r, double *u, int ldu);

// evolve advection on GPU over r timesteps, with (u,ldu) storing the field
void cudaAdvectSerial(int r, double *u, int ldu);

// kernels that it uses
__global__ void updateBoundaryEW(int M, int N, double *u, int ldu);
__global__ void updateBoundaryNS(int N, int M, double *u, int ldu);
__global__ void updateAdvect1(int M, int N, double *u, int ldu, double *ut,
                              int ldut, double dt, double sx, double sy);
__global__ void updateAdvect2(int M, int N, double *u, int ldu, double *ut,
                              int ldut, double dtdx, double dtdy);

