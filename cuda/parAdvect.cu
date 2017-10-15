// OpenMP parallel 2D advection solver module
// written for COMP4300/8300 Assignment 2, 2017
// v1.0 28 Apr 

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "serAdvect.h" // advection parameters

static int M, N, Gx, Gy, Bx, By; // local store of problem parameters
static int verbosity;

//sets up parameters above
void initParParams(int M_, int N_, int Gx_, int Gy_, int Bx_, int By_, 
		   int verb) {
  M = M_, N = N_; Gx = Gx_; Gy = Gy_;  Bx = Bx_; By = By_; 
  verbosity = verb;
} //initParParams()

/****************kernel for parallelization PAR2D*******************/
__global__ void updateBoundaryEWpar(int M, int N, double *u, int ldu) {
  int xid = threadIdx.x + blockDim.x * blockIdx.x;
  int yid = threadIdx.y + blockDim.y * blockIdx.y;
  int numy = blockDim.y * gridDim.y; // total threads at y direction
  int numx = blockDim.x * gridDim.x; // total threads at x direction
  int id1D = xid + yid*numx; // the 1D thread id
  for (int i=(id1D+1); i < M+1; i+=numy*numx) { 
    V(u, i, 0)   = V(u, i, N);
    V(u, i, N+1) = V(u, i, 1);
  }
}

__global__ void updateBoundaryNSpar(int N, int M, double *u, int ldu) {
  int xid = threadIdx.x + blockDim.x * blockIdx.x;
  int yid = threadIdx.y + blockDim.y * blockIdx.y;
  int numy = blockDim.y * gridDim.y; // total threads at y direction
  int numx = blockDim.x * gridDim.x; // total threads at x direction
  int id1D = xid + yid*numx; // the 1D thread id
  for (int j=id1D; j < N+2; j+=numx*numy) { 
    V(u, 0, j)   = V(u, M, j);
    V(u, M+1, j) = V(u, 1, j);
  }  
}

__global__ void updateAdvect1par(int M, int N, double *u, int ldu, double *ut, 
			      int ldut, double dt, double sx, double sy) {
  int xid = threadIdx.x + blockDim.x * blockIdx.x;
  int yid = threadIdx.y + blockDim.y * blockIdx.y;
  int numy = blockDim.y * gridDim.y; // total threads at y direction
  int numx = blockDim.x * gridDim.x; // total threads at x direction
  for (int j=xid; j < N+1; j+=numx) 
    for (int i=yid; i < M+1; i+=numy)
      V(ut,i,j) = 0.25*(V(u,i,j) + V(u,i-1,j) + V(u,i,j-1) + V(u,i-1,j-1))
	-0.5*dt*(sy*(V(u,i,j) + V(u,i,j-1) - V(u,i-1,j) - V(u,i-1,j-1)) +
		 sx*(V(u,i,j) + V(u,i-1,j) - V(u,i,j-1) - V(u,i-1,j-1)));
} 

__global__ void updateAdvect2par(int M, int N, double *u, int ldu, double *ut, 
			      int ldut, double dtdx, double dtdy) {
  int xid = threadIdx.x + blockDim.x * blockIdx.x;
  int yid = threadIdx.y + blockDim.y * blockIdx.y;
  int numy = blockDim.y * gridDim.y; // total threads at y direction
  int numx = blockDim.x * gridDim.x; // total threads at x direction
  for (int j=xid; j < N; j+=numx) 
    for (int i=yid; i < M; i+=numy)
      V(u, i, j) +=
	- dtdy * (V(ut,i+1,j+1) + V(ut,i+1,j) - V(ut,i,j) - V(ut,i,j+1))
	- dtdx * (V(ut,i+1,j+1) + V(ut,i,j+1) - V(ut,i,j) - V(ut,i+1,j));
}

// evolve advection over reps timesteps, with (u,ldu) containing the field
// parallel (2D decomposition) variant
void cuda2DAdvect(int reps, double *u, int ldu) {
  int ldut = M+1;
  double *ut;
  HANDLE_ERROR( cudaMalloc(&ut, ldut*(N+1)*sizeof(double)) );
  
  dim3 dimG(Gx, Gy);
  dim3 dimB(Bx, By);
  
  for (int r = 0; r < reps; r++) {
    updateBoundaryEWpar <<<dimG,dimB>>> (M, N, u, ldu);
    updateBoundaryNSpar <<<dimG,dimB>>> (N, M, u, ldu);

    double sx = 0.5 * Velx / deltax, sy = 0.5 * Vely / deltay;
    updateAdvect1par <<<dimG,dimB>>> (M, N, &V(u,1,1), ldu, ut, ldut, dt, sx, sy);

    double dtdx = 0.5 * dt / deltax, dtdy = 0.5 * dt / deltay;
    updateAdvect2par <<<dimG,dimB>>> (M, N, &V(u,1,1), ldu, ut, ldut, dtdx, dtdy);
  } //for(r...)
    
  HANDLE_ERROR( cudaFree(ut) );
} //cuda2DAdvect()


/****************kernel for parallelization OPT *******************/
__device__ void updateAdvect1Opt(int M, int N, double *u, int ldu, double *ut, 
			      int ldut, double dt, double sx, double sy) {
  int xid = threadIdx.x + blockDim.x * blockIdx.x;
  int yid = threadIdx.y + blockDim.y * blockIdx.y;
  int numy = blockDim.y * gridDim.y; // total threads at y direction
  int numx = blockDim.x * gridDim.x; // total threads at x direction
  for (int j=xid; j < N+1; j+=numx) 
    for (int i=yid; i < M+1; i+=numy)
      V(ut,i,j) = 0.25*(V(u,i,j) + V(u,i-1,j) + V(u,i,j-1) + V(u,i-1,j-1))
	-0.5*dt*(sy*(V(u,i,j) + V(u,i,j-1) - V(u,i-1,j) - V(u,i-1,j-1)) +
		 sx*(V(u,i,j) + V(u,i-1,j) - V(u,i,j-1) - V(u,i-1,j-1)));
} 

__device__ void updateAdvect2Opt(int M, int N, double *u, int ldu, double *ut, 
			      int ldut, double dtdx, double dtdy) {
  int xid = threadIdx.x + blockDim.x * blockIdx.x;
  int yid = threadIdx.y + blockDim.y * blockIdx.y;
  int numy = blockDim.y * gridDim.y; // total threads at y direction
  int numx = blockDim.x * gridDim.x; // total threads at x direction
  for (int j=xid; j < N; j+=numx) 
    for (int i=yid; i < M; i+=numy)
      V(u, i, j) +=
	- dtdy * (V(ut,i+1,j+1) + V(ut,i+1,j) - V(ut,i,j) - V(ut,i,j+1))
	- dtdx * (V(ut,i+1,j+1) + V(ut,i,j+1) - V(ut,i,j) - V(ut,i+1,j));
}

// usiing shared memory.
// 1. cp boundary from global memory to shared memory
// 2. cp center grid from global memory to shared memory
// 3. updateAdvect1
// 4. synchronize
// 5. updateAdvect2
// 6. synchronize
// 7. write cu back to u

__global__ void cudaOptAdvectKernel(int M, int N, double *u, int ldu, double *ut, int ldut, double dt, double sx, double sy,double dtdx, double dtdy) {

  int xid = threadIdx.x + blockDim.x * blockIdx.x;
  int yid = threadIdx.y + blockDim.y * blockIdx.y;
  int numy = blockDim.y * gridDim.y; // total threads at y direction
  int numx = blockDim.x * gridDim.x; // total threads at x direction

  extern __shared__ char shared[];
  double *cu = (double *) shared;
  double *cut = (double *)((M+2)*(N+2)*sizeof(double) + shared);
  int ldcu = M + 2;
  int ldcut = M + 1;

  // cp boundary from global memory to shared memory
  if ( threadIdx.x == 0 ) { // East
    for (int j=xid+1; j < N+1; j+=numx) 
      for (int i=yid+1; i < M+1; i+=numy)
        V(cu, i, j-1)   = V(u, i, j-1);
  }
  if ( threadIdx.x == blockDim.x-1 ) { // West
    for (int j=xid+1; j < N+1; j+=numx) 
      for (int i=yid+1; i < M+1; i+=numy)
        V(cu, i, j+1)   = V(u, i, j+1);
  }
  if ( threadIdx.y == 0 ) { // North
    for (int j=xid+1; j < N+1; j+=numx){ 
      for (int i=yid+1; i < M+1; i+=numy) {
        V(cu, i-1, j)   = V(u, i-1, j);
        if ( threadIdx.x == 0 ) {
          V(cu, i-1, j-1) = V(u,i-1,j-1);
        } else if ( threadIdx.x == blockDim.x - 1 ) {
          V(cu, i-1, j+1) = V(u, i-1, j+1);
        }
      }
    }
  }
  if ( threadIdx.y == blockDim.y-1 ) { // East
    for (int j=xid+1; j < N+1; j+=numx) {
      for (int i=yid+1; i < M+1; i+=numy){
        V(cu, i+1, j)   = V(u, i+1, j);
        if ( threadIdx.x == 0 ) {
          V(cu, i+1, j-1) = V(u,i+1,j-1);
        } else if ( threadIdx.x == blockDim.x - 1 ) {
          V(cu, i+1, j+1) = V(u, i+1, j+1);
        }
      }
    }
  }
  // cp center grid from global memory to shared memory
  // pay attention to +2, it update the west and south boundary of cu, if total threads cannot be divided by M,N
  for (int j=xid+1; j < N+1; j+=numx)  
    for (int i=yid+1; i < M+1; i+=numy)
      V(cu,i,j) = V(u,i,j);
  __syncthreads();
  // updateAdvect1
  updateAdvect1Opt(M, N, &V(cu,1,1), ldcu, cut, ldcut, dt, sx, sy);
  // synchronize
  __syncthreads();
  // updateAdvect2
  updateAdvect2Opt(M, N, &V(cu,1,1), ldcu, cut, ldcut, dtdx, dtdy);
  // synchronize
  __syncthreads();
  // write cu back to u
  for (int j=xid+1; j < N+1; j+=numx) 
    for (int i=yid+1; i < M+1; i+=numy)
      V(u,i,j) = V(cu,i,j);
}

// ... optimized parallel variant
// using shared memory to do optimization
void cudaOptAdvect(int reps, double *u, int ldu, int w) {
  int ldut = M+1;
  double *ut;
  HANDLE_ERROR( cudaMalloc(&ut, ldut*(N+1)*sizeof(double)) );
  
  dim3 dimG(Gx, Gy);
  dim3 dimB(Bx, By);
  //int sharedDim = (Bx+2) * (By+2);
  
  for (int r = 0; r < reps; r++) {
    double sx = 0.5 * Velx / deltax, sy = 0.5 * Vely / deltay;
    double dtdx = 0.5 * dt / deltax, dtdy = 0.5 * dt / deltay;

    updateBoundaryEWpar <<<dimG,dimB>>> (M, N, u, ldu);
    updateBoundaryNSpar <<<dimG,dimB>>> (N, M, u, ldu);
    cudaOptAdvectKernel <<<dimG, dimB,((M+2)*(N+2)+(M+1)*(N+1))*sizeof(double)>>> (M, N, u, ldu, ut, ldut, dt, sx, sy, dtdx, dtdy);
  } //for(r...)
    
  HANDLE_ERROR( cudaFree(ut) );
} //cudaOptAdvect()
//void cudaOptAdvect(int reps, double *u, int ldu, int w) { }
