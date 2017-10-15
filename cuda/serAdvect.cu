// serial 2D advection solver module
// written by Peter Strazdins, Mar 17 for COMP4300/8300 Assignment 1 
// updated (slightly) Apr 17 for COMP4300/8300 Assignment 2
// v1.0 28 Apr 

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h> // sin(), fabs()

#include "serAdvect.h"

void cudaHandleError(cudaError_t err, const char *file, int line) {
  if (err != cudaSuccess) {
    printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
    exit(EXIT_FAILURE);
  }
}

// advection parameters
static const double CFL = 0.25;      // CFL condition number
const double Velx = 1.0, Vely = 1.0; //advection velocity
double dt;                           //time for 1 step
double deltax, deltay;               //grid spacing//

static int M, N;

void initAdvectParams(int M_, int N_) {
  M = M_; N = N_;
  assert (M > 0 && N > 0); // advection not defined for empty grids
  deltax = 1.0 / N;
  deltay = 1.0 / M;
  dt = CFL * (deltax < deltay? deltax: deltay);
  // printf("dx=%e dy=%e dt=%e\n", deltax, deltay, dt);
}

static double initCond(double x, double y, double t) {
  x = x - Velx*t;
  y = y - Vely*t;
  return (sin(4.0*M_PI*x) * sin(2.0*M_PI*y)) ;
}

void initAdvectField(int M0, int N0, int m, int n, double *u, int ldu) {
  int i, j;
  for (j=0; j < n; j++) {
    double x = deltax * (j + N0);
    for (i=0; i < m; i++) {
      double y = deltay * (i + M0);
      V(u, i, j) = initCond(x, y, 0.0);
    }
  }
}

double errAdvectField(int r, int M0, int N0, int m, int n, double *u, int ldu){
  int i, j;
  double err = 0.0;
  double t = r * dt;
  for (j=0; j < n; j++) {
    double x = deltax * (j + N0);
    for (i=0; i < m; i++) {
      double y = deltay * (i + M0);
      err += fabs(V(u, i, j) - initCond(x, y, t));
    }
  }
  return (err);
}

const int AdvFLOPsPerElt = 23; //count 'em

void printAdvectField(std::string label, int m, int n, double *u, int ldu) {
  int i, j;
  printf("%s\n", label.c_str());
  for (i=0; i < m; i++) {
    for (j=0; j < n; j++) 
      printf(" %+0.2f", V(u, i, j));
    printf("\n");
  }
}


void hostAdvectSerial(int reps, double *u, int ldu) {
  int i, j, ldut = M+1;
  double *ut = (double *) malloc(ldut*(N+1)*sizeof(double));

  for (int r = 0; r < reps; r++) {
    double sx = 0.5 * Velx / deltax, sy = 0.5 * Vely / deltay;
    double dtdx = 0.5 * dt / deltax, dtdy = 0.5 * dt / deltay;

    for (i=1; i < M+1; i++) { //update left & right boundariesK
      V(u, i, 0)   = V(u, i, N);
      V(u, i, N+1) = V(u, i, 1);
    }
    for (j=0; j < N+2; j++) { //update top & bottom boundaries
      V(u, 0, j)   = V(u, M, j);
      V(u, M+1, j) = V(u, 1, j);
    }
    u = &V(u, 1, 1); // make u relative to the interior points for the updates

    for (j=0; j < N+1; j++) // advection update stage 1
      for (i=0; i < M+1; i++)
        V(ut,i,j) = 0.25*(V(u,i,j) + V(u,i-1,j) + V(u,i,j-1) + V(u,i-1,j-1))
          -0.5*dt*(sy*(V(u,i,j) + V(u,i,j-1) - V(u,i-1,j) - V(u,i-1,j-1)) +
                   sx*(V(u,i,j) + V(u,i-1,j) - V(u,i,j-1) - V(u,i-1,j-1)));

    for (j=0; j < N; j++) // advection update stage 2
      for (i=0; i < M; i++)
        V(u, i, j) +=
          - dtdy * (V(ut,i+1,j+1) + V(ut,i+1,j) - V(ut,i,j) - V(ut,i,j+1))
          - dtdx * (V(ut,i+1,j+1) + V(ut,i,j+1) - V(ut,i,j) - V(ut,i+1,j));

    u = &V(u, -1, -1); // restore to include the boundary points      
  } //for(r...)
  free(ut);
} //hostAdvectSerial()


/********************** serial CPU area **********************************/

__global__ void updateBoundaryEW(int M, int N, double *u, int ldu) {
  for (int i=1; i < M+1; i++) { 
    V(u, i, 0)   = V(u, i, N);
    V(u, i, N+1) = V(u, i, 1);
  }
}

__global__ void updateBoundaryNS(int N, int M, double *u, int ldu) {
  for (int j=0; j < N+2; j++) { 
    V(u, 0, j)   = V(u, M, j);
    V(u, M+1, j) = V(u, 1, j);
  }  
}

__global__ void updateAdvect1(int M, int N, double *u, int ldu, double *ut, 
			      int ldut, double dt, double sx, double sy) {
  for (int j=0; j < N+1; j++) 
    for (int i=0; i < M+1; i++)
      V(ut,i,j) = 0.25*(V(u,i,j) + V(u,i-1,j) + V(u,i,j-1) + V(u,i-1,j-1))
	-0.5*dt*(sy*(V(u,i,j) + V(u,i,j-1) - V(u,i-1,j) - V(u,i-1,j-1)) +
		 sx*(V(u,i,j) + V(u,i-1,j) - V(u,i,j-1) - V(u,i-1,j-1)));
} 

__global__ void updateAdvect2(int M, int N, double *u, int ldu, double *ut, 
			      int ldut, double dtdx, double dtdy) {
  for (int j=0; j < N; j++) 
    for (int i=0; i < M; i++)
      V(u, i, j) +=
	- dtdy * (V(ut,i+1,j+1) + V(ut,i+1,j) - V(ut,i,j) - V(ut,i,j+1))
	- dtdx * (V(ut,i+1,j+1) + V(ut,i,j+1) - V(ut,i,j) - V(ut,i+1,j));
}

// evolve advection over reps timesteps, with (u,ldu) containing the field
// serial GPU implementation 
void cudaAdvectSerial(int reps, double *u, int ldu) {
  int ldut = M+1;
  double *ut;
  HANDLE_ERROR( cudaMalloc(&ut, ldut*(N+1)*sizeof(double)) );
  
  for (int r = 0; r < reps; r++) {
    updateBoundaryEW <<<1,1>>> (M, N, u, ldu);
    updateBoundaryNS <<<1,1>>> (N, M, u, ldu);

    double sx = 0.5 * Velx / deltax, sy = 0.5 * Vely / deltay;
    updateAdvect1 <<<1,1>>> (M, N, &V(u,1,1), ldu, ut, ldut, dt, sx, sy);

    double dtdx = 0.5 * dt / deltax, dtdy = 0.5 * dt / deltay;
    updateAdvect2 <<<1,1>>> (M, N, &V(u,1,1), ldu, ut, ldut, dtdx, dtdy);
  } //for(r...)
    
  HANDLE_ERROR( cudaFree(ut) );
} //cudaAdvectSerial()

