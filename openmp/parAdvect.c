// OpenMP parallel 2D advection solver module
// written for COMP4300/8300 Assignment 2, 2017
// v1.0 28 Apr 

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <omp.h>
#include "serAdvect.h" // advection parameters

static int M, N, P, Q; // local store of problem parameters
static int verbosity;
static int P0, Q0; // 2D thread id
int M0,N0; //local field starting point (M0,N0)
int M_loc, N_loc; // local advection field size (excluding halo)

//sets up parameters above
void initParParams(int M_, int N_, int P_, int Q_, int verb) {
  M = M_, N = N_; P = P_, Q = Q_;
  verbosity = verb;
} //initParParams()


// evolve advection over reps timesteps, with (u,ldu) containing the field
// using 1D parallelization
void omp1dAdvect(int reps, double *u, int ldu) {
  int r, i, j;
  int ldut = M+1;
  double *ut = malloc(ldut*(N+1)*sizeof(double));

  for (r = 0; r < reps; r++) {
    double sx = 0.5 * Velx / deltax, sy = 0.5 * Vely / deltay;
    double dtdx = 0.5 * dt / deltax, dtdy = 0.5 * dt / deltay;

#pragma omp parallel for schedule(static) default(shared)
    for (i=1; i < M+1; i++) { //update left & right boundaries
      V(u, i, 0)   = V(u, i, N);
      V(u, i, N+1) = V(u, i, 1);
    }
#pragma omp parallel for schedule(static) default(shared)
    for (j=0; j < N+2; j++) { //update top & bottom boundaries
      V(u, 0, j)   = V(u, M, j);
      V(u, M+1, j) = V(u, 1, j);      
    }

    u = &V(u, 1, 1); // make u relative to the interior points for the updates

// 1. maximize performance
#pragma omp parallel for schedule(static) default(shared) private(i)
    for (j=0; j < N+1; j++) // advection update stage 1
      for (i=0; i < M+1; i++) 
	V(ut,i,j) = 0.25*(V(u,i,j) + V(u,i-1,j) + V(u,i,j-1) + V(u,i-1,j-1))
	  -0.5*dt*(sy*(V(u,i,j) + V(u,i,j-1) - V(u,i-1,j) - V(u,i-1,j-1)) +
		   sx*(V(u,i,j) + V(u,i-1,j) - V(u,i,j-1) - V(u,i-1,j-1)));

#pragma omp parallel for schedule(static) default(shared) private(i)
    for (j=0; j < N; j++) // advection update stage 2
      for (i=0; i < M; i++) 
	V(u, i, j) +=
	  - dtdy * (V(ut,i+1,j+1) + V(ut,i+1,j) - V(ut,i,j) - V(ut,i,j+1))
	  - dtdx * (V(ut,i+1,j+1) + V(ut,i,j+1) - V(ut,i,j) - V(ut,i+1,j));

// 2. maximize the number of OpenMP parallel region entry/exits
/*
    if ( N > M ) {
      for (j=0; j < N+1; j++) // advection update stage 1
#pragma omp parallel for schedule(static) default(shared)
        for (i=0; i < M+1; i++) 
  	V(ut,i,j) = 0.25*(V(u,i,j) + V(u,i-1,j) + V(u,i,j-1) + V(u,i-1,j-1))
  	  -0.5*dt*(sy*(V(u,i,j) + V(u,i,j-1) - V(u,i-1,j) - V(u,i-1,j-1)) +
  		   sx*(V(u,i,j) + V(u,i-1,j) - V(u,i,j-1) - V(u,i-1,j-1)));
  
      for (j=0; j < N; j++) // advection update stage 2
#pragma omp parallel for schedule(static) default(shared)
        for (i=0; i < M; i++) 
  	V(u, i, j) +=
  	  - dtdy * (V(ut,i+1,j+1) + V(ut,i+1,j) - V(ut,i,j) - V(ut,i,j+1))
  	  - dtdx * (V(ut,i+1,j+1) + V(ut,i,j+1) - V(ut,i,j) - V(ut,i+1,j));
    } else {
      for (i=0; i < M+1; i++) // advection update stage 1
#pragma omp parallel for schedule(static) default(shared)
        for (j=0; j < N+1; j++) 
  	V(ut,i,j) = 0.25*(V(u,i,j) + V(u,i-1,j) + V(u,i,j-1) + V(u,i-1,j-1))
  	  -0.5*dt*(sy*(V(u,i,j) + V(u,i,j-1) - V(u,i-1,j) - V(u,i-1,j-1)) +
  		   sx*(V(u,i,j) + V(u,i-1,j) - V(u,i,j-1) - V(u,i-1,j-1)));
  
      for (i=0; i < M; i++) // advection update stage 2
#pragma omp parallel for schedule(static) default(shared)
        for (j=0; j < N; j++) 
  	V(u, i, j) +=
  	  - dtdy * (V(ut,i+1,j+1) + V(ut,i+1,j) - V(ut,i,j) - V(ut,i,j+1))
  	  - dtdx * (V(ut,i+1,j+1) + V(ut,i,j+1) - V(ut,i,j) - V(ut,i+1,j));
    }
*/

// 3. maximize shared cache misses involving coherent reads
/*
#pragma omp parallel for schedule(static,1) default(shared) private(i)
    for (j=0; j < N+1; j++) // advection update stage 1
      for (i=0; i < M+1; i++) 
	V(ut,i,j) = 0.25*(V(u,i,j) + V(u,i-1,j) + V(u,i,j-1) + V(u,i-1,j-1))
	  -0.5*dt*(sy*(V(u,i,j) + V(u,i,j-1) - V(u,i-1,j) - V(u,i-1,j-1)) +
		   sx*(V(u,i,j) + V(u,i-1,j) - V(u,i,j-1) - V(u,i-1,j-1)));

#pragma omp parallel for schedule(static,1) default(shared) private(i)
    for (j=0; j < N; j++) // advection update stage 2
      for (i=0; i < M; i++) 
	V(u, i, j) +=
	  - dtdy * (V(ut,i+1,j+1) + V(ut,i+1,j) - V(ut,i,j) - V(ut,i,j+1))
	  - dtdx * (V(ut,i+1,j+1) + V(ut,i,j+1) - V(ut,i,j) - V(ut,i+1,j));
*/

// 4. maximize shared cache misses involving coherent writes
/*
    for (j=0; j < N+1; j++) // advection update stage 1
#pragma omp parallel for schedule(static,1) default(shared)
      for (i=0; i < M+1; i++) 
	V(ut,i,j) = 0.25*(V(u,i,j) + V(u,i-1,j) + V(u,i,j-1) + V(u,i-1,j-1))
	  -0.5*dt*(sy*(V(u,i,j) + V(u,i,j-1) - V(u,i-1,j) - V(u,i-1,j-1)) +
		   sx*(V(u,i,j) + V(u,i-1,j) - V(u,i,j-1) - V(u,i-1,j-1)));

    for (j=0; j < N; j++) // advection update stage 2
#pragma omp parallel for schedule(static,1) default(shared)
      for (i=0; i < M; i++) 
	V(u, i, j) +=
	  - dtdy * (V(ut,i+1,j+1) + V(ut,i+1,j) - V(ut,i,j) - V(ut,i,j+1))
	  - dtdx * (V(ut,i+1,j+1) + V(ut,i,j+1) - V(ut,i,j) - V(ut,i+1,j));
*/

    u = &V(u, -1, -1); // restore to include the boundary points
  } //for (r...)

  free(ut);
} //omp1dAdvect()


// ... using 2D parallelization
void omp2dAdvect(int reps, double *u, int ldu) {
  int r, i, j;
  int ldut = M+1;
  double *ut = malloc(ldut*(N+1)*sizeof(double));
  int itmp, jtmp;

  int num_threads, tid;
  num_threads = omp_get_num_threads();

#pragma omp parallel default(shared) private(i,j,tid,Q0,P0,N0,M0,N_loc,M_loc,itmp,jtmp,r) 
{
  for (r = 0; r < reps; r++) {
    double sx = 0.5 * Velx / deltax, sy = 0.5 * Vely / deltay;
    double dtdx = 0.5 * dt / deltax, dtdy = 0.5 * dt / deltay;

    tid = omp_get_thread_num();

    Q0 = tid % Q;
    N0 = (N / Q) * Q0;
    N_loc = (Q0 < Q-1)? (N / Q): (N - N0); 
  
    P0 = tid / Q; 
    M0 = (M / P) * P0;
    M_loc = (P0 < P-1)? (M / P): (M - M0); 

    if ( Q0 == 0 ) { //update left & right boundaries
      for (i=M0+1; i < M0+M_loc+1; i++) { 
        V(u, i, 0)   = V(u, i, N);
        V(u, i, N+1) = V(u, i, 1);
      }
    } 
#pragma omp barrier
    if ( P0 == 0 ) { //update top & bottom boundaries
      for (j=N0+1; j < N0+N_loc+1; j++) { 
        V(u, 0, j)   = V(u, M, j);
        V(u, M+1, j) = V(u, 1, j);      
      }
      if ( Q0 == 0 ) {
          V(u, 0, 0)   = V(u, M, 0);
          V(u, M+1, 0) = V(u, 1, 0);      
      } else if ( Q0 == Q-1 ) {
          V(u, 0, N+1)   = V(u, M, N+1);
          V(u, M+1, N+1) = V(u, 1, N+1);      
      }
    }

#pragma omp barrier
#pragma omp single
    {u = &V(u, 1, 1);} // make u relative to the interior points for the updates

    jtmp = (Q0==Q-1) ? (N_loc+1) : N_loc;
    itmp = (P0==P-1) ? (M_loc+1) : M_loc;
    for (j=N0; j < N0+jtmp; j++) // advection update stage 1
      for (i=M0; i < M0+itmp; i++) 
	V(ut,i,j) = 0.25*(V(u,i,j) + V(u,i-1,j) + V(u,i,j-1) + V(u,i-1,j-1))
	  -0.5*dt*(sy*(V(u,i,j) + V(u,i,j-1) - V(u,i-1,j) - V(u,i-1,j-1)) +
		   sx*(V(u,i,j) + V(u,i-1,j) - V(u,i,j-1) - V(u,i-1,j-1)));

#pragma omp barrier
    for (j=N0; j < N0+N_loc; j++) // advection update stage 2
      for (i=M0; i < M0+M_loc; i++) 
	V(u, i, j) +=
	  - dtdy * (V(ut,i+1,j+1) + V(ut,i+1,j) - V(ut,i,j) - V(ut,i,j+1))
	  - dtdx * (V(ut,i+1,j+1) + V(ut,i,j+1) - V(ut,i,j) - V(ut,i+1,j));

#pragma omp barrier
#pragma omp single
    {u = &V(u, -1, -1);} // restore to include the boundary points
  } //for (r...)
} // omp directive

  free(ut);
} //omp2dAdvect()


// ... extra optimization variant
// using SIMD to do extra optimization
void ompAdvectExtra(int reps, double *u, int ldu) {
  int r, i, j;
  int ldut = M+1;
  double *ut = malloc(ldut*(N+1)*sizeof(double));

  for (r = 0; r < reps; r++) {
    double sx = 0.5 * Velx / deltax, sy = 0.5 * Vely / deltay;
    double dtdx = 0.5 * dt / deltax, dtdy = 0.5 * dt / deltay;

#pragma omp parallel for schedule(static) default(shared)
    for (i=1; i < M+1; i++) { //update left & right boundaries
      V(u, i, 0)   = V(u, i, N);
      V(u, i, N+1) = V(u, i, 1);
    }
#pragma omp parallel for schedule(static) default(shared)
    for (j=0; j < N+2; j++) { //update top & bottom boundaries
      V(u, 0, j)   = V(u, M, j);
      V(u, M+1, j) = V(u, 1, j);      
    }

    u = &V(u, 1, 1); // make u relative to the interior points for the updates

#pragma omp parallel for schedule(static) default(shared) private(i)
    for (j=0; j < N+1; j++) // advection update stage 1
#pragma omp simd
      for (i=0; i < M+1; i++) 
	V(ut,i,j) = 0.25*(V(u,i,j) + V(u,i-1,j) + V(u,i,j-1) + V(u,i-1,j-1))
	  -0.5*dt*(sy*(V(u,i,j) + V(u,i,j-1) - V(u,i-1,j) - V(u,i-1,j-1)) +
		   sx*(V(u,i,j) + V(u,i-1,j) - V(u,i,j-1) - V(u,i-1,j-1)));

#pragma omp parallel for schedule(static) default(shared) private(i)
    for (j=0; j < N; j++) // advection update stage 2
#pragma omp simd
      for (i=0; i < M; i++) 
	V(u, i, j) +=
	  - dtdy * (V(ut,i+1,j+1) + V(ut,i+1,j) - V(ut,i,j) - V(ut,i,j+1))
	  - dtdx * (V(ut,i+1,j+1) + V(ut,i,j+1) - V(ut,i,j) - V(ut,i+1,j));

    u = &V(u, -1, -1); // restore to include the boundary points
  } //for (r...)

  free(ut);
} //ompAdvectExtra()
