// CUDA 2D advection solver test program
// written by Peter Strazdins, Apr 17 for COMP4300/8300 Assignment 2
// v1.0 28 Apr 

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h> //getopt()
#include <assert.h>
#include <sys/time.h> //gettimeofday()
#include <string>   //std::string

#include "serAdvect.h"
#include "parAdvect.h"

#define USAGE   "testAdvect [-h] [-s] [-g Gx[,Gy]] [-b Bx[,By]] [-o] [-w w] [-v v] M N [r]"
#define DEFAULTS "Gx=Gy=Bx=By=r=1"
#define OPTCHARS "hsg:b:ow:v:"

static int M, N;               // advection field size
static int Gx=1, Gy=1;         // grid dimensions
static int Bx=1, By=1;         // (thread) block dimensions
static int r = 1;              // number of timesteps for the simulation
static int optH = 0;           // set if -h specified
static int optS = 0;           // set if -s specified
static int optO = 0;           // set if -o specified
static int verbosity = 0;      // v, above
static int w = 0;              // optional extra tuning parameter

// print a usage message for this program and exit with a status of 1
void usage(std::string msg) {
  printf("testAdvect: %s\n", msg.c_str());
  printf("usage: %s\n\tdefault values: %s\n", USAGE, DEFAULTS);
  fflush(stdout);
  exit(1);
}

void getArgs(int argc, char *argv[]) {
  extern char *optarg; // points to option argument (for -p option)
  extern int optind;   // index of last option parsed by getopt()
  extern int opterr;
  int optchar;        // option character returned my getopt()
  opterr = 0;          // suppress getopt() error message for invalid option
  while ((optchar = getopt(argc, argv, OPTCHARS)) != -1) {
    // extract next option from the command line     
    switch (optchar) {
    case 'h':
      optH = 1;
      break;
    case 's':
      optS = 1;
      break;
    case 'g':
      if (sscanf(optarg, "%d,%d", &Gx, &Gy) < 1) // invalid integer 
	usage("bad value for Gx");
      break;
    case 'b':
      if (sscanf(optarg, "%d,%d", &Bx, &By) < 1) // invalid integer 
	usage("bad value for Bx");
      break;
    case 'o':
      optO = 1;
      break;
    case 'w':
      if (sscanf(optarg, "%d", &w) != 1) // invalid integer 
	usage("bad value for w");
      break;
    case 'v':
      if (sscanf(optarg, "%d", &verbosity) != 1) // invalid integer 
	usage("bad value for v");
      break;
    default:
      usage("unknown option");
      break;
    } //switch 
   } //while

  if (optind < argc) {
    if (sscanf(argv[optind], "%d", &M) != 1) 
      usage("bad value for M");
  } else
    usage("missing M");
  N = M;
  if (optind+1 < argc)
    if (sscanf(argv[optind+1], "%d", &N) != 1) 
      usage("bad value for N");
  if (optind+2 < argc)
    if (sscanf(argv[optind+2], "%d", &r) != 1) 
      usage("bad value for r");
} //getArgs()


static void printAvgs(std::string name, double total, int nVals) {
  printf("Average %s %.3e\n", name.c_str(), total / nVals);
}

//return wall time in seconds
static double Wtime() {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return(1.0*tv.tv_sec + 1.0e-6*tv.tv_usec);
}

int main(int argc, char** argv) {
  double *u; int ldu; //local advection field
  double t, gflops; //time

  getArgs(argc, argv);

  printf("Advection of a %dx%d global field on %s" 
	 " for %d steps.\n", M, N, optH? "host": "GPU", r);
  if (optS)
    printf("\tusing serial computation\n");
  else if (optO)
    printf("\tusing optimizations (Gx,Gy=%d,%d Bx,By=%d,%d w=%d)\n", 
	   Gx, Gy, Bx, By, w);
  else if (!optH)
    printf("\tusing %dx%d blocks of %dx%d threads (1D decomposition)\n", 
	   Gx, Gy, Bx, By);  
  initAdvectParams(M, N);  
  initParParams(M, N, Gx, Gy, Bx, By, verbosity);

  ldu = M+2;
  HANDLE_ERROR( cudaMallocManaged(&u, ldu*(N+2)*sizeof(double) ));
  initAdvectField(0, 0, M, N, &V(u,1,1), ldu);
  if (verbosity > 1)
    printAdvectField("init u", M, N, &V(u,1,1), ldu);

  t = Wtime();
  if (optH)
    hostAdvectSerial(r, u, ldu);
  else if (optS)
    cudaAdvectSerial(r, u, ldu);
  else if (optO)    
    cudaOptAdvect(r, u, ldu, w); 
  else
    cuda2DAdvect(r, u, ldu);
  cudaDeviceSynchronize();
  t = Wtime() - t;

  gflops = 1.0e-09 * AdvFLOPsPerElt * M * N * r;
  printf("Advection time %.3e, GFLOPs rate=%.2e\n", t, gflops / t); 

  if (verbosity > 1)
    printAdvectField("final u", M+2, N+2, u, ldu);
  printAvgs("error of final field: ", 
	    errAdvectField(r, 0, 0, M, N, &V(u,1,1), ldu), M*N);

  HANDLE_ERROR( cudaFree(u) );
  return 0;
} //main()

