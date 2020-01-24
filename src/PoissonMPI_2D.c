#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <limits.h>
#include "mpi.h"
// Solve Poisson's equations using the Conjugate-Gradient method, with MPI parallization
// dividing the grid into a 2D mesh of processors 

#define N  21             /* N x N is size of interior grid */
#define Tol  0.0001
#define maxitr 200 //2*N
#define h 0.1

int isperiodic[2]={0,0};  // Flags for periodic boundary conditions are off

// MPI Stuff
struct mpi_vars {
  int NProcs;
  int MyID;
  MPI_Comm D1Comm; // 1D communicator
  int nbrdown, nbrup;

  MPI_Comm D2Comm; // 2D communicator
  int dims[2], IDcoord[2]; //  processors in each direction in 2D. dims[0] is the number of 
  int nbrleft, nbrright; // procs in the x direction and dims[1] in the y direction
}; 

struct mpi_vars mpi_start(int argc, char** argv, int dim)
{
  struct mpi_vars mpi;

  MPI_Init(&argc, &argv);

  // We always want a 1D array of processors so just duplicate default 
  MPI_Comm_dup(MPI_COMM_WORLD, &mpi.D1Comm);

  MPI_Comm_size(mpi.D1Comm, &mpi.NProcs);
  if (dim == 1) {
    MPI_Comm_rank(mpi.D1Comm, &mpi.MyID);
    mpi.nbrup = mpi.MyID + 1;
    mpi.nbrdown = mpi.MyID -1;

    mpi.D2Comm = MPI_COMM_NULL;
  }

  // We sometimes want a 2D array so define if requested
  if (dim == 2) {
    // Find an integer closest to the square root of the processor number
    mpi.dims[0] = (int) (sqrt(((double)mpi.NProcs))+2.*FLT_EPSILON);
    // Assign as many as possible to other dimension (could leave some unused)
    mpi.dims[1] = mpi.NProcs/mpi.dims[0];

    /* Create cartesian grid of processors */
    MPI_Cart_create(MPI_COMM_WORLD, 2, mpi.dims, isperiodic, 1, &mpi.D2Comm);
    MPI_Comm_rank(mpi.D2Comm, &mpi.MyID); //need rank from 2D Comm
    if (mpi.MyID == 0)
      printf("number of processors %i: %i x %i\n", 
	     mpi.NProcs, mpi.dims[0], mpi.dims[1]);
  
    MPI_Cart_coords(mpi.D2Comm, mpi.MyID, 2, mpi.IDcoord);
    MPI_Cart_shift(mpi.D2Comm, 0, 1, &mpi.nbrleft, &mpi.nbrright);
    MPI_Cart_shift(mpi.D2Comm, 1, 1, &mpi.nbrdown, &mpi.nbrup);
  }

  return mpi;
}

// breakup array size N x N in 2D into parts for each processor
void compute_my_size(int *xsize, int *ysize, int *sameview, struct mpi_vars mpi)
{
  int xremainder = N  % mpi.dims[0];
  *xsize = (N - xremainder)/mpi.dims[0];
  if(mpi.IDcoord[0] < xremainder)
    *xsize = *xsize + 2;
  else 
    *xsize = *xsize + 1;
  
  int yremainder = N  % mpi.dims[1];
  *ysize = (N - yremainder)/mpi.dims[1];
  if(mpi.IDcoord[1] < yremainder)
    *ysize = *ysize + 2;
  else
    *ysize = *ysize + 1;

  *sameview = !(xremainder || yremainder);

  printf("compute grid (1 + %i + 1) x (1 + %i + 1) set on processor %i \n",
	*xsize-1,*ysize-1,mpi.MyID);
}

double **matrix(int m, int n)
{
    /* Note that you must allocate the array as one block in order to use */
    /* MPI derived data types on them.  Note also that calloc initializes */
    /* all entries to zero. */
    
    double ** ptr = (double **)calloc(m, sizeof(double *));
    ptr[0]=(double *)calloc(m*n, sizeof(double));
    for(int i = 1; i < m ;i++)
      ptr[i]=ptr[i-1]+n;
    return (ptr);
}

void initialize(double ** r, double **x, int xsize, int ysize, struct mpi_vars mpi)
{
  /* Subtract off b from a particularly boring set of boundary conditions where x=1 on the boundary */
  for(int i = 1; i < ysize; i++){
    r[i][1] -= 1;
    r[i][xsize-1] -= 1;
    x[i][0]=1;    // we won't use the x on the boundary but this will be useful for the output at the end
    x[i][xsize]=1;
  }
  for(int j = 1; j < xsize; j++){
    r[1][j] -= 1;
    r[ysize-1][j] -= 1;
    x[0][j]=1;
    x[ysize][j]=1;
  }
  x[0][0]=x[0][xsize]=x[ysize][0]=x[ysize][xsize]=1;    /* fill in the corners */
}

void AD(double **Ad, double **d, int xstart, int xfinish, 
		int ystart, int yfinish)
{ // Compute Ad= A*d using the auxiliary layer around the outside that is all zero 
  // for d to account for boundary */
  int i, j;
  for(i = ystart; i < yfinish; i++)
    for(j = xstart; j < xfinish; j++){
      Ad[i][j] = -(d[i+1][j] + d[i-1][j] + d[i][j+1] + d[i][j-1]-4.0*d[i][j]);
    }
  return;
}

/* Output result */
void output(double **x, int xsize, int ysize, struct mpi_vars mpi)
{
  char str[20];
  FILE *fp;

  sprintf(str,"SolutionCG2D%d_%d.Txt",mpi.IDcoord[0],mpi.IDcoord[1]);
  fp = fopen(str,"wt");

  if(mpi.IDcoord[1] == 0) {
    if (mpi.IDcoord[0] == 0)
      fprintf(fp,"%6.4f ",x[0][0]);
    for(int j = 0; j < xsize+1; j++)
      fprintf(fp,"%6.4f ",x[0][j]);
    if (mpi.IDcoord[0] == mpi.dims[0]-1)
      fprintf(fp,"%6.4f ",x[0][xsize]);
    fprintf(fp,"\n");
  }
  for(int i = 0; i < ysize+1; i++) {
    if (mpi.IDcoord[0] == 0)
      fprintf(fp,"%6.4f ",x[i][0]);
    for(int j = 0; j < xsize+1; j++)
      fprintf(fp,"%6.4f ",x[i][j]);
    if (mpi.IDcoord[0] == mpi.dims[0]-1)
      fprintf(fp,"%6.4f ",x[i][xsize]);
    fprintf(fp,"\n");
  }
  if(mpi.IDcoord[1] == mpi.dims[1] - 1) {
    if (mpi.IDcoord[0] == 0)
      fprintf(fp,"%6.4f ",x[ysize][0]);
    for(int j = 0; j < xsize+1; j++)
      fprintf(fp,"%6.4f ",x[ysize][j]);
    if (mpi.IDcoord[0] == mpi.dims[0]-1)
      fprintf(fp,"%6.4f ",x[ysize][xsize]);
    fprintf(fp,"\n");
  }
  fclose(fp);
}

int main(int argc, char** argv) 
{   
  // Start MPI with a 2D cartesian grid of processors
  struct mpi_vars mpi = mpi_start(argc, argv, 2);

  MPI_Status status;
  MPI_Request req_send10, req_send20, req_recv10, req_recv20;
  MPI_Request req_send30, req_send40, req_recv30, req_recv40;

  double delta, lambda, olddelta, alpha, deltaG, lambdaG;

  /* breakup compute grid amoung processors*/
  int xsize,ysize;
  int sameview;
  compute_my_size(&xsize, &ysize, &sameview, mpi);

  /* create arrays and initialize to all zeros */
  /* Note: Although these are column vectors in the algorithm, they represent locations */
  /* in 2D space so it is easiest to store them as 2D arrays */
  double **x = matrix(ysize+1,xsize+1);
  double **r = matrix(ysize+1,xsize+1);
  double **d = matrix(ysize+1,xsize+1);
  double **u = matrix(ysize+1,xsize+1);

  /* Work out A x(0), note that we make use of the x being zero on the boundary so that */
  /* we don't need the boundaries to be special cases in the iteration equations */ 
  AD(r,x,1,xsize,1,ysize);

  initialize(r,x,xsize,ysize,mpi); // setups boundary conditions

  /* Create data type to send columns of data */
  /* If you didn't get this to work, look at the comment in the matrix */
  /* allocation routine */
  MPI_Datatype stridetype;
  MPI_Type_vector(ysize-1,1,xsize+1,MPI_DOUBLE,&stridetype);
  MPI_Type_commit(&stridetype);

  /* Fill in d */
  for (int i=1; i<ysize; i++)
    for (int j=1; j<xsize; j++)
      d[i][j]=-r[i][j];

  /* Compute first delta on each proc, then AllReduce*/
  delta=0.0;
  deltaG=0.0;
  for (int i=1; i<ysize; i++)
    for (int j=1; j<xsize; j++)
      delta+= r[i][j]*r[i][j];
  MPI_Allreduce(&delta, &deltaG,1,MPI_DOUBLE, MPI_SUM, mpi.D2Comm);
  printf("itr = %i deltaG = %g\r",0,deltaG);

  /* Main loop */
  int itr;
  for (itr=0; itr< maxitr; itr++){
 //   AD(u,d,1,size); // local call on interior subgrids

    /* send/receive at top of compute block */
    req_send10 = req_recv20 = MPI_REQUEST_NULL;
    if(mpi.IDcoord[1] < mpi.dims[1]-1){
      MPI_Isend(&d[ysize-1][1],xsize-1,MPI_DOUBLE,mpi.nbrup,10,mpi.D2Comm,&req_send10);
      MPI_Irecv(&d[ysize][1],xsize-1,MPI_DOUBLE,mpi.nbrup,20,mpi.D2Comm,&req_recv20);
    }

    /* send/receive at bottom of compute block */
    req_send20 = req_recv10 = MPI_REQUEST_NULL;
    if(mpi.IDcoord[1] > 0){
      MPI_Isend(&d[1][1],xsize-1,MPI_DOUBLE,mpi.nbrdown,20,mpi.D2Comm,&req_send20);
      MPI_Irecv(&d[0][1],xsize-1,MPI_DOUBLE,mpi.nbrdown,10,mpi.D2Comm,&req_recv10);
    }

    /* send/receive at right of compute block */
    req_send30 = req_recv40 = MPI_REQUEST_NULL;
    if(mpi.IDcoord[0] < mpi.dims[0]-1){
      MPI_Isend(&d[1][xsize-1],1,stridetype,mpi.nbrright,30,mpi.D2Comm,&req_send30);
      MPI_Irecv(&d[1][xsize],1,stridetype,mpi.nbrright,40,mpi.D2Comm,&req_recv40);
    }

    /* send/receive at left of compute block */
    req_send40 = req_recv30 = MPI_REQUEST_NULL;
    if(mpi.IDcoord[0] > 0){
      MPI_Isend(&d[1][1],1,stridetype,mpi.nbrleft,40,mpi.D2Comm,&req_send40);
      MPI_Irecv(&d[1][0],1,stridetype,mpi.nbrleft,30,mpi.D2Comm,&req_recv30);
    }

    /* update interior of compute block excluding beside boundaries */
    AD(u,d,2,xsize-1,2,ysize-1); // local call on interior subgrids

    /* update compute block beside boundaries as available */
    /* top edge */
    if(mpi.IDcoord[1] < mpi.dims[1]-1) MPI_Wait(&req_recv20,&status);
    AD(u,d,2,xsize-1,ysize-1,ysize);

    /* bottom edge */
    if(mpi.IDcoord[1] > 0) MPI_Wait(&req_recv10,&status);
    AD(u,d,2,xsize-1,1,2);

    /* right edge */
    if(mpi.IDcoord[0] < mpi.dims[0]-1) MPI_Wait(&req_recv40,&status);
    AD(u,d,xsize-1,xsize,1,ysize);

    /* left edge */
    if(mpi.IDcoord[0] > 0) MPI_Wait(&req_recv30,&status);
    AD(u,d,1,2,1,ysize);

    lambda=0.0;
    lambdaG=0.0;
    for (int i=1; i<ysize; i++)
      for (int j=1;j<xsize; j++)
	lambda+= d[i][j]*u[i][j];
    MPI_Allreduce(&lambda, &lambdaG,1,MPI_DOUBLE, MPI_SUM, mpi.D2Comm);    
    lambdaG=deltaG/lambdaG;

    for (int i=1; i<ysize; i++)
      for (int j=1;j<xsize; j++) {
	x[i][j] += lambdaG*d[i][j];
	r[i][j] += lambdaG*u[i][j];
      }

    olddelta=deltaG;
    delta=0.0;
    deltaG=0.0;
    for (int i=1; i<ysize; i++)
      for (int j=1; j<xsize; j++)
	delta+= r[i][j]*r[i][j];
    MPI_Allreduce(&delta, &deltaG,1,MPI_DOUBLE, MPI_SUM, mpi.D2Comm);
    printf("itr = %i deltaG = %g\r",itr,deltaG);

    if (sqrt(deltaG) < Tol)
      break;
    
    alpha=deltaG/olddelta;
    for (int i=1; i<ysize; i++)
      for (int j=1;j<xsize; j++) {
	d[i][j] = -r[i][j]+alpha*d[i][j];
      } 
  }

  printf("\nComplete in %d iterations \n",itr);

  /* Output result */
  output(x,xsize,ysize,mpi);

  free(r[0]);
  free(r);
  free(d[0]);
  free(d);
  free(u[0]);
  free(u);
  free(x[0]);
  free(x);
  MPI_Finalize();
  return 0;
}





