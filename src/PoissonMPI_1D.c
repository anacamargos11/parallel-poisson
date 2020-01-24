#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "mpi.h"
// Solve Poisson's equations using the Conjugate-Gradient method, with MPI parallization
// dividing the grid into a 1D array of processors 

#define N  100
#define Tol  0.0001
#define maxitr 2*N
#define h 0.1

// MPI Stuff
struct mpi_vars {
  int NProcs;
  int MyID;
}; 

struct mpi_vars mpi_start(int argc, char** argv)
{
  struct mpi_vars this_mpi;

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &this_mpi.NProcs);
  MPI_Comm_rank(MPI_COMM_WORLD, &this_mpi.MyID);

  return this_mpi;
}

// breakup array size N in 1D into parts for each processor
int compute_my_size(struct mpi_vars the_mpi)
{
  int remainder = (N - 1) % the_mpi.NProcs;
  int size = (N - 1 - remainder)/the_mpi.NProcs;
  if(the_mpi.MyID < remainder)  // extra rows added for MyID < remainder 
    size = size + 2;
  else
    size = size + 1;
  return size; // returns size of subgrid for each proc.
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
    return (ptr); // returns a C matrix using calloc()
}

void initialize(double ** r, double **x, int size)
{
  /* Subtract off b from a particularly boring set of boundary conditions where x=1 on the boundary */
  for(int i = 1; i < size; i++){
    r[i][1] -= 1;
    r[i][N-1] -= 1;
    x[i][0]=1;    // we won't use the x on the boundary but this will be useful for the output at the end
    x[i][N]=1;
  }
  for(int j = 1; j < N; j++){
    r[1][j] -= 1;
    r[size-1][j] -= 1;
    x[0][j]=1;
    x[size][j]=1;
  }
  x[0][0]=x[0][N]=x[size][0]=x[size][N]=1;    /* fill in the corners */
}

void AD(double **Ad, double **d, int start, int finish)
{ // Compute Ad= A*d using the auxiliary layer around the outside that is all zero 
  // for d to account for boundary */
  int i, j;
  for(i = start; i < finish; i++)
    for(j = 1; j < N; j++){
      Ad[i][j] = -(d[i+1][j] + d[i-1][j] + d[i][j+1] + d[i][j-1]-4.0*d[i][j]);
    }
  return;
}

/* Output result */
void output(double **x, int size, struct mpi_vars the_mpi)
{
  char str[20];
  FILE *fp;

  sprintf(str,"SolutionCG%d.Txt",the_mpi.MyID);
  fp = fopen(str,"w");
  if(the_mpi.MyID == 0) {
    for(int j = 0; j < N + 1; j++)
      fprintf(fp,"%6.4f ",x[0][j]);
    fprintf(fp,"\n");
  }
  for(int i = 0; i < size+1; i++) {
    for(int j = 0; j < N + 1; j++)
      fprintf(fp,"%6.4f ",x[i][j]);
    fprintf(fp,"\n");
  }
  if(the_mpi.MyID == the_mpi.NProcs - 1){
    for(int j = 0; j < N + 1; j++)
      fprintf(fp,"%6.4f ",x[size][j]);
    fprintf(fp,"\n");
  }
  fclose(fp);
}
int main(int argc, char** argv) 
{   
  struct mpi_vars the_mpi = mpi_start(argc, argv);

  MPI_Status status;
  MPI_Request req_send10, req_send20, req_recv10, req_recv20;  

  double delta, lambda, olddelta, alpha, deltaG, lambdaG;

  /* breakup compute grid amoung processors and initialize*/
  int size = compute_my_size(the_mpi);

  /* create arrays and initialize to all zeros */
  /* Note: Although these are column vectors in the algorithm, they represent locations */
  /* in 2D space so it is easiest to store them as 2D arrays */
  double **x = matrix(size+1,N+1);
  double **r = matrix(size+1,N+1);
  double **d = matrix(size+1,N+1);
  double **u = matrix(size+1,N+1);

  /* Work out A x(0), note that we make use of the x being zero on the boundary so that */
  /* we don't need the boundaries to be special cases in the iteration equations */ 
  AD(r,x,1,size);

  initialize(r,x,size); // setups boundary conditions

  /* Fill in d */
  for (int i=1; i<size; i++)
    for (int j=1; j<N; j++)
      d[i][j]=-r[i][j];

  /* Compute first delta on each proc, then AllReduce*/
  delta=0.0;
  deltaG=0.0;
  for (int i=1; i<size; i++)
    for (int j=1; j<N; j++)
      delta+= r[i][j]*r[i][j];
  MPI_Allreduce(&delta, &deltaG,1,MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  printf("itr = %i deltaG = %g\r",0,deltaG);

  /* Main loop */
  int itr;
  for (itr=0; itr< maxitr; itr++){
 //   AD(u,d,1,size); // local call on interior subgrids

    /* send/receive at top of compute block */
    req_send10 = req_recv20 = MPI_REQUEST_NULL;
    if(the_mpi.MyID < the_mpi.NProcs - 1){
      MPI_Isend(&d[size-1][1], N-1, MPI_DOUBLE, the_mpi.MyID+1, 10,
		MPI_COMM_WORLD, &req_send10);
      MPI_Irecv(&d[size][1], N-1, MPI_DOUBLE, the_mpi.MyID+1, 20,
		MPI_COMM_WORLD,&req_recv20);
    }
    /* send/receive at bottom of compute block */
    req_send20 = req_recv10 = MPI_REQUEST_NULL;
    if(the_mpi.MyID > 0){
      MPI_Isend(&d[1][1], N-1, MPI_DOUBLE, the_mpi.MyID-1, 20,
		MPI_COMM_WORLD, &req_send20);
      MPI_Irecv(&d[0][1], N-1, MPI_DOUBLE, the_mpi.MyID-1, 10,
		MPI_COMM_WORLD, &req_recv10);
    }

    /* update interior of compute block excluding beside boundaries */
    AD(u,d,2,size-1); // local call on interior subgrids

    /* update compute block beside boundaries as available */
    /* top edge */
    if(the_mpi.MyID < the_mpi.NProcs - 1)
      MPI_Wait(&req_recv20, &status);
    AD(u,d,size-1,size);

    /* bottom edge */
    if(the_mpi.MyID > 0)
      MPI_Wait(&req_recv10, &status);
    AD(u,d,1,2);

    lambda=0.0;
    lambdaG=0.0;
    for (int i=1; i<size; i++)
      for (int j=1;j<N; j++)
	lambda+= d[i][j]*u[i][j];
    MPI_Allreduce(&lambda, &lambdaG,1,MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);    
    lambdaG=deltaG/lambdaG;

    for (int i=1; i<size; i++)
      for (int j=1;j<N; j++) {
	x[i][j] += lambdaG*d[i][j];
	r[i][j] += lambdaG*u[i][j];
      }

    olddelta=deltaG;
    delta=0.0;
    deltaG=0.0;
    for (int i=1; i<size; i++)
      for (int j=1; j<N; j++)
	delta+= r[i][j]*r[i][j];
    MPI_Allreduce(&delta, &deltaG,1,MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    printf("itr = %i deltaG = %g\r",itr,deltaG);

    if (sqrt(deltaG) < Tol)
      break;
    
    alpha=deltaG/olddelta;
    for (int i=1; i<size; i++)
      for (int j=1;j<N; j++) {
	d[i][j] = -r[i][j]+alpha*d[i][j];
      } 
  }

  printf("\nComplete in %d iterations \n",itr);

  /* Output result */
  output(x,size,the_mpi);

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





