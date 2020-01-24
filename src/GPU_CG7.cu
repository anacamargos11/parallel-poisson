// Compiled with
// nvcc ./GPU_CG.cu -lcublas 
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <limits.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"

const int BLOCK_SIZE_X = 16;
const int BLOCK_SIZE_Y = 16;

__global__ void CGradientGPU(float* d_d, float* Ad_d, float* f_d, int SizeX, int SizeY, float h)
{
  int tx=threadIdx.x;
  int ty=threadIdx.y;
  int bx=blockIdx.x * blockDim.x;
  int by=blockIdx.y * blockDim.y;
  int x=tx+bx+1;  // need to shift over by one as we are only working on the interior
  int y=ty+by+1;

  Ad_d[x+y*SizeX] = 0.25f*(d_d[x+1+y*SizeX]+d_d[x-1+y*SizeX]+
			     d_d[x+(y+1)*SizeX]+d_d[x+(y-1)*SizeX]-h*h*f_d[x+y*SizeX]);
}

__global__ void rx_update(float* x_d, float* r_d, float lambda, float* d_d, float* u_d,
int SizeX, int SizeY)
{
  int tx=threadIdx.x;
  int ty=threadIdx.y;
  int bx=blockIdx.x * blockDim.x;
  int by=blockIdx.y * blockDim.y;
  int x=tx+bx+1;  // need to shift over by one as we are only working on the interior
  int y=ty+by+1;
  
  x_d[x+y*SizeX] += lambda*d_d[x+y*SizeX];
  r_d[x+y*SizeX] += lambda*u_d[x+y*SizeX];
}


void initialize(float *u, float *f, int xsize, int ysize)
{
  /* a particularly boring set of boundary conditions */
  //for(int i=0; i < ysize; i++)
  //  u[i*xsize] = u[i*xsize+xsize-1] =1.0;
  //for(int j=1; j< xsize; j++) 
  //  u[j] = u[(ysize-1)*xsize+j]=1.0;
  for(int k=ysize/4; k<3*ysize/4; k++) {
    f[ysize/4*xsize+k]=0.25;
    f[ysize/4*3*xsize+k]=-0.25;
  }
}


void Output(float *x, int xsize, int ysize)
{
  FILE *fp = fopen("Solution.txt","wt");
  for (int i=0; i<xsize; i++) {
    for (int j=0; j<ysize; j++)
      fprintf(fp," %f",x[j*xsize+i]);
    fprintf(fp,"\n");
  }
  fclose(fp);
}

int main(void)
{
  float *u_h, *f_h, *x_h;    // pointers to host memory
  float *u_d, *f_d, *x_d, *r_d, *d_d;    // pointers to device memory
  int ArraySizeX=1026;  // Here, ArraySize minus boundary layer must be exactly divisible by BLOCK_SIZE
  int ArraySizeY=1026;
  size_t size=ArraySizeX*ArraySizeY*sizeof(float);

  //Allocate arrays on host and initialize to zero
  u_h=(float *)calloc(ArraySizeX*ArraySizeY,sizeof(float));
  f_h=(float *)calloc(ArraySizeX*ArraySizeY,sizeof(float));
  x_h=(float *)calloc(ArraySizeX*ArraySizeY,sizeof(float));

  //Initialize arrays u_h and f_h boundaries
  initialize(u_h, f_h, ArraySizeX, ArraySizeY);

  //Allocate arrays on device
  cudaMalloc((void **) &d_d,size); 
  cudaMalloc((void **) &u_d,size);
  cudaMalloc((void **) &f_d,size);
  cudaMalloc((void **) &x_d,size);
  cudaMalloc((void **) &r_d,size);

  //Perform computation on GPU
  // Part 1 of 4: Copy data from host to device
  cudaMemcpy(d_d, u_h, size, cudaMemcpyHostToDevice);
  cudaMemcpy(x_d, u_h, size, cudaMemcpyHostToDevice);
  cudaMemcpy(r_d, u_h, size, cudaMemcpyHostToDevice);
  cudaMemcpy(u_d, u_h, size, cudaMemcpyHostToDevice);
  cudaMemcpy(f_d, f_h, size, cudaMemcpyHostToDevice);

  // Initialize BLAS context handle on GPU
  cublasHandle_t handle;
  cublasCreate(&handle);

  // Part 2 of 4: Set up execution configuration
  int nBlocksX=(ArraySizeX-2)/BLOCK_SIZE_X;
  int nBlocksY=(ArraySizeY-2)/BLOCK_SIZE_Y;
  printf("nBlocksX = %d nBlocksY = %d \n",nBlocksX,nBlocksY);

  dim3 dimBlock(BLOCK_SIZE_X,BLOCK_SIZE_Y);
  dim3 dimGrid(nBlocksX,nBlocksY);

  // first AD on x(0)
  CGradientGPU<<<dimGrid, dimBlock>>>(x_d,r_d,f_d,ArraySizeX,ArraySizeY, 1.0); 

  float delta, lambda, olddelta, alpha, lambda_dot;
  delta=0.0;
  //compute first delta
  cublasSdot(handle, (ArraySizeX)*(ArraySizeY),r_d, 1,r_d, 1,&delta);
  
  for (int nsteps=1; nsteps < 200000; nsteps++) {
    // Part 3 of 4: Call kernel with execution configuration
    CGradientGPU<<<dimGrid, dimBlock>>>(d_d,u_d,f_d,ArraySizeX,ArraySizeY, 1.0); 
    lambda_dot = 0.0; 
    lambda=0.0;
    // compute lambda
    cublasSdot(handle, (ArraySizeX)*(ArraySizeY),d_d, 1,u_d, 1,&lambda_dot); 
    lambda = delta/lambda_dot; 
    // update r and x
    cublasSaxpy(handle, (ArraySizeX)*(ArraySizeY), &lambda, d_d, 1, x_d, 1);
    cublasSaxpy(handle, (ArraySizeX)*(ArraySizeY), &lambda, u_d, 1, r_d, 1);    
  //  rx_update<<<dimGrid, dimBlock>>>(x_d,r_d,lambda,d_d,u_d,ArraySizeX,ArraySizeY);  
    // compute delta
    olddelta = delta;
    delta = 0.0;
    cublasSdot(handle, (ArraySizeX)*(ArraySizeY),r_d, 1,r_d, 1,&delta);
    // break if Tol rate achieved
    if (sqrt(delta) < 0.0001)
      break;
    // compute alpha and update d
    alpha = delta/olddelta;
      // d = alpha*d
    cublasSscal(handle, (ArraySizeX)*(ArraySizeY),&alpha, d_d, 1);    
      // d = -r + d
    float coef = -1.0;
    cublasSaxpy(handle, (ArraySizeX)*(ArraySizeY),&coef, r_d, 1, d_d,1);  
 
    //swap old and new arrays
    float *tmp = d_d;
    d_d = u_d;
    u_d = tmp; 
  }

  // Part 4 of 4: Copy result from device to host
  cudaMemcpy(x_h, x_d, size, cudaMemcpyDeviceToHost);

  // Output results
  Output(x_h, ArraySizeX, ArraySizeY);

  //Cleanup
  cublasDestroy(handle);
  cudaFree(d_d);
  cudaFree(u_d);
  cudaFree(f_d);
  cudaFree(x_d);
  cudaFree(r_d);
  free(u_h);
  free(f_h);
  free(x_h);
}

