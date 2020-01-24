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

__global__ void d_fill(float* d_d, float* r_d, int SizeX, int SizeY)
{
  int tx=threadIdx.x;
  int ty=threadIdx.y;
  int bx=blockIdx.x * blockDim.x;
  int by=blockIdx.y * blockDim.y;
  int x=tx+bx+1;  // shift by 1
  int y=ty+by+1;
  
   d_d[x+y*SizeX] = - r_d[x+y*SizeX];
}

__global__ void CGradientGPU(float* d_d, float* Ad_d, int SizeX, int SizeY, float h)
{
  int tx=threadIdx.x;
  int ty=threadIdx.y;
  int bx=blockIdx.x * blockDim.x;
  int by=blockIdx.y * blockDim.y;
  int x=tx+bx+1;  // shift by 1
  int y=ty+by+1;

  Ad_d[x+y*SizeX] = -(d_d[x+1+y*SizeX]+d_d[x-1+y*SizeX]+
			     d_d[x+(y+1)*SizeX]+d_d[x+(y-1)*SizeX]-4.0f*d_d[x+y*SizeX]);
}

__global__ void rx_update(float* x_d, float* r_d, float lambda, float* d_d, float* u_d,
int SizeX, int SizeY)
{
  int tx=threadIdx.x;
  int ty=threadIdx.y;
  int bx=blockIdx.x * blockDim.x;
  int by=blockIdx.y * blockDim.y;
  int x=tx+bx+1;  // shift by 1
  int y=ty+by+1;
  
  x_d[x+y*SizeX] += lambda*d_d[x+y*SizeX];
  r_d[x+y*SizeX] += lambda*u_d[x+y*SizeX];
}

void initialize(float *x, float *r, int xsize, int ysize)
{
  for(int i=1; i < ysize - 1 ; i++){
    x[i*xsize] = x[i*xsize+xsize-1] = 1.0;
    r[i*xsize + 1] -= 1.0;
    r[i*xsize + xsize - 2] -= 1.0;}  
  for(int j=1; j< xsize-1; j++){
    x[j] = x[(ysize-1)*xsize+j] = 1.0;
    r[xsize + j] -= 1.0;
    r[(ysize-2)*xsize + j] -= 1.0;}
  x[0] = x[xsize-1] = x[(ysize-1)*xsize] = x[ysize*xsize-1] = 1.0;
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
  float *u_h, *x_h, *r_h, *d_h;    // pointers to host memory
  float *u_d, *x_d, *r_d, *d_d;    // pointers to device memory
  int ArraySizeX=1026;  // Here, ArraySize minus boundary layer must be exactly divisible by BLOCK_SIZE
  int ArraySizeY=1026;
  size_t size=ArraySizeX*ArraySizeY*sizeof(float);


  // System compatibility checking ------------------------------------------------------------
  // How many GPU cards are available
  int deviceCount = 0;
  cudaError_t error_id = cudaGetDeviceCount(&deviceCount); 
  // This function call returns 0 if there are no CUDA capable devices. 
  if (deviceCount == 0){
  printf("There are no available device(s) that support CUDA\n");
  exit(EXIT_FAILURE); }
  else {
  printf("Detected %d CUDA Capable device(s)\n", deviceCount); }

  int dev;
  for (dev = 0; dev < deviceCount; ++dev) {
    cudaSetDevice(dev);
    cudaDeviceProp deviceProp; cudaGetDeviceProperties(&deviceProp, dev);
  // Thread availability per block
  if(BLOCK_SIZE_X*BLOCK_SIZE_Y > deviceProp.maxThreadsPerBlock){
    printf(" program uses %i threads, while only %d are available \n", 
    BLOCK_SIZE_X*BLOCK_SIZE_Y,deviceProp.maxThreadsPerBlock);
    exit(EXIT_FAILURE);}
  // Total device memory available
  if((size*6/1048576.0f)>deviceProp.totalGlobalMem){
    char msg[256];
    sprintf(msg, " program will need %f Mbytes of global memory when only %.0f Mbytes are available \n",
    size*6/1048576.0f,(float)deviceProp.totalGlobalMem/1048576.0f);
    printf("%s", msg);
    exit(EXIT_FAILURE);}
  }//------------------------------------------------------------------------------------------


  //Allocate arrays on host and initialize to zero
  u_h=(float *)calloc(ArraySizeX*ArraySizeY,sizeof(float));
  x_h=(float *)calloc(ArraySizeX*ArraySizeY,sizeof(float));
  d_h=(float *)calloc(ArraySizeX*ArraySizeY,sizeof(float));
  r_h=(float *)calloc(ArraySizeX*ArraySizeY,sizeof(float));

  //Initialize arrays u_h and f_h boundaries
  initialize(x_h, r_h, ArraySizeX, ArraySizeY);

  //Allocate arrays on device
  cudaMalloc((void **) &d_d,size); 
  cudaMalloc((void **) &u_d,size);
  cudaMalloc((void **) &x_d,size);
  cudaMalloc((void **) &r_d,size);

  //Perform computation on GPU
  // Part 1 of 4: Copy data from host to device
  cudaMemcpy(d_d, d_h, size, cudaMemcpyHostToDevice);
  cudaMemcpy(x_d, x_h, size, cudaMemcpyHostToDevice);
  cudaMemcpy(r_d, r_h, size, cudaMemcpyHostToDevice);
  cudaMemcpy(u_d, u_h, size, cudaMemcpyHostToDevice);

  // Initialize BLAS context handle on GPU
  cublasHandle_t handle;
  cublasCreate(&handle);

  // Part 2 of 4: Set up execution configuration
  int nBlocksX=(ArraySizeX-2)/BLOCK_SIZE_X;
  int nBlocksY=(ArraySizeY-2)/BLOCK_SIZE_Y;
  printf("nBlocksX = %d nBlocksY = %d \n",nBlocksX,nBlocksY);

  dim3 dimBlock(BLOCK_SIZE_X,BLOCK_SIZE_Y);
  dim3 dimGrid(nBlocksX,nBlocksY);

  // Fill in d
  d_fill<<<dimGrid, dimBlock>>>(d_d,r_d,ArraySizeX,ArraySizeY);   

  float delta,olddelta,lambda,alpha;
  float coef = -1.0;
  //compute first delta
  cublasSdot(handle, (ArraySizeX)*(ArraySizeY),r_d, 1,r_d, 1,&delta);
  
  for (int nsteps=1; nsteps < 20000; nsteps++) {
    // Part 3 of 4: Call kernel with execution configuration
    CGradientGPU<<<dimGrid, dimBlock>>>(d_d,u_d,ArraySizeX,ArraySizeY, 1.0);  
    lambda=0.0;
    // compute lambda
    cublasSdot(handle, (ArraySizeX)*(ArraySizeY),d_d, 1,u_d, 1,&lambda);
    lambda=delta/lambda;  
    // update r and x
    rx_update<<<dimGrid, dimBlock>>>(x_d,r_d,lambda,d_d,u_d,ArraySizeX,ArraySizeY); 
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
    cublasSaxpy(handle, (ArraySizeX)*(ArraySizeY),&coef, r_d, 1, d_d,1); 
  }

  // Part 4 of 4: Copy result from device to host
  cudaMemcpy(x_h, x_d, size, cudaMemcpyDeviceToHost);

  // Output results
  Output(x_h, ArraySizeX, ArraySizeY);

  //Cleanup
  cublasDestroy(handle);
  cudaFree(d_d);
  cudaFree(u_d);
  cudaFree(x_d);
  cudaFree(r_d);
  free(u_h);
  free(r_h);
  free(x_h);
  free(d_h);
}

