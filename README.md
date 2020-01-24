# parallel-poisson
Solving the Poisson Equation using parallel programming techniques

In the files from this repository, I apply the Conjugate Gradient method to solve the Poisson Equation using finite differences on a 2D mesh of points. Parallel programming procedures are applied to improve performance. 

3 different approaches were considered. All 3 approaches are capable of solving the problem. The best approach to be applied depends on the resources available and the problem size.

Approach 1: 1D Parallelization using MPI protocols
This approach divides de mesh into N grids in a 1D fashion. The code is available in C and the parallelization is implemented in MPI. Each CPU core (process) operates independently on a grid.

Approach 2: 2D Parallelization using MPI protocols
This approach divides de mesh into N^2 grids in a 2D fashion. The code is available in C and the parallelization is implemented in MPI. Each CPU core (process) operates independently on a grid.

Approach 3: Thread-like Parallelization using CUDA
This approach applies a GPU paralelization scheme on the problem mesh with CUDA. Each cell of the mesh is processed by a thread on the GPU. 

BLAS and CUBLAS routines are implemented for Linear Algebra operations.
