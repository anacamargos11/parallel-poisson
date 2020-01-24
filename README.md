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



If you want to take a closer look at the modeling methodology, you can read the section below:

The aim is to solve the Poisson Equation by discretising its voltage values (in the electrostatic context) with the second order centered difference equation in place of the second derivatives in space. The Conjugate Gradient method is used to find the solution of the linear system Ax=b, in which A is the coefficient matrix for the mesh and b stores the boundary values. 

In the MPI programs, the solution (x at every point) and residue (r at every point) are initialised and updated locally, so there is no need to exchange them back and forth between processes. The directional vector d will however be needed in neighbouring processes. It is first initialised locally to -r, capturing the B.C.s, and sent to the top and bottom neighbours inside the main loop as needed for the iteration of Axd at the intermediate layers of the grid (the interior is iterated previously to that). The dot products needed for the computation of variables needed for the C.G. method along the way (delta and lambda) are conducted locally and gathered via MPI Allreduce. At the end, each process writes its own data file when calling the output() function.

The 2D MPI code is a generalisation of the 1D code, but it has an added complexity. We have to send the intermediate layers of d back and forth from left to right as well in order to run the Axd iteration at the dividing locations. In order to do that, we make use of the MPI Cart shift function to define the neighbours’ locations, and define a new type MPI Datatype stridetype that allows us to send and receive chunks of columns through MPI Isend and MPI Irecv. Also, we define a new communicator mpi.D2Comm that matches our 2D topology using the MPI Cart create function and use that new communicator in our MPI calls.

The CUDA program also solves the Poisson Equation using finite-differences with the conjugate-gradient method. There are three GPU kernels in the program: CGradientGPU computes a step towards the solution in a mesh site, given the neighbouring sites’ configurations, according to the governing differential equation, d_fill fills in the device directional vector, according to the set B.C.s, and rx update is a part of the iterative scheme that improves the solution at every site. Each GPU thread will work on a given site according to the kernel’s instructions.

In the main( ) section of the CUDA program, there is a system compatibility check procedure before the rest of the program is allowed to go on. It first checks whether there are any CUDA capable devices in the system, if not, the program exits with an error message. It then checks whether the required number of threads and the required device memory meet the availability of the device.

The problem is then set up in a way that matches the GPU grid. In order for that to be possible, the array size in each direction of the grid minus the boundary layers has to be divisible by the block sizes. Each thread works independently on a mesh site, operating under the same set of instructions. The GPU kernels are called from within a CPU loop along with a series of CUBLAS routines that enable us to perform the conjugate-gradient update on a given time step. both the CUBLAS and the kernel calls are GPU operations, and therefore act upon GPU variables, that were declared with a d indexation.

The B.C.s are initialised to 1.0 at the boundary layers with the initialise( ) call. After the completion of the solution update at each step of the loop, the ultimate solution needs to be sent from the device to the host. That is accomplished with the command cudaMemcpy. At the end, the host outputs the solution vector to a data file.


Feel free to use the programs under the ./src directory.
