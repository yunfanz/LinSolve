# LinSolve
CUDA accelerated linear solver (Still a working progress)

This module solves the linear system A*x=b by finding the Moore-Penrose inverse of an n by m matrix A. 
The Moore-Penrose psuedo-inverse is the unique solution of the least square problem 
|A*x-b|. 

For the most robust backend, I provide solutions in 2 ways:

1. Direct decompositions of A by SVD, QR, LU and Cholesky, calling the respective cuSolver functions. 

2. Construct the square matrix AtA, and find the MP pseudo-inverse of AtA, with x = (AtA)^(-1)*At*b as the solution. This method has the advantage of robustness whenever AtA has full rank, even though A may not. 


API:
This module provides both direct binary API and python integration using Cython. 

