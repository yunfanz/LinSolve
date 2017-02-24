#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <assert.h>
#include <cmath> 
#include <cuda_runtime.h>
#include "SI.h"
#include "cublas_v2.h"
#include "cusolverDn.h"
#include "helper_cuda.h"

#include "helper_cusolver.h"
#include "culib_wrappers.h"
/*
 *  solve A*x = b by Cholesky factorization
 *
 */
template <typename T_ELEM>
int linearSolverCHOL(
    cusolverDnHandle_t handle,
    int n,
    const T_ELEM *Acopy,
    int lda,
    const T_ELEM *b,
    T_ELEM *x)
{
    int bufferSize = 0;
    int *info = NULL;
    T_ELEM *buffer = NULL;
    T_ELEM *A = NULL;
    int h_info = 0;
    T_ELEM start, stop;
    T_ELEM time_solve;
    cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;

    checkCudaErrors(cusolverDnpotrf_bufferSize(handle, uplo, n, (T_ELEM*)Acopy, lda, &bufferSize));

    checkCudaErrors(cudaMalloc(&info, sizeof(int)));
    checkCudaErrors(cudaMalloc(&buffer, sizeof(T_ELEM)*bufferSize));
    checkCudaErrors(cudaMalloc(&A, sizeof(T_ELEM)*lda*n));


    // prepare a copy of A because potrf will overwrite A with L
    checkCudaErrors(cudaMemcpy(A, Acopy, sizeof(T_ELEM)*lda*n, cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaMemset(info, 0, sizeof(int)));

    start = second();
    start = second();

    checkCudaErrors(cusolverDnpotrf(handle, uplo, n, A, lda, buffer, bufferSize, info));

    checkCudaErrors(cudaMemcpy(&h_info, info, sizeof(int), cudaMemcpyDeviceToHost));

    if ( 0 != h_info ){
        fprintf(stderr, "Error: Cholesky factorization failed\n");
    }

    checkCudaErrors(cudaMemcpy(x, b, sizeof(T_ELEM)*n, cudaMemcpyDeviceToDevice));

    checkCudaErrors(cusolverDnpotrs(handle, uplo, n, 1, A, lda, x, n, info));

    checkCudaErrors(cudaDeviceSynchronize());
    stop = second();

    time_solve = stop - start;
    fprintf (stdout, "timing: cholesky = %10.6f sec\n", time_solve);

    if (info  ) { checkCudaErrors(cudaFree(info)); }
    if (buffer) { checkCudaErrors(cudaFree(buffer)); }
    if (A     ) { checkCudaErrors(cudaFree(A)); }

    return 0;
}


/*
 *  solve A*x = b by LU with partial pivoting
 *
 */
template <typename T_ELEM>
int linearSolverLU(
    cusolverDnHandle_t handle,
    int n,
    const T_ELEM *Acopy,
    int lda,
    const T_ELEM *b,
    T_ELEM *x)
{
    int bufferSize = 0;
    int *info = NULL;
    T_ELEM *buffer = NULL;
    T_ELEM *A = NULL;
    int *ipiv = NULL; // pivoting sequence
    int h_info = 0;
    T_ELEM start, stop;
    T_ELEM time_solve;

    checkCudaErrors(cusolverDngetrf_bufferSize(handle, n, n, (T_ELEM*)Acopy, lda, &bufferSize));

    checkCudaErrors(cudaMalloc(&info, sizeof(int)));
    checkCudaErrors(cudaMalloc(&buffer, sizeof(T_ELEM)*bufferSize));
    checkCudaErrors(cudaMalloc(&A, sizeof(T_ELEM)*lda*n));
    checkCudaErrors(cudaMalloc(&ipiv, sizeof(int)*n));


    // prepare a copy of A because getrf will overwrite A with L
    checkCudaErrors(cudaMemcpy(A, Acopy, sizeof(T_ELEM)*lda*n, cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaMemset(info, 0, sizeof(int)));

    start = second();
    start = second();

    checkCudaErrors(cusolverDngetrf(handle, n, n, A, lda, buffer, ipiv, info));
    checkCudaErrors(cudaMemcpy(&h_info, info, sizeof(int), cudaMemcpyDeviceToHost));

    if ( 0 != h_info ){
        fprintf(stderr, "Error: LU factorization failed\n");
    }

    checkCudaErrors(cudaMemcpy(x, b, sizeof(T_ELEM)*n, cudaMemcpyDeviceToDevice));
    checkCudaErrors(cusolverDngetrs(handle, CUBLAS_OP_N, n, 1, A, lda, ipiv, x, n, info));
    checkCudaErrors(cudaDeviceSynchronize());
    stop = second();

    time_solve = stop - start;
    fprintf (stdout, "timing: LU = %10.6f sec\n", time_solve);

    if (info  ) { checkCudaErrors(cudaFree(info  )); }
    if (buffer) { checkCudaErrors(cudaFree(buffer)); }
    if (A     ) { checkCudaErrors(cudaFree(A)); }
    if (ipiv  ) { checkCudaErrors(cudaFree(ipiv));}

    return 0;
}


/*
 *  solve A*x = b by QR
 *
 */
template <typename T_ELEM>
int linearSolverQR(
    cusolverDnHandle_t handle,
    int n,
    const T_ELEM *Acopy,
    int lda,
    const T_ELEM *b,
    T_ELEM *x)
{
    cublasHandle_t cublasHandle = NULL; // used in residual evaluation
    int bufferSize = 0;
    int bufferSize_geqrf = 0;
    int bufferSize_ormqr = 0;
    int *info = NULL;
    T_ELEM *buffer = NULL;
    T_ELEM *A = NULL;
    T_ELEM *tau = NULL;
    int h_info = 0;
    T_ELEM start, stop;
    T_ELEM time_solve;
    const T_ELEM one = 1.0;

    checkCudaErrors(cublasCreate(&cublasHandle));

    checkCudaErrors(cusolverDngeqrf_bufferSize(handle, n, n, (T_ELEM*)Acopy, lda, &bufferSize_geqrf));
    checkCudaErrors(cusolverDnormqr_bufferSize(
        handle,
        CUBLAS_SIDE_LEFT,
        CUBLAS_OP_T,
        n,
        1,
        n,
        A,
        lda,
        NULL,
        x,
        n,
        &bufferSize_ormqr));

    printf("buffer_geqrf = %d, buffer_ormqr = %d \n", bufferSize_geqrf, bufferSize_ormqr);
    
    bufferSize = (bufferSize_geqrf > bufferSize_ormqr)? bufferSize_geqrf : bufferSize_ormqr ; 

    checkCudaErrors(cudaMalloc(&info, sizeof(int)));
    checkCudaErrors(cudaMalloc(&buffer, sizeof(T_ELEM)*bufferSize));
    checkCudaErrors(cudaMalloc(&A, sizeof(T_ELEM)*lda*n));
    checkCudaErrors(cudaMalloc ((void**)&tau, sizeof(T_ELEM)*n));

// prepare a copy of A because getrf will overwrite A with L
    checkCudaErrors(cudaMemcpy(A, Acopy, sizeof(T_ELEM)*lda*n, cudaMemcpyDeviceToDevice));

    checkCudaErrors(cudaMemset(info, 0, sizeof(int)));

    start = second();
    start = second();

// compute QR factorization
    checkCudaErrors(cusolverDngeqrf(handle, n, n, A, lda, tau, buffer, bufferSize, info));

    checkCudaErrors(cudaMemcpy(&h_info, info, sizeof(int), cudaMemcpyDeviceToHost));

    if ( 0 != h_info ){
        fprintf(stderr, "Error: LU factorization failed\n");
    }

    checkCudaErrors(cudaMemcpy(x, b, sizeof(T_ELEM)*n, cudaMemcpyDeviceToDevice));

    // compute Q^T*b
    checkCudaErrors(cusolverDnormqr(
        handle,
        CUBLAS_SIDE_LEFT,
        CUBLAS_OP_T,
        n,
        1,
        n,
        A,
        lda,
        tau,
        x,
        n,
        buffer,
        bufferSize,
        info));

    // x = R \ Q^T*b
    checkCudaErrors(cublastrsm(
         cublasHandle,
         CUBLAS_SIDE_LEFT,
         CUBLAS_FILL_MODE_UPPER,
         CUBLAS_OP_N,
         CUBLAS_DIAG_NON_UNIT,
         n,
         1,
         &one,
         A,
         lda,
         x,
         n));
    checkCudaErrors(cudaDeviceSynchronize());
    stop = second();

    time_solve = stop - start;
    fprintf (stdout, "timing: QR = %10.6f sec\n", time_solve);

    if (cublasHandle) { checkCudaErrors(cublasDestroy(cublasHandle)); }
    if (info  ) { checkCudaErrors(cudaFree(info  )); }
    if (buffer) { checkCudaErrors(cudaFree(buffer)); }
    if (A     ) { checkCudaErrors(cudaFree(A)); }
    if (tau   ) { checkCudaErrors(cudaFree(tau)); }

    return 0;
}


template <typename T_ELEM>
int linearSolverSVD(
    cusolverDnHandle_t handle, 
    const int m, 
    const int n,
    const T_ELEM *Acopy,
    const int lda,
    const T_ELEM *bcopy,
    T_ELEM *x)
{
    cublasHandle_t cublasHandle = NULL; // used in residual evaluation
    int maxInd = (n<m) ? n : m;
    int bufferSize = 0;
    int *info = NULL;
    int h_info = 0;
    T_ELEM start, stop;
    T_ELEM time_solve;
    const T_ELEM one = 1.0;
    printf("m %d, n %d, lda %d", m, n, lda);

    // T_ELEM U[lda*m]; // m-by-m unitary matrix 
    // T_ELEM VT[lda*n]; // n-by-n unitary matrix
    // T_ELEM S[n]; //singular value 
    T_ELEM *d_A = NULL; T_ELEM *d_SI = NULL; 
    T_ELEM *d_b = NULL; T_ELEM *d_S = NULL; 
    T_ELEM *d_U = NULL; T_ELEM *d_VT = NULL; 
    T_ELEM *d_work = NULL; 
    T_ELEM *d_rwork = NULL; 
    T_ELEM *d_W = NULL; 
    signed char jobu = 'A'; // all m columns of U 
    signed char jobvt = 'A'; // all n columns of VT 
    // step 1: create cusolverDn/cublas handle 
    checkCudaErrors(cublasCreate(&cublasHandle)); 

    checkCudaErrors(cudaMalloc((void**)&d_A , sizeof(T_ELEM)*lda*n));
    checkCudaErrors(cudaMalloc((void**)&d_b , sizeof(T_ELEM)*m)); 
    checkCudaErrors(cudaMalloc((void**)&d_S , sizeof(T_ELEM)*maxInd)); 
    checkCudaErrors(cudaMalloc((void**)&d_SI , sizeof(T_ELEM)*lda*n)); 
    checkCudaErrors(cudaMalloc((void**)&d_U , sizeof(T_ELEM)*lda*m)); 
    checkCudaErrors(cudaMalloc((void**)&d_VT , sizeof(T_ELEM)*n*n)); 
    checkCudaErrors(cudaMalloc((void**)&info, sizeof(int))); 
    checkCudaErrors(cudaMalloc((void**)&d_W , sizeof(T_ELEM)*lda*n));
    checkCudaErrors(cudaMemcpy(d_A, Acopy, sizeof(T_ELEM)*lda*n, cudaMemcpyDeviceToDevice)); //gesvd destroys d_A on exit
    checkCudaErrors(cudaMemcpy(d_b, bcopy, sizeof(T_ELEM)*m, cudaMemcpyDeviceToDevice));

    // checkMatrix(m, n , d_A, lda, "SVD_A");
    // checkArray(d_b, m, "SVD_Atb");
    checkCudaErrors(cusolverDngesvd_bufferSize( handle, m, n, &bufferSize ));
    checkCudaErrors(cudaMalloc((void**)&d_work , sizeof(T_ELEM)*bufferSize));

    start = second();

    checkCudaErrors(cusolverDngesvd( 
        handle, jobu, jobvt, m, n, d_A, lda, d_S, d_U, lda, d_VT, n, d_work, bufferSize, d_rwork, info));
    //checkCudaErrors(cudaDeviceSynchronize());
    // checkArray(d_S, n, "dS");
    // checkCudaErrors(cudaMemcpy(U , d_U , sizeof(T_ELEM)*lda*m, cudaMemcpyDeviceToHost)); 
    // checkCudaErrors(cudaMemcpy(VT, d_VT, sizeof(T_ELEM)*lda*n, cudaMemcpyDeviceToHost)); 
    // checkCudaErrors(cudaMemcpy(S , d_S , sizeof(T_ELEM)*n , cudaMemcpyDeviceToHost)); 
    checkCudaErrors(cudaMemcpy(&h_info, info, sizeof(int), cudaMemcpyDeviceToHost));

    if ( 0 != h_info ){
        fprintf(stderr, "Error: SVD failed, check %d parameter\n", h_info);
    }

    // int BLOCK_DIM_X = 32; int BLOCK_DIM_Y = 32;
    // dim3 blockDim(BLOCK_DIM_X, BLOCK_DIM_Y);  
    // dim3 gridDim((n + BLOCK_DIM_X - 1) / BLOCK_DIM_X, (m + BLOCK_DIM_Y - 1) / BLOCK_DIM_Y);
    // initSIGPU<<<gridDim, blockDim>>>(d_SI, d_S, m, n);
    T_ELEM epsilon = 1.e-9;
    printf("epsilon = %f \n", epsilon);
    // Launch the SI CUDA Kernel
    //int initStat = initSICPU(d_SI, d_S, m, n, epsilon);
    initSI(d_SI, d_S, m, n, epsilon, 256);

    // U*S*VT*x=b; x = V*Si*UT*b
    // checkMatrix(m, n, d_SI, lda, "SVD_SI");
    // checkMatrix(m, m, d_U, lda, "SVD_U");
    // checkMatrix(n, n, d_VT, n, "SVD_VT");
    T_ELEM al = 1.0;// al =1
    T_ELEM bet = 0.0;// bet =0
    // checkArray(d_b, m, "db");
    checkCudaErrors(cublasgemv(cublasHandle,CUBLAS_OP_T, m, m, &al,d_U, m, d_b,1,&bet,d_b,1));
    // checkArray(d_b, m, "Utb");
    checkCudaErrors(cublasgemv(cublasHandle,CUBLAS_OP_T, n, m, &al,d_SI, m, d_b,1,&bet,d_b,1));
    // checkArray(d_b, n, "dSiUtb");
    checkCudaErrors(cublasgemv(cublasHandle,CUBLAS_OP_T, n, n, &al,d_VT, n, d_b, 1,&bet,x,1));
    checkCudaErrors(cudaDeviceSynchronize());
    stop = second();
    time_solve = stop - start; 
    fprintf (stdout, "timing: SVD = %10.6f sec\n", time_solve);
    // checkArray(x, n, "d_x");

    if (d_A ) cudaFree(d_A); 
    if (d_S ) cudaFree(d_S); 
    if (d_SI ) cudaFree(d_SI);
    if (d_U ) cudaFree(d_U); 
    if (d_VT ) cudaFree(d_VT); 
    if (info) cudaFree(info); 
    if (d_work ) cudaFree(d_work); 
    if (d_rwork) cudaFree(d_rwork); 
    if (d_W ) cudaFree(d_W); 
    if (cublasHandle ) cublasDestroy(cublasHandle); 
    // if (cusolverH) cusolverDnDestroy(cusolverH); 
    return 0;


}

// template
// int linearSolverCHOL<float>(
//     cusolverDnHandle_t handle,
//     int n,
//     const float *Acopy,
//     int lda,
//     const float *b,
//     float *x);
// template
// int linearSolverCHOL<double>(
//     cusolverDnHandle_t handle,
//     int n,
//     const double *Acopy,
//     int lda,
//     const double *b,
//     double *x);

// template
// int linearSolverLU<float>(
//     cusolverDnHandle_t handle,
//     int n,
//     const float *Acopy,
//     int lda,
//     const float *b,
//     float *x);
// template
// int linearSolverLU<double>(
//     cusolverDnHandle_t handle,
//     int n,
//     const double *Acopy,
//     int lda,
//     const double *b,
//     double *x);

// template
// int linearSolverQR<float>(
//     cusolverDnHandle_t handle,
//     int n,
//     const float *Acopy,
//     int lda,
//     const float *b,
//     float *x);
// template
// int linearSolverQR<double>(
//     cusolverDnHandle_t handle,
//     int n,
//     const double *Acopy,
//     int lda,
//     const double *b,
//     double *x);

// template
// int linearSolverSVD<float>(
//     cusolverDnHandle_t handle, 
//     const int m, 
//     const int n,
//     const float *Acopy,
//     const int lda,
//     const float *bcopy,
//     float *x);
// template
// int linearSolverSVD<double>(
//     cusolverDnHandle_t handle, 
//     const int m, 
//     const int n,
//     const double *Acopy,
//     const int lda,
//     const double *bcopy,
//     double *x);