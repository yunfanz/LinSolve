//#include "cuSolverDn_AtA.cu"
#include "Solver_manager.hh"
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

/*
 *  solve A*x = b by QR
 *
 */
int linearSolverQR(
    cusolverDnHandle_t handle,
    int n,
    const float *Acopy,
    int lda,
    const float *b,
    float *x)
{
    cublasHandle_t cublasHandle = NULL; // used in residual evaluation
    int bufferSize = 0;
    int bufferSize_geqrf = 0;
    int bufferSize_ormqr = 0;
    int *info = NULL;
    float *buffer = NULL;
    float *A = NULL;
    float *tau = NULL;
    int h_info = 0;
    float start, stop;
    float time_solve;
    const float one = 1.0;

    checkCudaErrors(cublasCreate(&cublasHandle));

    checkCudaErrors(cusolverDnSgeqrf_bufferSize(handle, n, n, (float*)Acopy, lda, &bufferSize_geqrf));
    checkCudaErrors(cusolverDnSormqr_bufferSize(
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
    checkCudaErrors(cudaMalloc(&buffer, sizeof(float)*bufferSize));
    checkCudaErrors(cudaMalloc(&A, sizeof(float)*lda*n));
    checkCudaErrors(cudaMalloc ((void**)&tau, sizeof(float)*n));

// prepare a copy of A because getrf will overwrite A with L
    checkCudaErrors(cudaMemcpy(A, Acopy, sizeof(float)*lda*n, cudaMemcpyDeviceToDevice));

    checkCudaErrors(cudaMemset(info, 0, sizeof(int)));

    start = second();
    start = second();

// compute QR factorization
    checkCudaErrors(cusolverDnSgeqrf(handle, n, n, A, lda, tau, buffer, bufferSize, info));

    checkCudaErrors(cudaMemcpy(&h_info, info, sizeof(int), cudaMemcpyDeviceToHost));

    if ( 0 != h_info ){
        fprintf(stderr, "Error: QR factorization failed, check %d parameter\n", h_info);
    }

    checkCudaErrors(cudaMemcpy(x, b, sizeof(float)*n, cudaMemcpyDeviceToDevice));

    // compute Q^T*b
    checkCudaErrors(cusolverDnSormqr(
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
    checkCudaErrors(cublasStrsm(
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


DnSolver::DnSolver (float* array_host_, float* rhs_, int rows_, int cols_) 
{
    h_A = array_host_;
    rowsA = rows_;
    colsA = cols_;
    lda = rows_;
    h_b = rhs_;


    checkCudaErrors(cusolverDnCreate(&handle));
    checkCudaErrors(cublasCreate(&cublasHandle));
    checkCudaErrors(cudaStreamCreate(&stream));

    checkCudaErrors(cusolverDnSetStream(handle, stream));
    checkCudaErrors(cublasSetStream(cublasHandle, stream));


    checkCudaErrors(cudaMalloc((void **)&d_A, sizeof(float)*lda*colsA));
    checkCudaErrors(cudaMalloc((void **)&d_x, sizeof(float)*colsA));
    checkCudaErrors(cudaMalloc((void **)&d_b, sizeof(float)*rowsA));
    // checkCudaErrors(cudaMalloc((void **)&d_r, sizeof(float)*rowsA));
    // checkCudaErrors(cudaMalloc((void **)&d_tr, sizeof(float)*rowsA));

    printf("prepare data on device\n");
    checkCudaErrors(cudaMemcpy(d_A, h_A, sizeof(float)*lda*colsA, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_b, h_b, sizeof(float)*rowsA, cudaMemcpyHostToDevice));
}

void DnSolver::solve() {
    printf("step 6: compute AtA \n");
    cublasStatus_t cbstat;
    float al =1.0;// al =1
    float bet =0.0;// bet =0
    //float* dAcopy;
    float* dAtA;
    //checkCudaErrors(cudaMalloc(&dAcopy, sizeof(float)*lda*colsA));
    checkCudaErrors(cudaMalloc(&dAtA, sizeof(float)*colsA*colsA));
    //checkCudaErrors(cudaMemcpy(dAcopy, d_A, sizeof(float)*lda*colsA, cudaMemcpyDeviceToDevice));
    //cbstat = cublasDgemm(cublasHandle,CUBLAS_OP_T,CUBLAS_OP_N,colsA,rowsA,rowsA,&al,d_A,colsA,d_A,rowsA,&bet,dAtA,colsA);
    cbstat = cublasSgemm(cublasHandle,CUBLAS_OP_T,CUBLAS_OP_N,colsA,colsA,rowsA,&al,d_A,rowsA,d_A,rowsA,&bet,dAtA,colsA);

    printf("step 7: compute At*b \n");
    float* d_Atb;
    checkCudaErrors(cudaMalloc((void **)&d_Atb, sizeof(float)*colsA));
    cbstat = cublasSgemv(cublasHandle,CUBLAS_OP_T,colsA,colsA,&al,d_A,colsA,d_b,1,&bet,d_Atb,1);

    if (cublasHandle) { checkCudaErrors(cublasDestroy(cublasHandle)); }
    checkCudaErrors(cublasCreate(&cublasHandle));
    checkCudaErrors(cublasSetStream(cublasHandle, stream));
    printf("step 8: solves AtA*x = At*b \n");

    linearSolverQR(handle, colsA, dAtA, colsA, d_Atb, d_x);

}

void DnSolver::retrieve_to(float* h_x)
{
    checkCudaErrors(cudaMemcpy(h_x, d_x, sizeof(float)*colsA, cudaMemcpyDeviceToHost));
    printf("x0 = %E \n", h_x[0]);
}

DnSolver::~DnSolver()
{
    if (handle) { checkCudaErrors(cusolverDnDestroy(handle)); }
    if (cublasHandle) { checkCudaErrors(cublasDestroy(cublasHandle)); }
    if (stream) { checkCudaErrors(cudaStreamDestroy(stream)); }

    if (h_A) { free(h_A); }
    if (h_x) { free(h_x); }
    if (h_b) { free(h_b); }

    if (d_A) { checkCudaErrors(cudaFree(d_A)); }
    if (d_x) { checkCudaErrors(cudaFree(d_x)); }
    if (d_b) { checkCudaErrors(cudaFree(d_b)); }
}
