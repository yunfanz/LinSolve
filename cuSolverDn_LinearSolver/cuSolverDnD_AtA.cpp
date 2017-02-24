/*
 * Copyright 2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/*
 *  Test three linear solvers, including Cholesky, LU and QR.
 *  The user has to prepare a sparse matrix of "matrix market format" (with extension .mtx).
 *  For example, the user can download matrices in Florida Sparse Matrix Collection.
 *  (http://www.cise.ufl.edu/research/sparse/matrices/)
 *
 *  The user needs to choose a solver by switch -R<solver> and
 *  to provide the path of the matrix by switch -F<file>, then
 *  the program solves
 *          A*x = b  where b = ones(m,1)
 *  and reports relative error
 *          |b-A*x|/(|A|*|x|)
 *
 *  The elapsed time is also reported so the user can compare efficiency of different solvers.
 *
 *  How to use
 *      ./cuSolverDn_LinearSolver                     // Default: cholesky
 *     ./cuSolverDn_LinearSolver -R=chol -filefile>   // cholesky factorization
 *     ./cuSolverDn_LinearSolver -R=lu -file<file>     // LU with partial pivoting
 *     ./cuSolverDn_LinearSolver -R=qr -file<file>     // QR factorization
 *
 *  Remark: the absolute error on solution x is meaningless without knowing condition number of A.
 *     The relative error on residual should be close to machine zero, i.e. 1.e-15.
 */


/* Modified by Yunfan Gerry Zhang (UC Berkeley Department of Astronomy)
 * Solves the A*x=b system by x=inv(At*A)*At*b for k by n matrix A in csr format
 * CuBLAS functions getrs, potrs computes x = Ai*b, so:
 * if b is provided, compute rs(rf(At*A), At*b)
 * if b is not provided, compute rs(rf(At*A), At)
 * TODO: implement cusolverDnSgesvd on AtA
 */

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


template <typename T_ELEM>
int loadMMSparseMatrix(
    char *filename,
    char elem_type,
    bool csrFormat,
    int *m,
    int *n,
    int *nnz,
    T_ELEM **aVal,
    int **aRowInd,
    int **aColInd,
    int extendSymMatrix);

void UsageDN(void)
{
    printf( "<options>\n");
    printf( "-h          : display this help\n");
    printf( "-R=<name>    : choose a linear solver\n");
    printf( "              chol (cholesky factorization), this is default\n");
    printf( "              qr   (QR factorization)\n");
    printf( "              lu   (LU factorization)\n");
    printf( "-lda=<int> : leading dimension of A , m by default\n");
    printf( "-file=<filename>: filename containing a matrix in MM format\n");
    printf( "-device=<device_id> : <device_id> if want to run on specific GPU\n");

    exit( 0 );
}
/*
 *  solve A*x = b by Cholesky factorization
 *
 */
int linearSolverCHOL(
    cusolverDnHandle_t handle,
    int n,
    const double *Acopy,
    int lda,
    const double *b,
    double *x)
{
    int bufferSize = 0;
    int *info = NULL;
    double *buffer = NULL;
    double *A = NULL;
    int h_info = 0;
    double start, stop;
    double time_solve;
    cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;

    checkCudaErrors(cusolverDnDpotrf_bufferSize(handle, uplo, n, (double*)Acopy, lda, &bufferSize));

    checkCudaErrors(cudaMalloc(&info, sizeof(int)));
    checkCudaErrors(cudaMalloc(&buffer, sizeof(double)*bufferSize));
    checkCudaErrors(cudaMalloc(&A, sizeof(double)*lda*n));


    // prepare a copy of A because potrf will overwrite A with L
    checkCudaErrors(cudaMemcpy(A, Acopy, sizeof(double)*lda*n, cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaMemset(info, 0, sizeof(int)));

    start = second();
    start = second();

    checkCudaErrors(cusolverDnDpotrf(handle, uplo, n, A, lda, buffer, bufferSize, info));

    checkCudaErrors(cudaMemcpy(&h_info, info, sizeof(int), cudaMemcpyDeviceToHost));

    if ( 0 != h_info ){
        fprintf(stderr, "Error: Cholesky factorization failed, check %d parameter\n", h_info);
    }

    checkCudaErrors(cudaMemcpy(x, b, sizeof(double)*n, cudaMemcpyDeviceToDevice));

    checkCudaErrors(cusolverDnDpotrs(handle, uplo, n, 1, A, lda, x, n, info));

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
int linearSolverLU(
    cusolverDnHandle_t handle,
    int n,
    const double *Acopy,
    int lda,
    const double *b,
    double *x)
{
    int bufferSize = 0;
    int *info = NULL;
    double *buffer = NULL;
    double *A = NULL;
    int *ipiv = NULL; // pivoting sequence
    int h_info = 0;
    double start, stop;
    double time_solve;

    checkCudaErrors(cusolverDnDgetrf_bufferSize(handle, n, n, (double*)Acopy, lda, &bufferSize));

    checkCudaErrors(cudaMalloc(&info, sizeof(int)));
    checkCudaErrors(cudaMalloc(&buffer, sizeof(double)*bufferSize));
    checkCudaErrors(cudaMalloc(&A, sizeof(double)*lda*n));
    checkCudaErrors(cudaMalloc(&ipiv, sizeof(int)*n));


    // prepare a copy of A because getrf will overwrite A with L
    checkCudaErrors(cudaMemcpy(A, Acopy, sizeof(double)*lda*n, cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaMemset(info, 0, sizeof(int)));

    start = second();
    start = second();

    checkCudaErrors(cusolverDnDgetrf(handle, n, n, A, lda, buffer, ipiv, info));
    checkCudaErrors(cudaMemcpy(&h_info, info, sizeof(int), cudaMemcpyDeviceToHost));

    if ( 0 != h_info ){
        fprintf(stderr, "Error: LU factorization failed, check %d parameter\n", h_info);
    }

    checkCudaErrors(cudaMemcpy(x, b, sizeof(double)*n, cudaMemcpyDeviceToDevice));
    checkCudaErrors(cusolverDnDgetrs(handle, CUBLAS_OP_N, n, 1, A, lda, ipiv, x, n, info));
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


int linearSolverSVD(
    cusolverDnHandle_t handle, 
    int n,
    const double *Acopy,
    int lda,
    const double *bcopy,
    double *x)
{
    cublasHandle_t cublasHandle = NULL; // used in residual evaluation
    int m = lda;
    int bufferSize = 0;
    int *info = NULL;
    int h_info = 0;
    double start, stop;
    double time_solve;
    const double one = 1.0;

    // double U[lda*m]; // m-by-m unitary matrix 
    // double VT[lda*n]; // n-by-n unitary matrix
    // double S[n]; //singular value 
    double *d_A = NULL; double *d_SI = NULL; 
    double *d_b = NULL; double *d_S = NULL; 
    double *d_U = NULL; double *d_VT = NULL; 
    double *d_work = NULL; 
    double *d_rwork = NULL; 
    double *d_W = NULL; 
    signed char jobu = 'A'; // all m columns of U 
    signed char jobvt = 'A'; // all n columns of VT 
    // step 1: create cusolverDn/cublas handle 
    checkCudaErrors(cublasCreate(&cublasHandle)); 

    checkCudaErrors(cudaMalloc((void**)&d_A , sizeof(double)*lda*n)); \
    checkCudaErrors(cudaMalloc((void**)&d_b , sizeof(double)*m)); 
    checkCudaErrors(cudaMalloc((void**)&d_S , sizeof(double)*n)); 
    checkCudaErrors(cudaMalloc((void**)&d_SI , sizeof(double)*lda*n)); 
    checkCudaErrors(cudaMalloc((void**)&d_U , sizeof(double)*lda*m)); 
    checkCudaErrors(cudaMalloc((void**)&d_VT , sizeof(double)*lda*n)); 
    checkCudaErrors(cudaMalloc((void**)&info, sizeof(int))); 
    checkCudaErrors(cudaMalloc((void**)&d_W , sizeof(double)*lda*n));
    checkCudaErrors(cudaMemcpy(d_A, Acopy, sizeof(double)*lda*n, cudaMemcpyDeviceToDevice)); //gesvd destroys d_A on exit
    checkCudaErrors(cudaMemcpy(d_b, bcopy, sizeof(double)*m, cudaMemcpyDeviceToDevice));
    
    // checkMatrix(m, n, d_SI, lda, "zero_SI");
    // checkMatrix(m, n , d_A, lda, "SVD_AtA");
    // checkArray(d_b, m, "SVD_Atb");
    checkCudaErrors(cusolverDnDgesvd_bufferSize( handle, m, n, &bufferSize ));
    checkCudaErrors(cudaMalloc((void**)&d_work , sizeof(double)*bufferSize));

    start = second();

    checkCudaErrors(cusolverDnDgesvd( 
        handle, jobu, jobvt, m, n, d_A, lda, d_S, d_U, lda, d_VT, lda, d_work, bufferSize, d_rwork, info));
    //checkCudaErrors(cudaDeviceSynchronize());
    
    // checkCudaErrors(cudaMemcpy(U , d_U , sizeof(double)*lda*m, cudaMemcpyDeviceToHost)); 
    // checkCudaErrors(cudaMemcpy(VT, d_VT, sizeof(double)*lda*n, cudaMemcpyDeviceToHost)); 
    // checkCudaErrors(cudaMemcpy(S , d_S , sizeof(double)*n , cudaMemcpyDeviceToHost)); 
    checkCudaErrors(cudaMemcpy(&h_info, info, sizeof(int), cudaMemcpyDeviceToHost));

    if ( 0 != h_info ){
        fprintf(stderr, "Error: SVD failed, check %d parameter\n", h_info);
    }

    // int BLOCK_DIM_X = 32; int BLOCK_DIM_Y = 32;
    // dim3 blockDim(BLOCK_DIM_X, BLOCK_DIM_Y);  
    // dim3 gridDim((n + BLOCK_DIM_X - 1) / BLOCK_DIM_X, (m + BLOCK_DIM_Y - 1) / BLOCK_DIM_Y);
    // initSIGPU<<<gridDim, blockDim>>>(d_SI, d_S, m, n);
    double epsilon = 1.e-9;
    printf("epsilon = %f \n", epsilon);
    initSI(d_SI, d_S, m, n, epsilon, 256);
    //int initStat = initSICPU(d_SI, d_S, m, n, epsilon);
    // U*S*V*x=b; x = VT*Si*UT*b
    // checkMatrix(m, n, d_SI, lda, "SVD_SI");
    // checkArray(d_S, n, "dS");
    // checkMatrix(m, m, d_U, lda, "SVD_U");
    // checkMatrix(n, n, d_VT, lda, "SVD_VT");
    double al = 1.0;// al =1
    double bet = 0.0;// bet =0
    // checkArray(d_b, n, "db");
    checkCudaErrors(cublasDgemv(cublasHandle,CUBLAS_OP_T, m, m, &al,d_U, m, d_b,1,&bet,d_b,1));
    // checkArray(d_b, n, "dUtb");
    checkCudaErrors(cublasDgemv(cublasHandle,CUBLAS_OP_N, m, n, &al,d_SI, m, d_b,1,&bet,d_b,1));
    // checkArray(d_b, n, "dSiUtb");
    checkCudaErrors(cublasDgemv(cublasHandle,CUBLAS_OP_T, n, n, &al,d_VT, n, d_b, 1,&bet,x,1));
    checkCudaErrors(cudaDeviceSynchronize());
    stop = second();
    time_solve = stop - start; 
    fprintf (stdout, "timing: SVD = %10.6f sec\n", time_solve);
    // checkArray(x, 20, "d_x");

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
/*
 *  solve A*x = b by QR
 *
 */
int linearSolverQR(
    cusolverDnHandle_t handle,
    int n,
    const double *Acopy,
    int lda,
    const double *b,
    double *x)
{
    cublasHandle_t cublasHandle = NULL; // used in residual evaluation
    int bufferSize = 0;
    int bufferSize_geqrf = 0;
    int bufferSize_ormqr = 0;
    int *info = NULL;
    double *buffer = NULL;
    double *A = NULL;
    double *tau = NULL;
    int h_info = 0;
    double start, stop;
    double time_solve;
    const double one = 1.0;

    checkCudaErrors(cublasCreate(&cublasHandle));

    checkCudaErrors(cusolverDnDgeqrf_bufferSize(handle, n, n, (double*)Acopy, lda, &bufferSize_geqrf));
    checkCudaErrors(cusolverDnDormqr_bufferSize(
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
    checkCudaErrors(cudaMalloc(&buffer, sizeof(double)*bufferSize));
    checkCudaErrors(cudaMalloc(&A, sizeof(double)*lda*n));
    checkCudaErrors(cudaMalloc ((void**)&tau, sizeof(double)*n));

// prepare a copy of A because getrf will overwrite A with L
    checkCudaErrors(cudaMemcpy(A, Acopy, sizeof(double)*lda*n, cudaMemcpyDeviceToDevice));

    checkCudaErrors(cudaMemset(info, 0, sizeof(int)));

    start = second();

    // BENCHMARKING: 
    // for (int i=0; i< 400; i++) {
    //         // compute QR factorization
    //     checkCudaErrors(cusolverDnDgeqrf(handle, n, n, A, lda, tau, buffer, bufferSize, info));

    //     checkCudaErrors(cudaMemcpy(&h_info, info, sizeof(int), cudaMemcpyDeviceToHost));

    //     if ( 0 != h_info ){
    //         fprintf(stderr, "Error: QR factorization failed, check %d parameter\n", h_info);
    //     }

    //     checkCudaErrors(cudaMemcpy(x, b, sizeof(double)*n, cudaMemcpyDeviceToDevice));

    //     // compute Q^T*b
    //     checkCudaErrors(cusolverDnDormqr(
    //         handle,
    //         CUBLAS_SIDE_LEFT,
    //         CUBLAS_OP_T,
    //         n,
    //         1,
    //         n,
    //         A,
    //         lda,
    //         tau,
    //         x,
    //         n,
    //         buffer,
    //         bufferSize,
    //         info));

    //     // x = R \ Q^T*b
    //     checkCudaErrors(cublasDtrsm(
    //          cublasHandle,
    //          CUBLAS_SIDE_LEFT,
    //          CUBLAS_FILL_MODE_UPPER,
    //          CUBLAS_OP_N,
    //          CUBLAS_DIAG_NON_UNIT,
    //          n,
    //          1,
    //          &one,
    //          A,
    //          lda,
    //          x,
    //          n));
    //     checkCudaErrors(cudaDeviceSynchronize());
    // }
// compute QR factorization
    checkCudaErrors(cusolverDnDgeqrf(handle, n, n, A, lda, tau, buffer, bufferSize, info));

    checkCudaErrors(cudaMemcpy(&h_info, info, sizeof(int), cudaMemcpyDeviceToHost));

    if ( 0 != h_info ){
        fprintf(stderr, "Error: QR factorization failed, check %d parameter\n", h_info);
    }

    checkCudaErrors(cudaMemcpy(x, b, sizeof(double)*n, cudaMemcpyDeviceToDevice));

    // compute Q^T*b
    checkCudaErrors(cusolverDnDormqr(
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
    checkCudaErrors(cublasDtrsm(
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


void parseCommandLineArguments(int argc, char *argv[], struct testOpts &opts)
{
    memset(&opts, 0, sizeof(opts));

    if (checkCmdLineFlag(argc, (const char **)argv, "-h"))
    {
        UsageDN();
    }

    if (checkCmdLineFlag(argc, (const char **)argv, "R"))
    {
        char *solverType = NULL;
        getCmdLineArgumentString(argc, (const char **)argv, "R", &solverType);

        if (solverType)
        {
            if ((STRCASECMP(solverType, "svd") != 0) && (STRCASECMP(solverType, "chol") != 0) && (STRCASECMP(solverType, "lu") != 0) && (STRCASECMP(solverType, "qr") != 0))
            {
                printf("\nIncorrect argument passed to -R option\n");
                UsageDN();
            }
            else
            {
                opts.testFunc = solverType;
            }
        }
    }

    if (checkCmdLineFlag(argc, (const char **)argv, "file"))
    {
        char *fileName = 0;
        getCmdLineArgumentString(argc, (const char **)argv, "file", &fileName);

        if (fileName)
        {
            opts.sparse_mat_filename = fileName;
        }
        else
        {
            printf("\nIncorrect filename passed to -file \n ");
            UsageDN();
        }
    }

    if (checkCmdLineFlag(argc, (const char **)argv, "lda"))
    {
        opts.lda = getCmdLineArgumentInt(argc, (const char **)argv, "lda");
    }
}

int is_symmetric(double* h_A, int colsA, int rowsA, int lda)
{
    int issym = 1;
    for(int j = 0 ; j < colsA ; j++)
    {
        for(int i = j ; i < rowsA ; i++)
        {
            double Aij = h_A[i + j*lda];
            double Aji = h_A[j + i*lda];
            if ( Aij != Aji )
            {
                issym = 0;
                break;
            }
        }
    }
    if (!issym)
    {
        printf("A has no symmetric pattern, please use LU or QR \n");
        exit(EXIT_FAILURE);
    } 
    return issym;
}

int main (int argc, char *argv[])
{
    struct testOpts opts;
    cusolverDnHandle_t handle = NULL;
    cublasHandle_t cublasHandle = NULL; // used in residual evaluation
    cudaStream_t stream = NULL;

    int rowsA = 0; // number of rows of A
    int colsA = 0; // number of columns of A
    int nnzA  = 0; // number of nonzeros of A
    int baseA = 0; // base index in CSR format
    int lda   = 0; // leading dimension in dense matrix

    // CSR(A) from I/O
    int *h_csrRowPtrA = NULL;
    int *h_csrColIndA = NULL;
    double *h_csrValA = NULL;
    // CSC(A) from I/O
    // int *h_cscColPtrA = NULL;
    // int *h_cscRowIndA = NULL;
    // double *h_cscValA = NULL;

    double *h_A = NULL; // dense matrix from CSR(A)
    double *h_x = NULL; // a copy of d_x
    double *h_b = NULL; // b = ones(m,1)
    double *h_r = NULL; // r = b - A*x, a copy of d_r
    double *h_tr = NULL;

    double *d_A = NULL; // a copy of h_A
    double *d_x = NULL; // x = A \ b
    double *d_b = NULL; // a copy of h_b
    double *d_r = NULL; // r = b - A*x
    double *d_tr = NULL; // tr = Atb - AtA*x

    // the constants are used in residual evaluation, r = b - A*x
    const double minus_one = -1.0;
    const double one = 1.0;

    double x_inf = 0.0;
    double r_inf = 0.0;
    double A_inf = 0.0;
    double b_inf = 0.0;
    double Ax_inf = 0.0;
    double tr_inf = 0.0;
    int errors = 0;

    parseCommandLineArguments(argc, argv, opts);

    if (NULL == opts.testFunc)
    {
        opts.testFunc = "chol"; // By default running Cholesky as NO solver selected with -R option.
    }

    findCudaDevice(argc, (const char **)argv);

    printf("step 1: read matrix market format\n");

    if (opts.sparse_mat_filename == NULL)
    {
        opts.sparse_mat_filename =  sdkFindFilePath("gr_900_900_crg.mtx", argv[0]);
        if (opts.sparse_mat_filename != NULL)
            printf("Using default input file [%s]\n", opts.sparse_mat_filename);
        else
            printf("Could not find gr_900_900_crg.mtx\n");
    }
    else
    {
        printf("Using input file [%s]\n", opts.sparse_mat_filename);
    }

    if (opts.sparse_mat_filename == NULL)
    {
        fprintf(stderr, "Error: input matrix is not provided\n");
        return EXIT_FAILURE;
    }

    if (loadMMSparseMatrix<double>(opts.sparse_mat_filename, 'd', true , &rowsA, &colsA,
               &nnzA, &h_csrValA, &h_csrRowPtrA, &h_csrColIndA, true))
    {
        exit(EXIT_FAILURE);
    }
    baseA = h_csrRowPtrA[0]; // baseA = {0,1}

    // if (loadMMSparseMatrix<double>(opts.sparse_mat_filename, 'd', true , &rowsA, &colsA,
    //            &nnzA, &h_cscValA, &h_cscColPtrA, &h_cscRowIndA,  true))
    // {
    //     exit(EXIT_FAILURE);
    // }
    // baseA = h_cscColPtrA[0]; // baseA = {0,1}

    printf("sparse matrix A is %d x %d with %d nonzeros, base=%d\n", rowsA, colsA, nnzA, baseA);

    // if ( rowsA != colsA )
    // {
    //     fprintf(stderr, "Error: only support square matrix\n");
    //     exit(EXIT_FAILURE);
    // }

    printf("step 2: convert CSR(A) to dense matrix\n");

    lda = opts.lda ? opts.lda : rowsA;
    if (lda < rowsA)
    {
        fprintf(stderr, "Error: lda must be greater or equal to dimension of A\n");
        exit(EXIT_FAILURE);
    }
    //printf("%d\n",lda);

    h_A = (double*)malloc(sizeof(double)*lda*colsA);
    h_x = (double*)malloc(sizeof(double)*colsA);
    h_b = (double*)malloc(sizeof(double)*rowsA);
    h_r = (double*)malloc(sizeof(double)*rowsA);
    h_tr = (double*)malloc(sizeof(double)*colsA);
    assert(NULL != h_A);
    assert(NULL != h_x);
    assert(NULL != h_b);
    assert(NULL != h_r);

    memset(h_A, 0, sizeof(double)*lda*colsA);

    // for(int row = 0 ; row < rowsA ; row++)
    // {
    //     const int start = h_csrRowPtrA[row  ] - baseA;
    //     const int end   = h_csrRowPtrA[row+1] - baseA;
    //     for(int colidx = start ; colidx < end ; colidx++)
    //     {
    //         const int col = h_csrColIndA[colidx] - baseA;
    //         const double Areg = h_csrValA[colidx];
    //        // h_A[row + col*lda] = Areg;
    //         h_A[col + colsA*row] = Areg
    //     }
    // }
    for(int row = 0 ; row < rowsA ; row++)
    {
        const int start = h_csrRowPtrA[row  ]-baseA;
        const int end   = h_csrRowPtrA[row+1]-baseA;
        for(int colidx = start ; colidx < end ; colidx++)
        {
            const int col = h_csrColIndA[colidx]-baseA;
            const double Areg = h_csrValA[colidx];
            h_A[row + col*lda] = Areg; //col major order
            //h_A[col + colsA*row] = Areg;  //row major order, should be wrong
        }
    }
    // printMatrix(rowsA, colsA, h_A, lda, "h_A");
    // for(int col = 0 ; col < colsA ; col++)
    // {
    //     const int start = h_cscColPtrA[col  ];
    //     const int end   = h_cscColPtrA[col+1];
    //     printf("%d\n", start);
    //     for(int rowidx = start ; rowidx < end ; rowidx++)
    //     {
    //         const int row = h_cscRowIndA[rowidx];
    //         const double Areg = h_cscValA[rowidx];
    //        // h_A[row + col*lda] = Areg;
    //         h_A[col + colsA*(row)] = Areg;
    //     }
    // }


    printf("step 3: set right hand side vector (b) to 1\n");
    for(int row = 0 ; row < rowsA ; row++)
    {
        h_b[row] = 1.0;
    }

    // // verify if A is symmetric or not.
    // if ( 0 == strcmp(opts.testFunc, "chol") )
    // {
    //     int issym = 1;
    //     for(int j = 0 ; j < colsA ; j++)
    //     {
    //         for(int i = j ; i < rowsA ; i++)
    //         {
    //             double Aij = h_A[i + j*lda];
    //             double Aji = h_A[j + i*lda];
    //             if ( Aij != Aji )
    //             {
    //                 issym = 0;
    //                 break;
    //             }
    //         }
    //     }
    //     if (!issym)
    //     {
    //         printf("Error: A has no symmetric pattern, please use LU or QR \n");
    //         exit(EXIT_FAILURE);
    //     }
    // }

    checkCudaErrors(cusolverDnCreate(&handle));
    checkCudaErrors(cublasCreate(&cublasHandle));
    checkCudaErrors(cudaStreamCreate(&stream));

    checkCudaErrors(cusolverDnSetStream(handle, stream));
    checkCudaErrors(cublasSetStream(cublasHandle, stream));


    checkCudaErrors(cudaMalloc((void **)&d_A, sizeof(double)*lda*colsA));
    checkCudaErrors(cudaMalloc((void **)&d_x, sizeof(double)*colsA));
    checkCudaErrors(cudaMalloc((void **)&d_b, sizeof(double)*rowsA));
    checkCudaErrors(cudaMalloc((void **)&d_r, sizeof(double)*rowsA));
    checkCudaErrors(cudaMalloc((void **)&d_tr, sizeof(double)*rowsA));

    printf("step 4: prepare data on device\n");
    checkCudaErrors(cudaMemcpy(d_A, h_A, sizeof(double)*lda*colsA, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_b, h_b, sizeof(double)*rowsA, cudaMemcpyHostToDevice));

    printf("step 6: compute AtA \n");
    cublasStatus_t cbstat;
    double al =1.0;// al =1
    double bet =0.0;// bet =0
    //double* dAcopy;
    double* dAtA;
    //checkCudaErrors(cudaMalloc(&dAcopy, sizeof(double)*lda*colsA));
    checkCudaErrors(cudaMalloc(&dAtA, sizeof(double)*colsA*colsA));
    //checkCudaErrors(cudaMemcpy(dAcopy, d_A, sizeof(double)*lda*colsA, cudaMemcpyDeviceToDevice));
    //cbstat = cublasDgemm(cublasHandle,CUBLAS_OP_T,CUBLAS_OP_N,colsA,rowsA,rowsA,&al,d_A,colsA,d_A,rowsA,&bet,dAtA,colsA);
    cbstat = cublasDgemm(cublasHandle,CUBLAS_OP_T,CUBLAS_OP_N,colsA,colsA,rowsA,&al,d_A,rowsA,d_A,rowsA,&bet,dAtA,colsA);

    //checkMatrix(colsA, colsA, dAtA, colsA, "hAtA");

    //checkCudaErrors(cudaDeviceSynchronize());

    //if (dAcopy) { checkCudaErrors(cudaFree(dAcopy)); }

    printf("step 7: compute At*b \n");
    double* d_Atb;
    checkCudaErrors(cudaMalloc((void **)&d_Atb, sizeof(double)*colsA));
    cbstat = cublasDgemv(cublasHandle,CUBLAS_OP_T,colsA,colsA,&al,d_A,colsA,d_b,1,&bet,d_Atb,1);
    //checkArray(d_Atb, colsA, "Atb");
    // if (d_b) { checkCudaErrors(cudaFree(d_b)); }
    //if (d_A) { checkCudaErrors(cudaFree(d_A)); }
    //checkCudaErrors(cudaDeviceSynchronize());

    if (cublasHandle) { checkCudaErrors(cublasDestroy(cublasHandle)); }
    checkCudaErrors(cublasCreate(&cublasHandle));
    checkCudaErrors(cublasSetStream(cublasHandle, stream));
    printf("step 8: solves AtA*x = At*b \n");

    // d_A and d_b are read-only
    if ( 0 == strcmp(opts.testFunc, "svd") )
    {
        linearSolverSVD(handle, colsA, dAtA, colsA, d_Atb, d_x);
    }
    else if ( 0 == strcmp(opts.testFunc, "chol") )
    {
        linearSolverCHOL(handle, colsA, dAtA, colsA, d_Atb, d_x);
    }
    else if ( 0 == strcmp(opts.testFunc, "lu") )
    {
        linearSolverLU(handle, colsA, dAtA, colsA, d_Atb, d_x);
    }
    else if ( 0 == strcmp(opts.testFunc, "qr") )
    {
        linearSolverQR(handle, colsA, dAtA, colsA, d_Atb, d_x);
    }
    else
    {
        fprintf(stderr, "Error: %s is unknown function\n", opts.testFunc);
        exit(EXIT_FAILURE);
    }
    printf("step 9: evaluate residual\n");
    checkCudaErrors(cudaMemcpy(d_r, d_b, sizeof(double)*rowsA, cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaMemcpy(d_tr, d_Atb, sizeof(double)*colsA, cudaMemcpyDeviceToDevice));
    // r = b - A*x
    checkCudaErrors(cublasDgemm_v2(
        cublasHandle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        rowsA,
        1,
        colsA,
        &minus_one,
        d_A,
        lda,
        d_x,
        rowsA,
        &one,
        d_r,
        rowsA));
    checkCudaErrors(cublasDgemm_v2(
        cublasHandle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        colsA,
        1,
        colsA,
        &minus_one,
        dAtA,
        colsA,
        d_x,
        colsA,
        &one,
        d_tr,
        colsA));
//asd
    checkCudaErrors(cudaMemcpy(h_x, d_x, sizeof(double)*colsA, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(h_r, d_r, sizeof(double)*rowsA, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(h_tr, d_tr, sizeof(double)*colsA, cudaMemcpyDeviceToHost));
    b_inf = vec_norminf(rowsA, h_b);
    x_inf = vec_norminf(colsA, h_x);
    r_inf = vec_norminf(rowsA, h_r);
    r_inf = vec_norminf(colsA, h_tr);
    A_inf = mat_norminf(rowsA, colsA, h_A, lda);

    // printArray(h_x, colsA);
    printf("|b - A*x| = %E \n", r_inf);
    printf("|Atb - AtA*x| = %E \n", tr_inf);
    printf("|A| = %E \n", A_inf);
    printf("|x| = %E \n", x_inf);
    printf("|b| = %E \n", b_inf);
    printf("|b - A*x|/(|A|*|x|) = %E \n", r_inf/(A_inf * x_inf));

    if (handle) { checkCudaErrors(cusolverDnDestroy(handle)); }
    if (cublasHandle) { checkCudaErrors(cublasDestroy(cublasHandle)); }
    if (stream) { checkCudaErrors(cudaStreamDestroy(stream)); }

    if (h_csrValA   ) { free(h_csrValA); }
    if (h_csrRowPtrA) { free(h_csrRowPtrA); }
    if (h_csrColIndA) { free(h_csrColIndA); }
    // if (h_cscValA   ) { free(h_cscValA); }
    // if (h_cscColPtrA) { free(h_cscColPtrA); }
    // if (h_cscRowIndA) { free(h_cscRowIndA); }

    if (h_A) { free(h_A); }
    if (h_x) { free(h_x); }
    if (h_b) { free(h_b); }
    if (h_r) { free(h_r); }
    if (h_tr) { free(h_tr); }

    if (d_A) { checkCudaErrors(cudaFree(d_A)); }
    if (d_x) { checkCudaErrors(cudaFree(d_x)); }
    if (d_b) { checkCudaErrors(cudaFree(d_b)); }
    if (d_r) { checkCudaErrors(cudaFree(d_r)); }
    if (d_tr) { checkCudaErrors(cudaFree(d_tr)); }

    return 0;
}

