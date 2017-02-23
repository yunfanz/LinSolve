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

#include <cuda_runtime.h>

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
    const float *Acopy,
    int lda,
    const float *b,
    float *x)
{
    int bufferSize = 0;
    int *info = NULL;
    float *buffer = NULL;
    float *A = NULL;
    int h_info = 0;
    float start, stop;
    float time_solve;
    cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;

    checkCudaErrors(cusolverDnSpotrf_bufferSize(handle, uplo, n, (float*)Acopy, lda, &bufferSize));

    checkCudaErrors(cudaMalloc(&info, sizeof(int)));
    checkCudaErrors(cudaMalloc(&buffer, sizeof(float)*bufferSize));
    checkCudaErrors(cudaMalloc(&A, sizeof(float)*lda*n));


    // prepare a copy of A because potrf will overwrite A with L
    checkCudaErrors(cudaMemcpy(A, Acopy, sizeof(float)*lda*n, cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaMemset(info, 0, sizeof(int)));

    start = second();
    start = second();

    checkCudaErrors(cusolverDnSpotrf(handle, uplo, n, A, lda, buffer, bufferSize, info));

    checkCudaErrors(cudaMemcpy(&h_info, info, sizeof(int), cudaMemcpyDeviceToHost));

    if ( 0 != h_info ){
        fprintf(stderr, "Error: Cholesky factorization failed\n");
    }

    checkCudaErrors(cudaMemcpy(x, b, sizeof(float)*n, cudaMemcpyDeviceToDevice));

    checkCudaErrors(cusolverDnSpotrs(handle, uplo, n, 1, A, lda, x, n, info));

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
    const float *Acopy,
    int lda,
    const float *b,
    float *x)
{
    int bufferSize = 0;
    int *info = NULL;
    float *buffer = NULL;
    float *A = NULL;
    int *ipiv = NULL; // pivoting sequence
    int h_info = 0;
    float start, stop;
    float time_solve;

    checkCudaErrors(cusolverDnSgetrf_bufferSize(handle, n, n, (float*)Acopy, lda, &bufferSize));

    checkCudaErrors(cudaMalloc(&info, sizeof(int)));
    checkCudaErrors(cudaMalloc(&buffer, sizeof(float)*bufferSize));
    checkCudaErrors(cudaMalloc(&A, sizeof(float)*lda*n));
    checkCudaErrors(cudaMalloc(&ipiv, sizeof(int)*n));


    // prepare a copy of A because getrf will overwrite A with L
    checkCudaErrors(cudaMemcpy(A, Acopy, sizeof(float)*lda*n, cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaMemset(info, 0, sizeof(int)));

    start = second();
    start = second();

    checkCudaErrors(cusolverDnSgetrf(handle, n, n, A, lda, buffer, ipiv, info));
    checkCudaErrors(cudaMemcpy(&h_info, info, sizeof(int), cudaMemcpyDeviceToHost));

    if ( 0 != h_info ){
        fprintf(stderr, "Error: LU factorization failed\n");
    }

    checkCudaErrors(cudaMemcpy(x, b, sizeof(float)*n, cudaMemcpyDeviceToDevice));
    checkCudaErrors(cusolverDnSgetrs(handle, CUBLAS_OP_N, n, 1, A, lda, ipiv, x, n, info));
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
        fprintf(stderr, "Error: LU factorization failed\n");
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
            if ((STRCASECMP(solverType, "chol") != 0) && (STRCASECMP(solverType, "lu") != 0) && (STRCASECMP(solverType, "qr") != 0))
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

int is_symmetric(float* h_A, int colsA, int rowsA, int lda)
{
    int issym = 1;
    for(int j = 0 ; j < colsA ; j++)
    {
        for(int i = j ; i < rowsA ; i++)
        {
            float Aij = h_A[i + j*lda];
            float Aji = h_A[j + i*lda];
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
    float *h_csrValA = NULL;

    float *h_A = NULL; // dense matrix from CSR(A)
    float *h_x = NULL; // a copy of d_x
    float *h_b = NULL; // b = ones(m,1)
    float *h_r = NULL; // r = b - A*x, a copy of d_r
    float *h_tr = NULL;

    float *d_A = NULL; // a copy of h_A
    float *d_x = NULL; // x = A \ b
    float *d_b = NULL; // a copy of h_b
    float *d_r = NULL; // r = b - A*x
    float *d_tr = NULL; // tr = Atb - AtA*x

    // the constants are used in residual evaluation, r = b - A*x
    const float minus_one = -1.0;
    const float one = 1.0;

    float x_inf = 0.0;
    float r_inf = 0.0;
    float A_inf = 0.0;
    float b_inf = 0.0;
    float Ax_inf = 0.0;
    float tr_inf = 0.0;
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

    if (loadMMSparseMatrix<float>(opts.sparse_mat_filename, 's', true , &rowsA, &colsA,
               &nnzA, &h_csrValA, &h_csrRowPtrA, &h_csrColIndA, true))
    {
        exit(EXIT_FAILURE);
    }
    baseA = h_csrRowPtrA[0]; // baseA = {0,1}

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

    h_A = (float*)malloc(sizeof(float)*lda*colsA);
    h_x = (float*)malloc(sizeof(float)*colsA);
    h_b = (float*)malloc(sizeof(float)*rowsA);
    h_r = (float*)malloc(sizeof(float)*rowsA);
    h_tr = (float*)malloc(sizeof(float)*colsA);
    assert(NULL != h_A);
    assert(NULL != h_x);
    assert(NULL != h_b);
    assert(NULL != h_r);

    memset(h_A, 0, sizeof(float)*lda*colsA);

    for(int row = 0 ; row < rowsA ; row++)
    {
        const int start = h_csrRowPtrA[row  ] - baseA;
        const int end   = h_csrRowPtrA[row+1] - baseA;
        for(int colidx = start ; colidx < end ; colidx++)
        {
            const int col = h_csrColIndA[colidx] - baseA;
            const float Areg = h_csrValA[colidx];
            h_A[row + col*lda] = Areg;
        }
    }

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
    //             float Aij = h_A[i + j*lda];
    //             float Aji = h_A[j + i*lda];
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


    checkCudaErrors(cudaMalloc((void **)&d_A, sizeof(float)*lda*colsA));
    checkCudaErrors(cudaMalloc((void **)&d_x, sizeof(float)*colsA));
    checkCudaErrors(cudaMalloc((void **)&d_b, sizeof(float)*rowsA));
    checkCudaErrors(cudaMalloc((void **)&d_r, sizeof(float)*rowsA));
    checkCudaErrors(cudaMalloc((void **)&d_tr, sizeof(float)*rowsA));

    printf("step 4: prepare data on device\n");
    checkCudaErrors(cudaMemcpy(d_A, h_A, sizeof(float)*lda*colsA, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_b, h_b, sizeof(float)*rowsA, cudaMemcpyHostToDevice));

    printf("step 6: compute AtA \n");
    cublasStatus_t cbstat;
    float al =1.0f;// al =1
    float bet =0.0f;// bet =0
    //float* dAcopy;
    float* dAtA;
    //checkCudaErrors(cudaMalloc(&dAcopy, sizeof(float)*lda*colsA));
    checkCudaErrors(cudaMalloc(&dAtA, sizeof(float)*colsA*colsA));
    //checkCudaErrors(cudaMemcpy(dAcopy, d_A, sizeof(float)*lda*colsA, cudaMemcpyDeviceToDevice));
    cbstat = cublasSgemm(cublasHandle,CUBLAS_OP_T,CUBLAS_OP_N,colsA,colsA,rowsA,&al,d_A,colsA,d_A,rowsA,&bet,dAtA,colsA);
    checkCudaErrors(cudaDeviceSynchronize());

    //if (dAcopy) { checkCudaErrors(cudaFree(dAcopy)); }

    printf("step 7: compute At*b \n");
    float* d_Atb;
    checkCudaErrors(cudaMalloc((void **)&d_Atb, sizeof(float)*colsA));
    cbstat = cublasSgemv(cublasHandle,CUBLAS_OP_T,colsA,colsA,&al,d_A,colsA,d_b,1,&bet,d_Atb,1);
    // if (d_b) { checkCudaErrors(cudaFree(d_b)); }
    // if (d_A) { checkCudaErrors(cudaFree(d_A)); }
    checkCudaErrors(cudaDeviceSynchronize());

    printf("step 8: solves AtA*x = At*b \n");

    // d_A and d_b are read-only
    if ( 0 == strcmp(opts.testFunc, "chol") )
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
    checkCudaErrors(cudaMemcpy(d_r, d_b, sizeof(float)*rowsA, cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaMemcpy(d_tr, d_Atb, sizeof(float)*colsA, cudaMemcpyDeviceToDevice));
    // r = b - A*x
    checkCudaErrors(cublasSgemm_v2(
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
    checkCudaErrors(cublasSgemm_v2(
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
    checkCudaErrors(cudaMemcpy(h_x, d_x, sizeof(float)*colsA, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(h_r, d_r, sizeof(float)*rowsA, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(h_tr, d_tr, sizeof(float)*colsA, cudaMemcpyDeviceToHost));
    b_inf = vec_norminf(rowsA, h_b);
    x_inf = vec_norminf(colsA, h_x);
    r_inf = vec_norminf(rowsA, h_r);
    r_inf = vec_norminf(colsA, h_tr);
    A_inf = mat_norminf(rowsA, colsA, h_A, lda);

    printArray(h_x, colsA);
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

    if (h_A) { free(h_A); }
    if (h_x) { free(h_x); }
    if (h_b) { free(h_b); }
    if (h_r) { free(h_r); }

    if (d_A) { checkCudaErrors(cudaFree(d_A)); }
    if (d_x) { checkCudaErrors(cudaFree(d_x)); }
    if (d_b) { checkCudaErrors(cudaFree(d_b)); }
    if (d_r) { checkCudaErrors(cudaFree(d_r)); }

    return 0;
}

