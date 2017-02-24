#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <cuda_runtime.h>
#include "SI.h"
#include "cublas_v2.h"
#include "cusolverDn.h"
#include "helper_cuda.h"
#include "helper_cusolver.h"
/* 
 * Author: Yunfan Gerry Zhang
 * This file defines the template wrappers around cublas and cusolver kernels 
 * to easily use T_ELEM or double
 */

// template <typename T_ELEM>
// cusolverStatus_t cusolverDngetrf_bufferSize(cusolverDnHandle_t handle, int m, int n, T_ELEM *A, int lda, int *Lwork );

// template <typename T_ELEM>
// cusolverStatus_t cusolverDnpotrf_bufferSize(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, T_ELEM *A, int lda, int *Lwork );

// template <typename T_ELEM>
// cusolverStatus_t cusolverDnpotrs(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, int nrhs, const T_ELEM *A, int lda, T_ELEM *B, int ldb, int *devInfo);

// template <typename T_ELEM>
// cusolverStatus_t cusolverDngetrf_bufferSize(cusolverDnHandle_t handle, int m, int n, T_ELEM *A, int lda, int *Lwork ); 

// template <typename T_ELEM>
// cusolverStatus_t cusolverDngetrf(cusolverDnHandle_t handle, int m, int n, T_ELEM *A, int lda, T_ELEM *Workspace, int *devIpiv, int *devInfo );

// template <typename T_ELEM>
// cusolverStatus_t cusolverDngetrf_bufferSize(cusolverDnHandle_t handle, int m, int n, T_ELEM *A, int lda, int *Lwork );

// template <typename T_ELEM>
// cusolverStatus_t cusolverDngetrf(cusolverDnHandle_t handle, int m, int n, T_ELEM *A, int lda, T_ELEM *Workspace, int *devIpiv, int *devInfo );

// template <typename T_ELEM>
// cusolverStatus_t cusolverDngetrs(cusolverDnHandle_t handle, cublasOperation_t trans, int n, int nrhs, const T_ELEM *A, int lda, const int *devIpiv, T_ELEM *B, int ldb, int *devInfo );

// template <typename T_ELEM>
// cusolverStatus_t cusolverDngeqrf_bufferSize(cusolverDnHandle_t handle, int m, int n, T_ELEM *A, int lda, int *Lwork ); 

// template <typename T_ELEM>
// cusolverStatus_t cusolverDngeqrf(cusolverDnHandle_t handle, int m, int n, T_ELEM *A, int lda, T_ELEM *TAU, T_ELEM *Workspace, int Lwork, int *devInfo );

// template <typename T_ELEM>
// cusolverStatus_t cusolverDnormqr_bufferSize( cusolverDnHandle_t handle, cublasSideMode_t side, cublasOperation_t trans, int m, int n, int k, const T_ELEM *A, int lda, const T_ELEM *C, int ldc, int *lwork);

// template <typename T_ELEM>
// cusolverStatus_t cusolverDnormqr(cusolverDnHandle_t handle, cublasSideMode_t side, cublasOperation_t trans, int m, int n, int k, const T_ELEM *A, int lda, const T_ELEM *tau, T_ELEM *C, int ldc, T_ELEM *work, int lwork, int *devInfo);

// template <typename T_ELEM>
// cusolverStatus_t cusolverDnorgqr_bufferSize( cusolverDnHandle_t handle, int m, int n, int k, const T_ELEM *A, int lda, int *lwork);

// template <typename T_ELEM>
// cusolverStatus_t cusolverDnorgqr( cusolverDnHandle_t handle, int m, int n, int k, T_ELEM *A, int lda, const T_ELEM *tau, T_ELEM *work, int lwork, int *devInfo);


// template <typename T_ELEM>
// cublasStatus_t cublasgemv(cublasHandle_t handle, cublasOperation_t trans, int m, int n, const T_ELEM *alpha, const T_ELEM *A, int lda, const T_ELEM *x, int incx, const T_ELEM *beta, T_ELEM *y, int incy)

// template <typename T_ELEM>
// cublasStatus_t cublasgemm(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const T_ELEM *alpha, const T_ELEM *A, int lda, const T_ELEM *B, int ldb, const T_ELEM *beta, T_ELEM *C, int ldc)

// template <typename T_ELEM>
// cublasStatus_t cublastrsm(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const T_ELEM *alpha, const T_ELEM *A, int lda, T_ELEM *B, int ldb)


template <> cusolverStatus_t 
cusolverDngetrf_bufferSize<float>(cusolverDnHandle_t handle, int m, int n, float *A, int lda, int *Lwork )
{
	return cusolverDnSgetrf_bufferSize(handle, m, n,*A, lda, *Lwork );
}
template <> cusolverStatus_t 
cusolverDngetrf_bufferSize<double>(cusolverDnHandle_t handle, int m, int n, double *A, int lda, int *Lwork )
{
	return cusolverDnDgetrf_bufferSize(handle, m, n,*A, lda, *Lwork );
}
template <> cusolverStatus_t 
cusolverDnpotrf_bufferSize<float>(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, float *A, int lda, int *Lwork )
{
	return cusolverDnSpotrf_bufferSize(handle, uplo, n, *A, lda, *Lwork );
}
template <> cusolverStatus_t 
cusolverDnpotrf_bufferSize<double>(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, double *A, int lda, int *Lwork )
{
	return cusolverDnDpotrf_bufferSize(handle, uplo, n, *A, lda, *Lwork );
}

template <> cusolverStatus_t 
cusolverDnpotrs<float>(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, int nrhs, const float *A, int lda, float *B, int ldb, int *devInfo)
{
	return cusolverDnSpotrs(handle, uplo, n, nrhs, *A, lda, *B, ldb, *devInfo);
}

template <> cusolverStatus_t 
cusolverDngetrf_bufferSize<float>(cusolverDnHandle_t handle, int m, int n, float *A, int lda, int *Lwork )
{
	return cusolverDnSgetrf_bufferSize( handle, m, n, *A, lda, *Lwork ); 
}

template <> cusolverStatus_t
cusolverDngetrf<float>(cusolverDnHandle_t handle, int m, int n, float *A, int lda, float *Workspace, int *devIpiv, int *devInfo )
{
	cusolverDnSgetrf(handle, m, n, *A, lda, *Workspace, *devIpiv, *devInfo );
}

template <> cusolverStatus_t
cusolverDngetrf_bufferSize<float>(cusolverDnHandle_t handle, int m, int n, float *A, int lda, int *Lwork );
{
	return cusolverDnSgetrf_bufferSize( handle, m, n, *A, lda, *Lwork );
}

template <> cusolverStatus_t
cusolverDngetrf<float>(cusolverDnHandle_t handle, int m, int n, float *A, int lda, float *Workspace, int *devIpiv, int *devInfo );
{
	return cusolverDnSgetrf( handle, m, n, *A, lda, *Workspace, *devIpiv, *devInfo );
}

template <> cusolverStatus_t
cusolverDngetrs<float>(cusolverDnHandle_t handle, cublasOperation_t trans, int n, int nrhs, const float *A, int lda, const int *devIpiv, float *B, int ldb, int *devInfo )
{
	return cusolverDnSgetrs( handle, trans, n, nrhs, *A, lda, *devIpiv, *B, ldb, *devInfo );
}
template <> cusolverStatus_t
cusolverDngeqrf_bufferSize<float>(cusolverDnHandle_t handle, int m, int n, float *A, int lda, int *Lwork )
{
	return cusolverDnSgeqrf_bufferSize( handle, m, n, *A, lda, *Lwork ); 
}
template <> cusolverStatus_t
cusolverDngeqrf<float>(cusolverDnHandle_t handle, int m, int n, float *A, int lda, float *TAU, float *Workspace, int Lwork, int *devInfo )
{
	return cusolverDnSgeqrf( handle, m, n, *A, lda, *TAU, *Workspace, Lwork, *devInfo );

}
template <> cusolverStatus_t
cusolverDnormqr_bufferSize<float>( cusolverDnHandle_t handle, cublasSideMode_t side, cublasOperation_t trans, int m, int n, int k, const float *A, int lda, const float *C, int ldc, int *lwork)
{
	return cusolverDnSormqr_bufferSize(  handle, side, trans, m, n, k, *A, lda, *C, ldc, *lwork);
}
template <> cusolverStatus_t
cusolverDnormqr<float>(cusolverDnHandle_t handle, cublasSideMode_t side, cublasOperation_t trans, int m, int n, int k, const float *A, int lda, const float *tau, float *C, int ldc, float *work, int lwork, int *devInfo)
{
	return cusolverDnSormqr( handle, side, trans, m, n, k, *A, lda, *tau, *C, ldc, *work, lwork, *devInfo);
}
template <> cusolverStatus_t
cusolverDnorgqr_bufferSize<float>( cusolverDnHandle_t handle, int m, int n, int k, const float *A, int lda, int *lwork)
{
	return cusolverDnSorgqr_bufferSize(  handle, m, n, k, *A, lda, *lwork);
}
template <> cusolverStatus_t
cusolverDnorgqr<float>( cusolverDnHandle_t handle, int m, int n, int k, float *A, int lda, const float *tau, float *work, int lwork, int *devInfo)
{
	return cusolverDnSorgqr(  handle, m, n, k, *A, lda, *tau, *work, lwork, *devInfo);
}
template <> cusolverStatus_t
cublasgemv<float>(cublasHandle_t handle, cublasOperation_t trans, int m, int n, const float *alpha, const float *A, int lda, const float *x, int incx, const float *beta, float *y, int incy)
{
	return cublasSgemv(cublasHandle_t handle, trans, m, n, *alpha, *A, lda, *x, incx, *beta, *y, incy);
}
template <> cusolverStatus_t
cublasgemm<float>(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const float *alpha, const float *A, int lda, const float *B, int ldb, const float *beta, float *C, int ldc)
{
	return cublasSgemm(cublasHandle_t handle, transa, transb, m, n, k, *alpha, *A, lda, *B, ldb, *beta, *C, ldc);
}
template <> cublasStatus_t 
cublastrsm<float>(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const float *alpha, const float *A, int lda, float *B, int ldb)
{
	return cublasStrsm(cublasHandle_t handle, side,  uplo, trans, diag, m, n, *alpha, *A, lda, *B, ldb);
}



template <> cusolverStatus_t 
cusolverDnpotrs<double>(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, int nrhs, const double *A, int lda, double *B, int ldb, int *devInfo)
{
	return cusolverDnDpotrs(handle, uplo, n, nrhs, *A, lda, *B, ldb, *devInfo);
}

template <> cusolverStatus_t 
cusolverDngetrf_bufferSize<double>(cusolverDnHandle_t handle, int m, int n, double *A, int lda, int *Lwork )
{
	return cusolverDnDgetrf_bufferSize( handle, m, n, *A, lda, *Lwork ); 
}

template <> cusolverStatus_t
cusolverDngetrf<double>(cusolverDnHandle_t handle, int m, int n, double *A, int lda, double *Workspace, int *devIpiv, int *devInfo )
{
	cusolverDnDgetrf(handle, m, n, *A, lda, *Workspace, *devIpiv, *devInfo );
}

template <> cusolverStatus_t
cusolverDngetrf_bufferSize<double>(cusolverDnHandle_t handle, int m, int n, double *A, int lda, int *Lwork );
{
	return cusolverDnDgetrf_bufferSize( handle, m, n, *A, lda, *Lwork );
}

template <> cusolverStatus_t
cusolverDngetrf<double>(cusolverDnHandle_t handle, int m, int n, double *A, int lda, double *Workspace, int *devIpiv, int *devInfo );
{
	return cusolverDnDgetrf( handle, m, n, *A, lda, *Workspace, *devIpiv, *devInfo );
}

template <> cusolverStatus_t
cusolverDngetrs<double>(cusolverDnHandle_t handle, cublasOperation_t trans, int n, int nrhs, const double *A, int lda, const int *devIpiv, double *B, int ldb, int *devInfo )
{
	return cusolverDnDgetrs( handle, trans, n, nrhs, *A, lda, *devIpiv, *B, ldb, *devInfo );
}
template <> cusolverStatus_t
cusolverDngeqrf_bufferSize<double>(cusolverDnHandle_t handle, int m, int n, double *A, int lda, int *Lwork )
{
	return cusolverDnDgeqrf_bufferSize( handle, m, n, *A, lda, *Lwork ); 
}
template <> cusolverStatus_t
cusolverDngeqrf<double>(cusolverDnHandle_t handle, int m, int n, double *A, int lda, double *TAU, double *Workspace, int Lwork, int *devInfo )
{
	return cusolverDnDgeqrf( handle, m, n, *A, lda, *TAU, *Workspace, Lwork, *devInfo );

}
template <> cusolverStatus_t
cusolverDnormqr_bufferSize<double>( cusolverDnHandle_t handle, cublasSideMode_t side, cublasOperation_t trans, int m, int n, int k, const double *A, int lda, const double *C, int ldc, int *lwork)
{
	return cusolverDnDormqr_bufferSize(  handle, side, trans, m, n, k, *A, lda, *C, ldc, *lwork);
}
template <> cusolverStatus_t
cusolverDnormqr<double>(cusolverDnHandle_t handle, cublasSideMode_t side, cublasOperation_t trans, int m, int n, int k, const double *A, int lda, const double *tau, double *C, int ldc, double *work, int lwork, int *devInfo)
{
	return cusolverDnDormqr( handle, side, trans, m, n, k, *A, lda, *tau, *C, ldc, *work, lwork, *devInfo);
}
template <>
cusolverStatus_t cusolverDnorgqr_bufferSize<double>( cusolverDnHandle_t handle, int m, int n, int k, const double *A, int lda, int *lwork)
{
	return cusolverDnDorgqr_bufferSize(  handle, m, n, k, *A, lda, *lwork);
}
template <>
cusolverStatus_t cusolverDnorgqr<double>( cusolverDnHandle_t handle, int m, int n, int k, double *A, int lda, const double *tau, double *work, int lwork, int *devInfo)
{
	return cusolverDnDorgqr(  handle, m, n, k, *A, lda, *tau, *work, lwork, *devInfo);
}
template <> cusolverStatus_t
cublasgemv<double>(cublasHandle_t handle, cublasOperation_t trans, int m, int n, const double *alpha, const double *A, int lda, const double *x, int incx, const double *beta, double *y, int incy)
{
	return cublasDgemv(cublasHandle_t handle, trans, m, n, *alpha, *A, lda, *x, incx, *beta, *y, incy);
}
template <> cusolverStatus_t
cublasgemm<double>(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const double *alpha, const double *A, int lda, const double *B, int ldb, const double *beta, double *C, int ldc)
{
	return cublasDgemm(cublasHandle_t handle, transa, transb, m, n, k, *alpha, *A, lda, *B, ldb, *beta, *C, ldc);
}
template <> cublasStatus_t 
cublastrsm<double>(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag, int m, int n, const double *alpha, const double *A, int lda, double *B, int ldb)
{
	return cublasDtrsm(cublasHandle_t handle, side,  uplo, trans, diag, m, n, *alpha, *A, lda, *B, ldb);
}