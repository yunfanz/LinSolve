#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <assert.h>

#include <cuda_runtime.h>
#include "SI.h"
#include "cublas_v2.h"
#include "cusolverDn.h"
#include "helper_cuda.h"
#include "helper_cusolver.h"
//============================================================================================
// Device information utilities
//============================================================================================

#if defined(__cplusplus)
extern "C" {
#endif /* __cplusplus */

int getDeviceVersion(void)
{
    int device;
    struct cudaDeviceProp properties;

    if (cudaGetDevice(&device) != cudaSuccess)
    {
        printf("failed to get device\n");
        return 0;
    }

    if (cudaGetDeviceProperties(&properties, device) != cudaSuccess)
    {
        printf("failed to get properties\n");
        return 0;
    }

    return properties.major * 100 + properties.minor * 10;
}

size_t getDeviceMemory(void)
{
    struct cudaDeviceProp properties;
    int device;

    if (cudaGetDevice(&device) != cudaSuccess)
    {
        return 0;
    }

    if (cudaGetDeviceProperties(&properties, device) != cudaSuccess)
    {
        return 0;
    }

    return properties.totalGlobalMem;
}
#if defined(__cplusplus)
}
#endif /* __cplusplus */

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
