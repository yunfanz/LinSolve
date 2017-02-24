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
