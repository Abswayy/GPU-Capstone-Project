#include "cuda_utils.h"  // Note: Original was cuda_utils.h, but we've renamed the header to GPU_HELPERS_H internally
#include <cstdio>
#include <cstring>

// Implementation for verifying CUDA status
void verifyCudaStatus(cudaError_t result, const char *sourceFile, int sourceLine) {
    if (result != cudaSuccess) {
        printf("GPU Operation Failed: %s in %s at line %d\n", cudaGetErrorString(result), sourceFile, sourceLine);
        exit(result);
    }
}

// Implementation to fetch GPU device info
void fetchGpuDeviceInfo(GpuDeviceInfo* infoPtr, int gpuId) {
    if (!infoPtr) return;
    
    cudaDeviceProp deviceProp;
    GPU_VERIFY_CALL(cudaGetDeviceProperties(&deviceProp, gpuId));
    
    strncpy(infoPtr->deviceName, deviceProp.name, 255);
    infoPtr->deviceName[255] = '\0';  // Guarantee string termination
    infoPtr->computeMajor = deviceProp.major;
    infoPtr->computeMinor = deviceProp.minor;
    infoPtr->smCount = deviceProp.multiProcessorCount;
    infoPtr->globalMemSize = deviceProp.totalGlobalMem;
    infoPtr->maxBlockThreads = deviceProp.maxThreadsPerBlock;
    infoPtr->maxSmThreads = deviceProp.maxThreadsPerMultiProcessor;
    infoPtr->threadWarpSize = deviceProp.warpSize;
}

// Implementation to display GPU device info
void displayGpuDeviceInfo(int gpuId) {
    GpuDeviceInfo deviceInfo;
    fetchGpuDeviceInfo(&deviceInfo, gpuId);
    
    printf("GPU Device Details:\n");
    printf("  Device Name: %s\n", deviceInfo.deviceName);
    printf("  Compute Version: %d.%d\n", deviceInfo.computeMajor, deviceInfo.computeMinor);
    printf("  SM Quantity: %d\n", deviceInfo.smCount);
    printf("  Global Memory Total: %zu MB\n", deviceInfo.globalMemSize / (1024 * 1024));
    printf("  Maximum Threads/Block: %d\n", deviceInfo.maxBlockThreads);
    printf("  Maximum Threads/SM: %d\n", deviceInfo.maxSmThreads);
    printf("  Warp Thread Count: %d\n", deviceInfo.threadWarpSize);
}

// Implementation to compute best dimensions
void computeBestDimensions(
    int imgWidth, 
    int imgHeight, 
    dim3* blockSize, 
    dim3* gridSize,
    int desiredBlockThreads
) {
    if (!blockSize || !gridSize) return;
    
    blockSize->x = desiredBlockThreads;
    blockSize->y = desiredBlockThreads;
    blockSize->z = 1;
    
    gridSize->x = (imgWidth + blockSize->x - 1) / blockSize->x;
    gridSize->y = (imgHeight + blockSize->y - 1) / blockSize->y;
    gridSize->z = 1;
}