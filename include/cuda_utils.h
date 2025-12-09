#ifndef GPU_HELPERS_H
#define GPU_HELPERS_H

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#ifdef __cplusplus
extern "C" {
#endif

// Function to verify CUDA operation status
void verifyCudaStatus(cudaError_t result, const char *sourceFile, int sourceLine);

// Macro for verifying CUDA calls
#define GPU_VERIFY_CALL(expr) { verifyCudaStatus((expr), __FILE__, __LINE__); }

// Basic info structure for GPU device
typedef struct {
    char deviceName[256];
    int computeMajor;
    int computeMinor;
    int smCount;
    size_t globalMemSize;
    int maxBlockThreads;
    int maxSmThreads;
    int threadWarpSize;
} GpuDeviceInfo;

// Retrieve properties of a GPU device
void fetchGpuDeviceInfo(GpuDeviceInfo* infoPtr, int gpuId);

// Display properties of a GPU device
void displayGpuDeviceInfo(int gpuId);

// Determine best grid/block sizes for 2D tasks
void computeBestDimensions(
    int imgWidth, 
    int imgHeight, 
    dim3* blockSize, 
    dim3* gridSize,
    int desiredBlockThreads
);

#ifdef __cplusplus
}
#endif

#endif // GPU_HELPERS_H