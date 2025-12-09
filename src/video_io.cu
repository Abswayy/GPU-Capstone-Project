#include "video_io.h"
#include "cuda_utils.h"  // Updated to match renamed header
#include "kernels.h"     // Updated to match renamed header (kernels.h -> IMAGE_KERNELS_H internally)
#include <cstdio>
#include <cstdlib>
#include <cstring>

// External C++ functions from video_io_impl.cpp
extern "C" {
    void* initCvHandler();
    void cleanupCvHandler(void* handler);
    bool initCvInput(void* handler, const char* inputSrc, bool isLocalFile);
    bool initCvOutput(void* handler, const char* outputFile);
    bool isCvInputActive(void* handler);
    int getCvImgWidth(void* handler);
    int getCvImgHeight(void* handler);
    int getCvImgChannels(void* handler);
    double getCvFrameRate(void* handler);
    int getCvFrameTotal(void* handler);
    int getCvFrameCurrent(void* handler);
    bool fetchCvFrame(void* handler);
    bool saveCvFrame(void* handler);
    void shutdownCvInput(void* handler);
    
    // Access to frame buffers
    unsigned char* fetchCurrentFrameBuffer(void* handler);
    unsigned char* fetchOutputFrameBuffer(void* handler);
    void updateOutputFrameBuffer(void* handler, unsigned char* buffer);
}

// Structure for handling video
struct VideoHandler {
    void* cvHandler;  // Hidden pointer to CV impl
    unsigned char* currFrameBuffer;
    unsigned char* outFrameBuffer;
    int imgWidth;
    int imgHeight;
    int imgChannels;
    size_t bufferSize;
    
    // GPU buffers
    unsigned char* gpuInputBuffer;
    unsigned char* gpuOutputBuffer;
};

// Initialize video handler
VideoHandler* initVideoHandler() {
    VideoHandler* handler = (VideoHandler*)malloc(sizeof(VideoHandler));
    if (!handler) {
        printf("Failed to allocate VideoHandler memory\n");
        return NULL;
    }
    
    // Set defaults
    handler->cvHandler = initCvHandler();
    handler->currFrameBuffer = NULL;
    handler->outFrameBuffer = NULL;
    handler->imgWidth = 0;
    handler->imgHeight = 0;
    handler->imgChannels = 0;
    handler->bufferSize = 0;
    handler->gpuInputBuffer = NULL;
    handler->gpuOutputBuffer = NULL;
    
    if (!handler->cvHandler) {
        printf("Failed to initialize CV handler\n");
        free(handler);
        return NULL;
    }
    
    return handler;
}

// Cleanup video handler
void cleanupVideoHandler(VideoHandler* handler) {
    if (!handler) return;
    
    // Release GPU memory
    if (handler->gpuInputBuffer) {
        cudaFree(handler->gpuInputBuffer);
    }
    if (handler->gpuOutputBuffer) {
        cudaFree(handler->gpuOutputBuffer);
    }
    
    // Release host buffers
    if (handler->currFrameBuffer) {
        free(handler->currFrameBuffer);
    }
    if (handler->outFrameBuffer) {
        free(handler->outFrameBuffer);
    }
    
    // Cleanup CV handler
    cleanupCvHandler(handler->cvHandler);
    
    free(handler);
}

// Initialize input source
bool initInputSource(VideoHandler* handler, const char* inputSrc, bool isLocalFile) {
    if (!handler || !handler->cvHandler) {
        return false;
    }
    
    if (!initCvInput(handler->cvHandler, inputSrc, isLocalFile)) {
        return false;
    }
    
    // Retrieve properties
    handler->imgWidth = getCvImgWidth(handler->cvHandler);
    handler->imgHeight = getCvImgHeight(handler->cvHandler);
    handler->imgChannels = getCvImgChannels(handler->cvHandler);
    handler->bufferSize = handler->imgWidth * handler->imgHeight * handler->imgChannels;
    
    // Allocate buffers on host
    handler->currFrameBuffer = (unsigned char*)malloc(handler->bufferSize);
    handler->outFrameBuffer = (unsigned char*)malloc(handler->bufferSize);
    
    if (!handler->currFrameBuffer || !handler->outFrameBuffer) {
        printf("Failed to allocate buffer memory\n");
        return false;
    }
    
    // Allocate GPU buffers
    GPU_VERIFY_CALL(cudaMalloc((void**)&handler->gpuInputBuffer, handler->bufferSize));
    GPU_VERIFY_CALL(cudaMalloc((void**)&handler->gpuOutputBuffer, handler->bufferSize));
    
    return true;
}

// Initialize output file
bool initOutputFile(VideoHandler* handler, const char* outputFile) {
    if (!handler || !handler->cvHandler) {
        return false;
    }
    
    return initCvOutput(handler->cvHandler, outputFile);
}

// Check input status
bool isInputActive(VideoHandler* handler) {
    if (!handler || !handler->cvHandler) {
        return false;
    }
    
    return isCvInputActive(handler->cvHandler);
}

// Retrieve video details
int getInputWidth(VideoHandler* handler) {
    return handler ? handler->imgWidth : 0;
}

int getInputHeight(VideoHandler* handler) {
    return handler ? handler->imgHeight : 0;
}

int getInputChannels(VideoHandler* handler) {
    return handler ? handler->imgChannels : 0;
}

double getInputFrameRate(VideoHandler* handler) {
    if (!handler || !handler->cvHandler) {
        return 0.0;
    }
    
    return getCvFrameRate(handler->cvHandler);
}

int getInputFrameTotal(VideoHandler* handler) {
    if (!handler || !handler->cvHandler) {
        return 0;
    }
    
    return getCvFrameTotal(handler->cvHandler);
}

int getInputFrameCurrent(VideoHandler* handler) {
    if (!handler || !handler->cvHandler) {
        return 0;
    }
    
    return getCvFrameCurrent(handler->cvHandler);
}

// Fetch input frame
bool fetchInputFrame(VideoHandler* handler) {
    if (!handler || !handler->cvHandler) {
        return false;
    }
    
    if (!fetchCvFrame(handler->cvHandler)) {
        return false;
    }
    
    // Retrieve buffer from CV
    unsigned char* frameBuffer = fetchCurrentFrameBuffer(handler->cvHandler);
    if (!frameBuffer) {
        return false;
    }
    
    // Copy to local buffer
    memcpy(handler->currFrameBuffer, frameBuffer, handler->bufferSize);
    
    return true;
}

// Save output frame
bool saveOutputFrame(VideoHandler* handler) {
    if (!handler || !handler->cvHandler) {
        return false;
    }
    
    // Update CV output buffer
    updateOutputFrameBuffer(handler->cvHandler, handler->outFrameBuffer);
    
    return saveCvFrame(handler->cvHandler);
}

// Handle frame with GPU
bool handleFrameGpu(
    VideoHandler* handler,
    ImageFilter filterKind,  // Updated enum
    const ImageFilterSettings& filterSettings,  // Updated struct
    ImageTransform transformKind  // Updated enum
) {
    if (!handler || !handler->currFrameBuffer) {
        return false;
    }
    
    // Transfer input to GPU
    GPU_VERIFY_CALL(cudaMemcpy(handler->gpuInputBuffer, handler->currFrameBuffer, 
                                handler->bufferSize, cudaMemcpyHostToDevice));
    
    // Apply transform if specified
    if (transformKind != ImageTransform::NO_TRANSFORM) {
        performTransformation(
            handler->gpuInputBuffer,
            handler->gpuOutputBuffer,
            transformKind,
            handler->imgWidth,
            handler->imgHeight,
            handler->imgChannels
        );
        
        // Exchange buffers
        unsigned char* swapTemp = handler->gpuInputBuffer;
        handler->gpuInputBuffer = handler->gpuOutputBuffer;
        handler->gpuOutputBuffer = swapTemp;
    }
    
    // Apply filter if specified
    if (filterKind != ImageFilter::NO_FILTER) {
        performSpecialFilter(
            handler->gpuInputBuffer,
            handler->gpuOutputBuffer,
            filterKind,
            filterSettings,
            handler->imgWidth,
            handler->imgHeight,
            handler->imgChannels
        );
    } else {
        // Copy directly if no filter
        GPU_VERIFY_CALL(cudaMemcpy(handler->gpuOutputBuffer, handler->gpuInputBuffer, 
                                    handler->bufferSize, cudaMemcpyDeviceToDevice));
    }
    
    // Transfer back to host
    GPU_VERIFY_CALL(cudaMemcpy(handler->outFrameBuffer, handler->gpuOutputBuffer, 
                                handler->bufferSize, cudaMemcpyDeviceToHost));
    
    return true;
}

// Shutdown input
void shutdownInput(VideoHandler* handler) {
    if (!handler || !handler->cvHandler) {
        return;
    }
    
    shutdownCvInput(handler->cvHandler);
}