#ifndef IMAGE_KERNELS_H
#define IMAGE_KERNELS_H

#include <cuda_runtime.h>

// Types of image filters
enum class ImageFilter {
    NO_FILTER,
    SMOOTH,
    ENHANCE,
    EDGE_FIND,
    EMBOSS_EFFECT,
    VINTAGE,
    MONOCHROME,
    INVERT,
    COMIC_STYLE,
    DRAWING_STYLE,
    DARK_VISION,
    HEAT_MAP
};

// Types of image transformations
enum class ImageTransform {
    NO_TRANSFORM,
    TURN_90,
    TURN_180,
    TURN_270,
    MIRROR_HORIZ,
    MIRROR_VERT,
    DISTORT_PERSPECTIVE
};

// Params for perspective distortion
typedef struct {
    float originCoords[8]; // Origin coordinates (x1,y1,x2,y2,x3,y3,x4,y4)
    float targetCoords[8]; // Target coordinates (x1,y1,x2,y2,x3,y3,x4,y4)
} DistortPerspectiveInfo;

// Params for image filters
typedef struct {
    float strength;    // Overall strength level (0.0 - 1.0)
    float extraParams[4]; // Extra settings for certain filters
} ImageFilterSettings;

// Create various filter matrices
void createFilterMatrix(float* matrix, int matrixSize, ImageFilter filterKind, const ImageFilterSettings& settings);

// Wrapper for host-side convolution operation
void performConvolution(
    const unsigned char* hostInputImg,
    unsigned char* hostOutputImg,
    const float* hostMatrix,
    int matrixSize,
    int imgWidth,
    int imgHeight,
    int imgChannels,
    cudaStream_t gpuStream = 0
);

// Wrapper for host-side special filter applications
void performSpecialFilter(
    const unsigned char* hostInputImg,
    unsigned char* hostOutputImg,
    ImageFilter filterKind,
    const ImageFilterSettings& settings,
    int imgWidth,
    int imgHeight,
    int imgChannels,
    cudaStream_t gpuStream = 0
);

// Wrapper for host-side image transformations
void performTransformation(
    const unsigned char* hostInputImg,
    unsigned char* hostOutputImg,
    ImageTransform transformKind,
    int imgWidth,
    int imgHeight,
    int imgChannels,
    const void* transformSettings = nullptr,
    cudaStream_t gpuStream = 0
);

// Handle batch of frames (for video handling)
void handleFrameBatch(
    const unsigned char** hostInputFrames,
    unsigned char** hostOutputFrames,
    int frameCount,
    int frameWidth,
    int frameHeight,
    int frameChannels,
    ImageFilter filterKind,
    const ImageFilterSettings& settings,
    ImageTransform transformKind = ImageTransform::NO_TRANSFORM,
    const void* transformSettings = nullptr
);

// Detect movement between video frames
void findMotion(
    const unsigned char* prevFrame,
    const unsigned char* currFrame,
    unsigned char* motionOutput,
    int frameWidth,
    int frameHeight,
    float motionThreshold,
    cudaStream_t gpuStream = 0
);

// Calculate optical flow in frames
void calculateFlow(
    const unsigned char* prevFrame,
    const unsigned char* currFrame,
    float* flowX,
    float* flowY,
    int frameWidth,
    int frameHeight,
    cudaStream_t gpuStream = 0
);

// Create mask for object detection (basic version)
void createObjectMask(
    const unsigned char* inputFrame,
    unsigned char* maskOutput,
    int frameWidth,
    int frameHeight,
    float detectThreshold,
    cudaStream_t gpuStream = 0
);

#endif // IMAGE_KERNELS_H