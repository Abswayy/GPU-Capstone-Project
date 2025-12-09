#ifndef VIDEO_HANDLER_H
#define VIDEO_HANDLER_H

#include "kernels.h"  // Updated header

#ifdef __cplusplus
extern "C" {
#endif

// Opaque handle for video handler
typedef struct VideoHandler VideoHandler;

// Init and cleanup handler
VideoHandler* initVideoHandler();
void cleanupVideoHandler(VideoHandler* handler);

// Setup input source (local file or cam)
bool initInputSource(VideoHandler* handler, const char* src, bool isLocalFile);

// Setup output file
bool initOutputFile(VideoHandler* handler, const char* file);

// Input status check
bool isInputActive(VideoHandler* handler);

// Retrieve input details
int getInputWidth(VideoHandler* handler);
int getInputHeight(VideoHandler* handler);
int getInputChannels(VideoHandler* handler);
double getInputFrameRate(VideoHandler* handler);
int getInputFrameTotal(VideoHandler* handler);
int getInputFrameCurrent(VideoHandler* handler);

// Frame operations
bool fetchInputFrame(VideoHandler* handler);
bool saveOutputFrame(VideoHandler* handler);

// GPU frame handling
bool handleFrameGpu(
    VideoHandler* handler,
    ImageFilter filterKind,
    const ImageFilterSettings& filterSettings,
    ImageTransform transformKind
);

// Shutdown input
void shutdownInput(VideoHandler* handler);

#ifdef __cplusplus
}
#endif

#endif // VIDEO_HANDLER_H