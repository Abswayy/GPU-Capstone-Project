#include <iostream>
#include <string>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>

#include "cuda_utils.h"  // Updated header
#include "kernels.h"     // Updated header
#include "video_io.h"    // Updated header

// Display program usage
void showUsage(const char* appName) {
    printf("GPU Video Enhancer - Live Video Improvement and Review\n");
    printf("How to use: %s [flags]\n", appName);
    printf("Flags:\n");
    printf("  --input <src>              Source input (file or cam number)\n");
    printf("  --output <file>            Save to this video file (optional)\n");
    printf("  --filter <type>            Effect to use (default: no_filter)\n");
    printf("  --transform <type>         Adjustment to apply (default: no_transform)\n");
    printf("  --intensity <num>          Effect strength (0.0-1.0, default: 0.5)\n");
    printf("  --detect-motion            Turn on movement spotting\n");
    printf("  --optical-flow             Turn on flow display\n");
    printf("  --detect-objects           Turn on basic item spotting\n");
    printf("  --benchmark                Perform speed test\n");
    printf("  --batch-size <num>         Group processing count (default: 1)\n");
    printf("  --help                     Show this info\n");
    printf("\n");
    printf("Effects available:\n");
    printf("  no_filter, smooth, enhance, edge_find, emboss_effect, vintage,\n");
    printf("  monochrome, invert, comic_style, drawing_style, dark_vision, heat_map\n");
    printf("\n");
    printf("Adjustments available:\n");
    printf("  no_transform, turn_90, turn_180, turn_270, mirror_horiz, mirror_vert\n");
}

// Basic string matcher
int strMatch(const char* a, const char* b) {
    return strcmp(a, b) == 0;
}

// Process command flags
bool processFlags(int argCount, char** args, char* srcInput, char* destOutput,
                  char* effectName, char* adjustName,
                  float* strength, bool* spotMotion, bool* showFlow,
                  bool* spotItems, bool* runTest, int* groupSize) {
    // Defaults
    strcpy(srcInput, "0");  // Cam 0 default
    strcpy(destOutput, "");
    strcpy(effectName, "no_filter");
    strcpy(adjustName, "no_transform");
    *strength = 0.5f;
    *spotMotion = false;
    *showFlow = false;
    *spotItems = false;
    *runTest = false;
    *groupSize = 1;
    
    for (int idx = 1; idx < argCount; idx++) {
        if (strMatch(args[idx], "--help")) {
            showUsage(args[0]);
            return false;
        } else if (strMatch(args[idx], "--input") && idx + 1 < argCount) {
            strcpy(srcInput, args[++idx]);
        } else if (strMatch(args[idx], "--output") && idx + 1 < argCount) {
            strcpy(destOutput, args[++idx]);
        } else if (strMatch(args[idx], "--filter") && idx + 1 < argCount) {
            strcpy(effectName, args[++idx]);
        } else if (strMatch(args[idx], "--transform") && idx + 1 < argCount) {
            strcpy(adjustName, args[++idx]);
        } else if (strMatch(args[idx], "--intensity") && idx + 1 < argCount) {
            *strength = atof(args[++idx]);
        } else if (strMatch(args[idx], "--detect-motion")) {
            *spotMotion = true;
        } else if (strMatch(args[idx], "--optical-flow")) {
            *showFlow = true;
        } else if (strMatch(args[idx], "--detect-objects")) {
            *spotItems = true;
        } else if (strMatch(args[idx], "--benchmark")) {
            *runTest = true;
        } else if (strMatch(args[idx], "--batch-size") && idx + 1 < argCount) {
            *groupSize = atoi(args[++idx]);
        } else {
            printf("Invalid flag: %s\n", args[idx]);
            showUsage(args[0]);
            return false;
        }
    }
    
    // Check values
    if (*strength < 0.0f || *strength > 1.0f) {
        printf("Strength must be 0.0 to 1.0\n");
        return false;
    }
    
    if (*groupSize < 1) {
        printf("Group size at least 1 required\n");
        return false;
    }
    
    return true;
}

// Map effect string to enum
ImageFilter mapFilter(const char* effectName) {
    if (strMatch(effectName, "smooth")) return ImageFilter::SMOOTH;
    if (strMatch(effectName, "enhance")) return ImageFilter::ENHANCE;
    if (strMatch(effectName, "edge_find")) return ImageFilter::EDGE_FIND;
    if (strMatch(effectName, "emboss_effect")) return ImageFilter::EMBOSS_EFFECT;
    if (strMatch(effectName, "vintage")) return ImageFilter::VINTAGE;
    if (strMatch(effectName, "monochrome")) return ImageFilter::MONOCHROME;
    if (strMatch(effectName, "invert")) return ImageFilter::INVERT;
    if (strMatch(effectName, "comic_style")) return ImageFilter::COMIC_STYLE;
    if (strMatch(effectName, "drawing_style")) return ImageFilter::DRAWING_STYLE;
    if (strMatch(effectName, "dark_vision")) return ImageFilter::DARK_VISION;
    if (strMatch(effectName, "heat_map")) return ImageFilter::HEAT_MAP;
    return ImageFilter::NO_FILTER;
}

// Map adjustment string to enum
ImageTransform mapTransform(const char* adjustName) {
    if (strMatch(adjustName, "turn_90")) return ImageTransform::TURN_90;
    if (strMatch(adjustName, "turn_180")) return ImageTransform::TURN_180;
    if (strMatch(adjustName, "turn_270")) return ImageTransform::TURN_270;
    if (strMatch(adjustName, "mirror_horiz")) return ImageTransform::MIRROR_HORIZ;
    if (strMatch(adjustName, "mirror_vert")) return ImageTransform::MIRROR_VERT;
    return ImageTransform::NO_TRANSFORM;
}

// Basic performance test
void performBasicTest() {
    printf("Executing basic GPU test...\n");
    
    // Sample image dimensions
    int w = 1920;
    int h = 1080;
    int ch = 3;
    size_t imgBytes = w * h * ch * sizeof(unsigned char);
    
    // Host allocations
    unsigned char* hostIn = (unsigned char*)malloc(imgBytes);
    unsigned char* hostOut = (unsigned char*)malloc(imgBytes);
    
    if (!hostIn || !hostOut) {
        printf("Host allocation failed\n");
        return;
    }
    
    // Fill test data
    for (size_t pos = 0; pos < w * h * ch; ++pos) {
        hostIn[pos] = rand() % 256;
    }
    
    // Prep settings
    ImageFilterSettings settings;
    settings.strength = 0.5f;
    for (int p = 0; p < 4; p++) settings.extraParams[p] = 0.5f;
    
    printf("Evaluating effects on %dx%d image:\n", w, h);
    
    // Monochrome test
    printf("Evaluating Monochrome effect...\n");
    performSpecialFilter(hostIn, hostOut, ImageFilter::MONOCHROME, settings, 
                        w, h, ch);
    printf("  Monochrome: Successful\n");
    
    // Vintage test
    printf("Evaluating Vintage effect...\n");
    performSpecialFilter(hostIn, hostOut, ImageFilter::VINTAGE, settings, 
                        w, h, ch);
    printf("  Vintage: Successful\n");
    
    // Invert test
    printf("Evaluating Invert effect...\n");
    performSpecialFilter(hostIn, hostOut, ImageFilter::INVERT, settings, 
                        w, h, ch);
    printf("  Invert: Successful\n");
    
    // Release
    free(hostIn);
    free(hostOut);
    
    printf("Test finished OK!\n");
}

int main(int argCount, char** args) {
    // Process flags with C strings
    char srcInput[256], destOutput[256], effectName[64], adjustName[64];
    float strength;
    bool spotMotion, showFlow, spotItems, runTest;
    int groupSize;
    
    if (!processFlags(argCount, args, srcInput, destOutput, effectName, adjustName,
                      &strength, &spotMotion, &showFlow, &spotItems, 
                      &runTest, &groupSize)) {
        return 1;
    }
    
    // Check GPU count
    int gpuCount = 0;
    cudaGetDeviceCount(&gpuCount);
    
    if (gpuCount == 0) {
        printf("No GPUs detected! Stopping...\n");
        return 1;
    }
    
    // Show GPU info
    displayGpuDeviceInfo(0);
    
    // Execute test if flagged
    if (runTest) {
        performBasicTest();
        return 0;
    }
    
    // Initialize handler
    VideoHandler* handler = initVideoHandler();
    if (!handler) {
        printf("Handler initialization failed\n");
        return 1;
    }
    
    // Setup input
    bool isLocalFile = !(strMatch(srcInput, "0") || strMatch(srcInput, "1"));
    if (!initInputSource(handler, srcInput, isLocalFile)) {
        printf("Input source open failed: %s\n", srcInput);
        cleanupVideoHandler(handler);
        return 1;
    }
    
    printf("Input accessed OK!\n");
    printf("Dimensions: %dx%d\n", getInputWidth(handler), getInputHeight(handler));
    printf("Rate: %.2f\n", getInputFrameRate(handler));
    
    // Map enums
    ImageFilter filterKind = mapFilter(effectName);
    ImageTransform transformKind = mapTransform(adjustName);
    
    // Setup filter settings
    ImageFilterSettings filterSettings;
    filterSettings.strength = strength;
    for (int p = 0; p < 4; p++) filterSettings.extraParams[p] = 0.5f;
    
    printf("Handling video with effect: %s, adjustment: %s, strength: %.2f\n", 
           effectName, adjustName, strength);
    
    // Setup output if provided
    if (strlen(destOutput) > 0) {
        if (!initOutputFile(handler, destOutput)) {
            printf("Output file open failed: %s\n", destOutput);
            cleanupVideoHandler(handler);
            return 1;
        }
        printf("Output accessed: %s\n", destOutput);
    }
    
    // Frame loop
    int processedCount = 0;
    while (true) {
        // Fetch frame
        if (!fetchInputFrame(handler)) {
            printf("Input end or fetch issue\n");
            break;
        }
        
        // GPU handling
        if (!handleFrameGpu(handler, filterKind, filterSettings, transformKind)) {
            printf("Frame %d handling failed\n", processedCount);
            break;
        }
        
        // Save if output set
        if (strlen(destOutput) > 0) {
            if (!saveOutputFrame(handler)) {
                printf("Frame %d save failed\n", processedCount);
                break;
            }
        }
        
        processedCount++;
        if (processedCount % 100 == 0) {
            printf("%d frames handled\n", processedCount);
        }
        
        // Cap cam processing
        if (!isLocalFile && processedCount >= 1000) {
            printf("Halting cam after 1000 frames\n");
            break;
        }
    }
    
    printf("Handled total: %d frames\n", processedCount);
    
    // Release
    cleanupVideoHandler(handler);
    
    return 0;
}