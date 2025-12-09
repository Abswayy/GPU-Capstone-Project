#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <string>
#include <cstring>

// Hidden CV handler class
class CvHandler {
public:
    cv::VideoCapture inputCapture;
    cv::VideoWriter outputWriter;
    cv::Mat currMat;
    cv::Mat outMat;
    int imgWidth;
    int imgHeight;
    int imgChannels;
    double frameRate;
    int frameTotal;
    int frameCurrent;
    
    CvHandler() : imgWidth(0), imgHeight(0), imgChannels(0), frameRate(0.0), frameTotal(0), frameCurrent(0) {}
    
    ~CvHandler() {
        shutdown();
    }
    
    void shutdown() {
        if (inputCapture.isOpened()) {
            inputCapture.release();
        }
        if (outputWriter.isOpened()) {
            outputWriter.release();
        }
        currMat.release();
        outMat.release();
    }
};

// C wrapper functions
extern "C" {

void* initCvHandler() {
    try {
        return new CvHandler();
    } catch (...) {
        return nullptr;
    }
}

void cleanupCvHandler(void* handler) {
    if (handler) {
        delete static_cast<CvHandler*>(handler);
    }
}

bool initCvInput(void* handler, const char* inputSrc, bool isLocalFile) {
    CvHandler* hdl = static_cast<CvHandler*>(handler);
    if (!hdl) return false;
    
    try {
        hdl->shutdown();
        
        if (isLocalFile) {
            if (!hdl->inputCapture.open(std::string(inputSrc))) {
                std::cerr << "Failed to access video file: " << inputSrc << std::endl;
                return false;
            }
        } else {
            int camNum = std::atoi(inputSrc);
            if (!hdl->inputCapture.open(camNum)) {
                std::cerr << "Failed to access camera device: " << camNum << std::endl;
                return false;
            }
        }
        
        // Fetch input details
        hdl->imgWidth = static_cast<int>(hdl->inputCapture.get(cv::CAP_PROP_FRAME_WIDTH));
        hdl->imgHeight = static_cast<int>(hdl->inputCapture.get(cv::CAP_PROP_FRAME_HEIGHT));
        hdl->frameRate = hdl->inputCapture.get(cv::CAP_PROP_FPS);
        hdl->frameTotal = static_cast<int>(hdl->inputCapture.get(cv::CAP_PROP_FRAME_COUNT));
        hdl->frameCurrent = 0;
        
        // Determine channels from a sample frame
        cv::Mat sampleMat;
        if (hdl->inputCapture.read(sampleMat)) {
            hdl->imgChannels = sampleMat.channels();
            // Reset position if file
            if (isLocalFile) {
                hdl->inputCapture.set(cv::CAP_PROP_POS_FRAMES, 0);
            }
        } else {
            hdl->imgChannels = 3;  // Fallback to RGB
        }
        
        return true;
    } catch (const std::exception& err) {
        std::cerr << "Input initialization error: " << err.what() << std::endl;
        return false;
    }
}

bool initCvOutput(void* handler, const char* outputFile) {
    CvHandler* hdl = static_cast<CvHandler*>(handler);
    if (!hdl) return false;
    
    try {
        std::string fileStr(outputFile);
        std::string fileExt = fileStr.substr(fileStr.find_last_of('.') + 1);
        
        int codec;
        if (fileExt == "mp4") {
            codec = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
        } else if (fileExt == "avi") {
            codec = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');
        } else {
            codec = cv::VideoWriter::fourcc('X', 'V', 'I', 'D');
        }
        
        double rate = hdl->frameRate > 0 ? hdl->frameRate : 30.0;
        
        return hdl->outputWriter.open(fileStr, codec, rate, 
                                     cv::Size(hdl->imgWidth, hdl->imgHeight), true);
    } catch (const std::exception& err) {
        std::cerr << "Output initialization error: " << err.what() << std::endl;
        return false;
    }
}

bool isCvInputActive(void* handler) {
    CvHandler* hdl = static_cast<CvHandler*>(handler);
    return hdl ? hdl->inputCapture.isOpened() : false;
}

int getCvImgWidth(void* handler) {
    CvHandler* hdl = static_cast<CvHandler*>(handler);
    return hdl ? hdl->imgWidth : 0;
}

int getCvImgHeight(void* handler) {
    CvHandler* hdl = static_cast<CvHandler*>(handler);
    return hdl ? hdl->imgHeight : 0;
}

int getCvImgChannels(void* handler) {
    CvHandler* hdl = static_cast<CvHandler*>(handler);
    return hdl ? hdl->imgChannels : 0;
}

double getCvFrameRate(void* handler) {
    CvHandler* hdl = static_cast<CvHandler*>(handler);
    return hdl ? hdl->frameRate : 0.0;
}

int getCvFrameTotal(void* handler) {
    CvHandler* hdl = static_cast<CvHandler*>(handler);
    return hdl ? hdl->frameTotal : 0;
}

int getCvFrameCurrent(void* handler) {
    CvHandler* hdl = static_cast<CvHandler*>(handler);
    return hdl ? hdl->frameCurrent : 0;
}

bool fetchCvFrame(void* handler) {
    CvHandler* hdl = static_cast<CvHandler*>(handler);
    if (!hdl || !hdl->inputCapture.isOpened()) {
        return false;
    }
    
    bool ok = hdl->inputCapture.read(hdl->currMat);
    if (ok) {
        hdl->frameCurrent++;
    }
    return ok;
}

bool saveCvFrame(void* handler) {
    CvHandler* hdl = static_cast<CvHandler*>(handler);
    if (!hdl || !hdl->outputWriter.isOpened()) {
        return false;
    }
    
    if (hdl->outMat.empty()) {
        return false;
    }
    
    hdl->outputWriter.write(hdl->outMat);
    return true;
}

void shutdownCvInput(void* handler) {
    CvHandler* hdl = static_cast<CvHandler*>(handler);
    if (hdl) {
        hdl->shutdown();
    }
}

unsigned char* fetchCurrentFrameBuffer(void* handler) {
    CvHandler* hdl = static_cast<CvHandler*>(handler);
    if (!hdl || hdl->currMat.empty()) {
        return nullptr;
    }
    return hdl->currMat.data;
}

unsigned char* fetchOutputFrameBuffer(void* handler) {
    CvHandler* hdl = static_cast<CvHandler*>(handler);
    if (!hdl || hdl->outMat.empty()) {
        return nullptr;
    }
    return hdl->outMat.data;
}

void updateOutputFrameBuffer(void* handler, unsigned char* buffer) {
    CvHandler* hdl = static_cast<CvHandler*>(handler);
    if (!hdl || !buffer) {
        return;
    }
    
    // Initialize out mat if needed
    if (hdl->outMat.empty()) {
        hdl->outMat = cv::Mat(hdl->imgHeight, hdl->imgWidth, CV_8UC(hdl->imgChannels));
    }
    
    // Transfer buffer data
    size_t bufferLen = hdl->imgWidth * hdl->imgHeight * hdl->imgChannels;
    std::memcpy(hdl->outMat.data, buffer, bufferLen);
}

} // extern "C"