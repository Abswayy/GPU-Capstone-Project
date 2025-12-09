# GPU Video Handler Configuration Manual

This guide outlines steps for configuring and operating the GPU Video Handler tool.

## Hardware/Software Needs

- NVIDIA accelerator with parallel compute capability (5.0 or above suggested)
- NVIDIA Development Kit 11.0 or higher
- Image Processing Library 4.x with accelerator integration
- Updated C++ compiler (handles 17 specs)
- Configuration Tool 3.10 or later

## Configuration Procedure

### 1. Set Up Development Kit

#### Linux Variants:
```bash
# Include NVIDIA sources
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /"

# Set up kit
sudo apt update
sudo apt install cuda-toolkit-11-8
```
Apple Systems:
Obtain and set up from: https://developer.nvidia.com/cuda-downloads
Microsoft Systems:
Obtain and set up from: https://developer.nvidia.com/cuda-downloads
2. Set Up Library with Accelerator
Linux Variants:
```bash
# Set up required components
sudo apt update
sudo apt install -y build-essential cmake git pkg-config libgtk-3-dev \
    libavcodec-dev libavformat-dev libswscale-dev libv4l-dev \
    libxvidcore-dev libx264-dev libjpeg-dev libpng-dev libtiff-dev \
    gfortran openexr libatlas-base-dev python3-dev python3-numpy \
    libtbb2 libtbb-dev libdc1394-dev

# Obtain library and extras
cd ~
git clone https://github.com/opencv/opencv.git
git clone https://github.com/opencv/opencv_contrib.git

# Prepare assembly
cd opencv
mkdir assembly && cd assembly

# Set options
cmake -D CMAKE_BUILD_TYPE=RELEASE \
      -D CMAKE_INSTALL_PREFIX=/usr/local \
      -D OPENCV_EXTRA_MODULES_PATH=~/opencv_contrib/modules \
      -D WITH_CUDA=ON \
      -D WITH_CUBLAS=ON \
      -D WITH_TBB=ON \
      -D WITH_V4L=ON \
      -D WITH_GTK=ON \
      -D BUILD_TESTS=OFF \
      -D BUILD_PERF_TESTS=OFF \
      -D BUILD_EXAMPLES=OFF ..

# Assemble and deploy
make -j$(nproc)
sudo make install
sudo ldconfig
```
Apple Systems:
```bash
# Set up via package manager
brew install opencv

# Note: Managed version might lack accelerator. For full support, assemble from code as above.
```
Microsoft Systems:
Obtain ready builds from: https://opencv.org/releases/
or assemble following library guides.
3. Obtain and Assemble Project
```bash
# Obtain codebase
git clone <repo-link>
cd Parallel-Computing-Final-Project

# Assemble with script
./assemble.sh

# Or step-by-step:
mkdir -p assembly
cd assembly
cmake ..
make -j$(nproc)  # Microsoft: cmake --build . --config Release
```
Container Option
For container use, a config file is available:
```bash
# Assemble image
docker build -t gpu-video-handler .

# Operate container with accelerator
docker run --gpus all -it --rm gpu-video-handler --help

# Handle stream by mounting storage:
docker run --gpus all -it --rm -v /path/to/streams:/streams gpu-video-handler --input /streams/source.mp4 --output /streams/result.mp4 --filter heat_map
```
Confirmation Steps
To confirm proper operation:
```bash
./assembly/video_handler --help
./assembly/video_handler --benchmark
./assembly/video_handler --input sample.mp4 --filter smooth
```
## Issue Resolution
Accelerator Problems

Issue: No accelerators detected
Check drivers: nvidia-smi
Verify kit: nvcc --version

Kit version conflict
Match kit version to library build


## Library Problems

Can't locate library components
Confirm setup: pkg-config --modversion opencv4
Verify paths: echo $LD_LIBRARY_PATH

Execution issue on device features
Confirm library has accelerator: opencv_version --verbose


## Assembly Problems

Tool can't locate kit or library
Define variables:Bashexport OpenCV_DIR=/path/to/library/setup
export CUDA_HOME=/usr/local/cuda

Build failures
Verify compiler: g++ --version
Confirm C++17 support
