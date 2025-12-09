# GPU-Accelerated Video Handler

A high-performance tool for live video manipulation, improvement, and examination powered by GPU technology.

## Introduction

This final project showcases the capabilities of parallel computing on graphics hardware for immediate video handling. The tool utilizes advanced GPU features to execute demanding computations on video data simultaneously, delivering notable efficiency gains over traditional processor methods.

Main capabilities:

- Various accelerated visual adjustments (smoothing, enhancement, boundary identification, relief effect, etc.)
- Instantaneous video manipulation from recordings or live feeds
- Sophisticated graphical simulations such as heat mapping and low-light viewing
- Movement identification across sequential images
- Flow pattern display
- Basic item recognition
- Group handling for improved efficiency
- Efficiency evaluation utilities

## Prerequisites

- NVIDIA Parallel Computing Platform (version 11.0 or later)
- Computer Vision Library 4.x
- Modern C++ compiler (C++17 support)
- Build System 3.10 or higher

## Setup

### Method 1: Compiling from Code

1. Duplicate the project:
   ```bash
   git clone <project-link>
   cd Parallel-Computing-Final-Project
   ```

2. Set up a compilation folder and build:
   ```bash
   ./compile-script.sh
   ```
   
   Alternatively, step-by-step:
   ```bash
   mkdir -p compilation-dir
   cd compilation-dir
   cmake ..
   make -j$(nproc)
   ```

### Method 2: Container-Based

A container configuration file is included for isolated building and running:

```bash
docker build -t gpu-video-handler .
docker run --gpus all -it --rm gpu-video-handler
```

## Operation

The tool accepts multiple command-line parameters:

```bash
./video_handler [parameters]
```

### Parameters

- `--input <origin>`: Origin data (recording path or feed number)
- `--output <file>`: Result recording file (if needed)
- `--filter <adjustment-type>`: Adjustment to implement (default: no_filter)
- `--transform <modification>`: Modification to implement (default: no_transform)
- `--intensity <level>`: Adjustment level (0.0-1.0, default: 0.5)
- `--detect-motion`: Activate movement spotting
- `--optical-flow`: Activate flow pattern display
- `--detect-objects`: Activate basic item spotting
- `--benchmark`: Execute efficiency test
- `--batch-size <count>`: Group handling count (default: 1)
- `--help`: Show operation guide

### Supported Adjustments

- `no_filter`: No adjustment
- `smooth`: Softening effect
- `enhance`: Clarity boost
- `edge_find`: Boundary highlighting
- `emboss_effect`: Relief simulation
- `vintage`: Aged tone
- `monochrome`: Color removal
- `invert`: Tone reversal
- `comic_style`: Illustrated look
- `drawing_style`: Hand-drawn appearance
- `dark_vision`: Low-light simulation
- `heat_map`: Temperature visualization

### Supported Modifications

- `no_transform`: No modification
- `turn_90`: Turn by 90 degrees
- `turn_180`: Turn by 180 degrees
- `turn_270`: Turn by 270 degrees
- `mirror_horiz`: Horizontal reflection
- `mirror_vert`: Vertical reflection

### Samples

Handle a recording with softening:
```bash
./video_handler --input source_recording.mp4 --output result_recording.mp4 --filter smooth --intensity 0.7
```

Apply live feed with temperature visualization:
```bash
./video_handler --input 0 --filter heat_map --intensity 0.8
```

Spot movement with boundary highlighting:
```bash
./video_handler --input source_recording.mp4 --filter edge_find --detect-motion
```

Perform efficiency test:
```bash
./video_handler --input source_recording.mp4 --benchmark
```

## Functionality Explanation

### Accelerated Handling Sequence

1. **Data Capture**: Images are obtained from recordings or live sources.
2. **GPU Upload**: Image information is moved to graphics memory.
3. **Concurrent Execution**: Each element is handled simultaneously via numerous parallel threads.
4. **Adjustment Implementation**: Numerical transformations alter element values.
5. **Outcome Download**: Handled images are retrieved to main memory.
6. **Presentation/Saving**: Outcomes are shown immediately and/or stored.

### Parallel Functions

The project features multiple parallel functions for diverse image tasks:

- **Matrix Application Function**: Implements matrix-based adjustments (softening, clarity, boundaries, etc.)
- **Tone Alteration Functions**: Applies specific color changes (aged, monochrome, etc.)
- **Movement Spotting Function**: Determines variations between images
- **Unique Effect Functions**: Creates advanced simulations like temperature and low-light

## Efficiency

The hardware acceleration yields substantial gains over standard implementations:

- Handling full HD content (1920x1080) instantly at over 30 frames/second
- 10-20 times faster than comparable standard methods
- Optimized group handling for better output

Results vary based on hardware specs, adjustment intricacy, and image size.

## Permissions

[Open Source License](PERMISSIONS)

