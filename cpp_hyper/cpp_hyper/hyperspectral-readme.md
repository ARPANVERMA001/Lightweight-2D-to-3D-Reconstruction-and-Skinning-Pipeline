# Hyperspectral Inverse Skinning

A novel approach for Inverse Linear Blend Skinning (LBS) using hyperspectral techniques. This project retrieves skinning weights and transformations for mesh animation sequences by reframing inverse skinning as a high-dimensional optimization problem.

## Overview

The method minimizes reconstruction errors by optimizing the smallest-volume simplex in transformation space. It leverages hyperspectral techniques to handle the high-dimensional nature of the problem and provides robust solutions for animation rigging.

## Authors

- **Arpan Verma** - [arpan22105@iiitd.ac.in](mailto:arpan22105@iiitd.ac.in)
- **Harshit Gupta** - [harshit22209@iiitd.ac.in](mailto:harshit22209@iiitd.ac.in)

## Key Features

- **Per-Vertex Transformation Estimation**: High-dimensional optimization for vertex transformations
- **Flat Optimization**: Lower-dimensional handle flat optimization
- **Minimum Volume Enclosing Simplex**: Computes enclosing simplex for LBS rig handles
- **Integration with Ceres Solver**: Robust numerical optimization
- **Hyperspectral Data Handling**: High-dimensional Euclidean space formulation

## Technical Details

### Per-Vertex Transformation
The algorithm estimates transformations for each vertex by solving:

```
xi = argmin_x ∑_{j ∈ {i} ∪ N(i)} ||V_j x - v_j'||²
```

Where:
- Vi: Transformation matrix
- vi': Deformed vertex positions

### Flat Optimization
Minimizes projection error between vertex flats and a shared handle flat:

```
argmin_{p, B} ∑_i min_{z_i} ||V_i(p + B z_i) - v_i'||²
```

Where:
- p: Point on the flat
- B: Flat's spanning directions

## Prerequisites

### Dependencies
- Ceres Solver
- Eigen3
- GLPK
- GLEW
- glfw3
- glm

### Installation

1. Install required libraries:
```bash
# For Ubuntu/Debian
sudo apt update
sudo apt install libceres-dev libeigen3-dev libglpk-dev libglew-dev libglfw3-dev libglm-dev
```

2. Build the project:
```bash
mkdir build && cd build
cmake ..
make
```

3. Run the executable:
```bash
./skinning
```

## Project Structure

```
hyperspectral-inverse-skinning/
├── src/
│   ├── main.cpp
│   ├── transformation.cpp
│   ├── optimization.cpp
│   └── visualization.cpp
├── include/
│   ├── transformation.h
│   ├── optimization.h
│   └── visualization.h
├── data/
│   ├── meshes/
│   └── poses/
├── results/
├── CMakeLists.txt
└── README.md
```

## Results

Our implementation has been tested on various mesh sequences, including:
- Cat model with 8 poses and 10 bones
- Achieved 43% cost reduction in optimization
- Successful transformation and pose adaptation

## Current Status

### Completed Milestones
- ✅ Framework Design
- ✅ Flat Optimization Implementation
- ✅ Hyperspectral Data Collection
- ✅ Initial Guess Code

### In Progress
- 🔄 Minimum Volume Enclosing Simplex
- 🔄 Skinning Weights Estimation

## Challenges and Solutions

1. **Data Acquisition**
   - Challenge: Consistent pose data collection
   - Solution: Implemented robust data preprocessing pipeline

2. **Visualization**
   - Challenge: Initial unclear images
   - Solution: Integrated Phong model for improved rendering

3. **Optimization**
   - Challenge: Convergence issues
   - Solution: Migrated to Ceres Solver






