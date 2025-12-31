# Camera Geometry and Calibration from Scratch

## Overview
This project implements fundamental camera geometry algorithms from scratch, focusing on both 3D rotation modeling and camera calibration. The work emphasizes mathematical formulation and controlled use of OpenCV to recover camera parameters from geometric constraints.

The project consists of **two independent tasks**.

---

## Task 1: Rotation Matrix Computation
Implements rotation matrix construction and inversion using Euler angles.

**Key details:**
- Sequential rotations about multiple coordinate axes
- Forward rotation (world → camera)
- Inverse rotation (camera → world)
- Explicit matrix formulation without black-box helpers

---

## Task 2: Camera Calibration
Implements camera calibration using 3D–2D point correspondences from a checkerboard pattern.

**Key details:**
- Projection matrix estimation from world ↔ image coordinates
- Intrinsic parameter recovery (`fx`, `fy`, `ox`, `oy`)
- Extrinsic parameter estimation (rotation `R` and translation `T`)
- Linear formulation emphasizing geometric interpretation


