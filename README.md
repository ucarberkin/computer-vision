# Panorama Stitching

Automatic panorama construction from multiple overlapping images. Implements a full pipeline: SIFT feature detection, Lowe's ratio test for correspondence filtering, RANSAC homography estimation, perspective warping, and mask-based image blending.

## Framework

```
Input images (3 overlapping views for image set 1; 2 for image set 2)
    -> SIFT keypoint detection + descriptor extraction
    -> BFMatcher with Lowe's ratio test (correspondence filtering)
    -> RANSAC homography estimation (robust outlier rejection)
    -> Perspective warp into common coordinate frame
    -> Mask-based blending (normalize by overlap count)
    -> Final panorama (grayscale + color)
```

## Results

### Image Set 1
![Panorama Color](results/panorama_color.jpg)

### Image Set 2
![Extra Panorama Color](results/extra_panorama_color.jpg)

## Methodology

1. **Feature detection**: SIFT extracts scale- and rotation-invariant keypoints (~800-900 per image)
2. **Feature matching**: Brute-force matching with Lowe's ratio test (threshold 0.8) filters ambiguous correspondences
3. **Homography estimation**: RANSAC fits a projective transformation while rejecting outlier matches. Multiple thresholds (1.0, 3.0, 5.0, 10.0) are compared to analyze sensitivity
4. **Warping**: Left and right images are warped into the center image's coordinate frame via the estimated homographies
5. **Blending**: Overlapping regions are averaged by pixel-wise mask counts to reduce seams

The full process with intermediate visualizations (keypoints, correspondences, RANSAC filtering, individual warps) is documented in the [notebook PDF](panorama_stitching.pdf) and the [Jupyter notebook](panorama_stitching.ipynb).

## Setup

```bash
pip install -r requirements.txt
```

Requires Python 3.8+ with OpenCV, NumPy, scikit-image, matplotlib, and imageio.

## Usage

Open and run `panorama_stitching.ipynb`. Input images are in the `images/` directory.
