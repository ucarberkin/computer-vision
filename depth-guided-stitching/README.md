# Depth-Guided Panoramic Image Stitching

Berkin Ucar

---

## Overview

Standard panoramic stitching pipelines find correspondences between overlapping images, estimate homographies, and blend pixels together, yet the seam placement is typically a naive center-cut through the overlap region. This works well when the overlap is texturally uniform, but produces visible ghosting and double-edge artifacts when the seam crosses depth boundaries (e.g., a foreground tree in front of a distant building).

This project uses **monocular depth estimation** to address this problem in two ways:

1. **Depth-guided seam selection:** a dynamic programming path through the overlap region whose cost function penalizes both image gradient magnitude (standard) and depth disagreement between the two views. The result is a non-linear seam that naturally routes around foreground-background boundaries.

2. **Depth-consistency evaluation metric:** a quantitative measure of seam quality that computes normalized depth disagreement in a band around the seam. This provides a geometry-aware complement to traditional color-based metrics and requires no ground truth.

Both components are evaluated across 10 scenes (outdoor and indoor) using two depth models: **Depth Anything v2 Large** and **Marigold**.

---

## Method

### Stitching Pipeline

The pipeline generalizes a standard SIFT + RANSAC stitching approach to N images with global optimization:

1. **All-pairs matching**: every image pair (i, j) is matched via SIFT descriptors, Lowe's ratio test (0.8), and RANSAC (threshold 5.0 px). Pairs with fewer than 80 inliers are discarded. This eliminates the fragility of sequential (adjacent-only) matching, where a single weak pair breaks the entire chain.

2. **BFS homography initialization**: valid pairs form a connectivity graph. BFS from the reference image (center of the sequence) composes pairwise homographies along shortest paths to produce an initial H_to_ref for every reachable image. Disconnected images are dropped automatically.

3. **Bundle adjustment**: all homographies are jointly refined via Levenberg-Marquardt (`scipy.optimize.least_squares`), minimizing reprojection error across all valid pairs simultaneously. Each homography is parameterized as 8 free values (normalized so H[2,2] = 1); the reference image is anchored to identity.

4. **Canvas layout and warping**: warped image corners determine the canvas dimensions. All images are warped into the shared coordinate frame. Black borders are auto-cropped.

5. **Seam-based compositing**: images are composited left-to-right. At each step, a pluggable seam function partitions the overlap between the accumulated canvas and the new image. The baseline uses a straight vertical center-cut; the depth-guided variant uses the DP seam described below.

### Depth Estimation

Two monocular depth models are compared:

| Model | Architecture | Parameters | Speed (MPS) |
|-------|-------------|------------|-------------|
| **Depth Anything v2 Large** | DINOv2 + DPT head | 335M | ~3s / image |
| **Marigold** | Latent diffusion (Stable Diffusion backbone) | ~1B | ~60s / image |

Both produce relative (not metric) depth maps, which are independently normalized to [0, 1]. Depth maps are cached to `.npz` files after first inference.

### Global Depth Scale Alignment

Monocular depth is inherently scale-and-shift ambiguous: each image's depth map lives in its own affine coordinate system. Before depth values can be compared across images, they must be aligned.

After warping all depth maps into the panorama canvas frame (using the same homographies as RGB warping), we solve for per-image affine parameters (s_i, t_i) that minimize depth disagreement across all overlap regions simultaneously:

```
min_{s_i, t_i}  sum_{(i,j)}  sum_{p in overlap_ij}  (s_i * d_i(p) + t_i  -  s_j * d_j(p) - t_j)^2
```

This is linear in the unknowns and reduces to a single `Ax = b` system solved in closed form via `np.linalg.lstsq`. The reference image is anchored at s = 1, t = 0 to resolve the global scale ambiguity.

### Depth-Guided DP Seam

In the overlap region between two adjacent warped images, we define a per-pixel cost:

```
cost(r, c) = alpha * grad_cost(r, c) + (1 - alpha) * depth_diff_cost(r, c)
```

where:
- **grad_cost** is the Sobel gradient magnitude of the average of both images (penalizes cutting through visible edges)
- **depth_diff_cost** is `|aligned_depth_A - aligned_depth_B|` (penalizes cutting where the two views disagree on scene geometry)

Both terms are independently normalized to [0, 1] within the overlap before mixing, so alpha controls the relative weight (default 0.5).

The optimal seam is found via dynamic programming over rows:

```
dp[r, c] = cost[r, c] + min(dp[r-1, c-1], dp[r-1, c], dp[r-1, c+1])
```

This produces a one-pixel-wide 8-connected path (one column per row) that can shift left, straight, or right at each step. Traceback from the minimum of the last row gives the full seam path. Pixels left of the seam come from image A; pixels at or right of the seam come from image B.

### Evaluation Metrics

Three complementary metrics evaluate seam quality. All are **lower-is-better** and require no ground-truth reference:

| Metric | What it measures |
|--------|-----------------|
| **Gradient energy** | Mean image gradient magnitude in a +/-10px band around the seam. High values indicate the seam cuts through a visually busy region. |
| **Color discontinuity** | Mean absolute RGB difference between the pixel immediately left and immediately right of the seam. Directly measures the visible color jump at the cut. |
| **Depth boundary score** | Depth disagreement across all label boundaries in the panorama, normalized per adjacent-pair using a local affine fit. |

The depth boundary score is **topology-agnostic**: it operates on the label map (which source image owns each pixel) rather than explicit seam paths, naturally handling N-image overlaps and Y-junctions. For each adjacent label pair:

1. Identify the boundary and dilate it into a +/-10px evaluation band
2. Fit a local affine mapping `s * d_i + t ~ d_j` on overlap pixels **outside** the band (out-of-sample fitting avoids circularity with the global alignment)
3. Evaluate `|s * d_i + t - d_j| / depth_range` at band pixels

At junction pixels (boundary of 3+ labels), scores from all adjacent pairs are averaged uniformly.

---

## Datasets

All except `test-berkin` image sequences are from the [Image Stitching Dataset](https://github.com/visionxiang/Image-Stitching-Dataset) (Xiang et al., *Pattern Recognition*, 2018). Planar subsets were extracted to avoid wide-angle distortion inherent to planar homography projection.

| Subset | Images | Scene |
|--------|--------|-------|
| CMU0-1 | 10 | Outdoor campus quad |
| CMU0-2 | 12 | Outdoor campus, ivy building |
| CMU0-3 | 9 | Outdoor campus, flagpole |
| CMU1-1 | 3 | Outdoor, glass building |
| CMU1-2 | 3 | Outdoor campus plaza |
| CMU1-3 | 3 | Outdoor, brick buildings |
| CMU1-4 | 3 | Outdoor, trees/buildings |
| NSH-1 | 9 | Indoor atrium |
| NSH-2 | 9 | Indoor corridor |
| NSH-3 | 6 | Indoor, window grid |
| Flower | 4 | Outdoor, purple flower |

---

## Repository Structure

```
depth-guided-stitching/
├── README.md
├── tests.ipynb            # Single-scene development / debug notebook
├── experiments.ipynb      # All scenes: figures, tables, model comparison
├── src/
│   ├── pipeline.py        # All-pairs matching, BFS init, bundle adjustment,
│   │                      #   canvas layout, warping, seam compositing
│   ├── depth_utils.py     # Depth model loading, inference, warping,
│   │                      #   global scale alignment, caching
│   ├── seam.py            # DP seam finder (gradient + depth cost)
│   └── metrics.py         # Gradient energy, color discontinuity,
│                          #   depth boundary score
├── dataset/
│   └── project-set/       # Planar subsets for experiments
│       ├── test-CMU0/     # 3 subsets (10, 12, 9 images)
│       ├── test-CMU1/     # 4 subsets (3 images each)
│       └── test-NSH/      # 3 subsets (9, 9, 6 images)
└── outputs/
    └── experiments/       # All-scene outputs
        ├── panoramas/     # Baseline + depth-guided PNGs
        └── figures/       # Tables, charts, comparisons
```

---

## Getting Started

### Requirements

- Python 3.10+
- Apple Silicon recommended (MPS acceleration for depth inference)
- ~2 GB disk for depth model weights (downloaded automatically)

### Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Running

1. **Single-scene test** — open `tests.ipynb` and run top-to-bottom on a single scene (e.g., CMU0-2) to verify the full pipeline. Point `DATA` at any folder of JPEGs to try your own images.

2. **Full experiments** — open `experiments.ipynb` and run top-to-bottom. Sections 2-5 run the pipeline across all 10 scenes with both depth models. Sections 6-9 produce quantitative tables and qualitative figures. Depth maps are cached after first inference, so subsequent runs skip the expensive model forward passes.

---

## Key Design Decisions

**All-pairs matching over sequential chaining.** The initial pipeline matched only adjacent images and chained homographies through the sequence. This was fragile: a single weak pair (e.g., images from non-overlapping viewpoints placed adjacently by filename) would break the entire panorama. All-pairs matching with BFS initialization through the connectivity graph is robust to arbitrary image orderings and gracefully handles non-overlapping subsets.

**Joint global optimization at every stage.** Both homography estimation (bundle adjustment) and depth alignment (joint affine least-squares) solve overdetermined systems over all image pairs simultaneously, with a reference anchor to resolve ambiguity. This is fundamentally more stable than pairwise approaches that accumulate error along chains.

**Topology-agnostic depth metric.** The depth boundary score uses the label map rather than explicit seam paths. This means it naturally handles any panorama topology — 1D horizontal sweeps, N-image overlaps, and Y-junctions where three or more images meet — without special-casing.

**Out-of-sample affine fitting in the metric.** The depth boundary score fits its own per-pair affine normalization using overlap pixels *outside* the evaluation band. This avoids circularity with the global depth alignment (which uses *all* overlap pixels) and makes the metric independent of the alignment quality.

**Pluggable seam interface.** The compositing function accepts any `seam_fn(warped_a, warped_b, mask_a, mask_b) -> (assign_a, assign_b)`. The baseline center-cut, the depth-guided DP seam, and any future seam strategy (e.g., graph-cut) all share this interface.

---

## Limitations

- **Planar homography model**: assumes all scene points lie on a plane (or equivalently, pure camera rotation). Breaks down for indoor scenes with significant depth variation and wide baselines, producing extreme perspective distortion (visible on full-set NSH scenes).
- **Left-to-right compositing order**: the sequential compositing means seam placement depends on the order images are added. A global graph-cut formulation would produce better seams but at higher computational cost.
- **Monocular depth quality**: depth maps from monocular models are approximate and can disagree on fine structures. The affine alignment corrects global scale/shift but not local errors.
- **No exposure compensation**: brightness differences between images are not corrected, which can cause visible seams even when the geometric placement is good.

---

## Acknowledgments

- [Depth Anything v2](https://github.com/DepthAnything/Depth-Anything-V2) (Yang et al., 2024)
- [Marigold](https://github.com/prs-eth/Marigold) (Ke et al., 2024)
- [Image Stitching Dataset](https://github.com/visionxiang/Image-Stitching-Dataset) (Xiang et al., 2018)
