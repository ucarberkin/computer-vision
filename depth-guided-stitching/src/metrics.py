"""
metrics.py — seam quality evaluation metrics.

Three complementary metrics:

  1. seam_gradient_energy      — mean image gradient magnitude in a band around
                                 the seam path (high → cuts through busy regions)
  2. seam_color_discontinuity  — mean absolute RGB difference across the seam
                                 boundary (high → visible color jump at cut)
  3. depth_boundary_score      — depth disagreement across all label boundaries,
                                 normalized per-pair using out-of-band overlap pixels
                                 (no dependence on global depth alignment)

All three are lower-is-better and require no ground-truth reference.

depth_boundary_score uses a label_map (which image owns each pixel) rather than
explicit seam paths, making it topology-agnostic: works for 1-D horizontal
panoramas, N-image overlaps, and 2-D mosaics with Y-junctions alike.
At junction pixels (boundary of 3+ labels), scores from all adjacent pairs are
averaged uniformly.
"""

import numpy as np
import cv2
from scipy.ndimage import binary_dilation


# ---------------------------------------------------------------------------
# Metric 1 — Gradient energy in band around seam
# ---------------------------------------------------------------------------

def seam_gradient_energy(pano, seam, band=10):
    """
    Mean image gradient magnitude in a ±band pixel window around the seam path.

    Args:
        pano:  (H, W, 3) uint8 RGB panorama
        seam:  (H,) int array — seam[r] is the seam column at row r
        band:  half-width of evaluation window in pixels

    Returns:
        float — mean gradient magnitude (lower = smoother region at seam)
    """
    gray = cv2.cvtColor(pano, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
    grad_mag = np.sqrt(gx ** 2 + gy ** 2)

    H, W = grad_mag.shape
    seam = np.asarray(seam)
    rows = np.arange(H)

    # Build a mask of all pixels within ±band of the seam column per row
    cols = np.arange(W)[None, :]          # (1, W)
    seam_2d = seam[:, None]               # (H, 1)
    band_mask = (cols >= np.maximum(seam_2d - band, 0)) & \
                (cols < np.minimum(seam_2d + band + 1, W))

    return float(grad_mag[band_mask].mean()) if band_mask.any() else 0.0


# ---------------------------------------------------------------------------
# Metric 2 — Color discontinuity across seam boundary
# ---------------------------------------------------------------------------

def seam_color_discontinuity(pano, seam):
    """
    Mean absolute RGB difference between the column immediately left of the
    seam and the column immediately right of it.

    Args:
        pano:  (H, W, 3) uint8 RGB panorama
        seam:  (H,) int array

    Returns:
        float — mean absolute per-channel difference (0–255 scale)
    """
    pano_f = pano.astype(np.float32)
    H, W = pano_f.shape[:2]
    seam = np.asarray(seam)
    rows = np.arange(H)

    valid = (seam > 0) & (seam < W - 1)
    if not valid.any():
        return 0.0

    r = rows[valid]
    c = seam[valid]
    diffs = np.abs(pano_f[r, c - 1] - pano_f[r, c + 1]).mean(axis=1)
    return float(diffs.mean())


# ---------------------------------------------------------------------------
# Metric 3 — Depth boundary consistency (label-map based, topology-agnostic)
# ---------------------------------------------------------------------------

def depth_boundary_score(label_map, warped_depths, band=10, n_fit_samples=5000,
                         n_eval_samples=5000):
    """
    Mean depth disagreement across all label boundaries in the panorama,
    normalized per label-pair using pixels outside the seam band.

    For each unique adjacent label pair (i, j):
      1. Build the seam band: pixels within `band` px of the i↔j boundary.
      2. Fit affine parameters (s_ij, t_ij) using overlap pixels OUTSIDE the
         seam band  →  s_ij * d_i(p) + t_ij ≈ d_j(p).
         These pixels are never used in evaluation (out-of-sample).
      3. Evaluate at seam band pixels:
         score(p) = |s_ij * d_i(p) + t_ij - d_j(p)| / local_depth_range(p)
    Junction pixels (on boundary of 3+ labels) receive the uniform average
    of all adjacent-pair scores.

    Args:
        label_map:     (H, W) int array — label_map[r,c] = index of source image
                       owning that pixel; -1 for unassigned pixels
        warped_depths: list of N float32 depth maps in canvas frame (raw, not
                       globally aligned — this function does its own per-pair
                       normalization to avoid circularity with global alignment)
        band:          seam band half-width in pixels
        n_fit_samples: max pixels sampled from fitting region per pair (for speed)

    Returns:
        float — mean normalized depth discontinuity across all boundary pixels
    """
    H, W = label_map.shape
    struct_1px = np.array([[0, 1, 0],
                            [1, 1, 1],
                            [0, 1, 0]], dtype=bool)
    struct_band = np.ones((2 * band + 1, 2 * band + 1), dtype=bool)

    # --- Discover all unique adjacent label pairs (vectorized) ---
    pairs = set()
    for dy, dx in [(0, 1), (1, 0)]:
        a = label_map[:-dy or None,  :-dx or None]
        b = label_map[dy:,            dx:]
        mask = (a != b) & (a >= 0) & (b >= 0)
        if mask.any():
            ab = np.column_stack([a[mask].ravel(), b[mask].ravel()])
            ab.sort(axis=1)
            for row in np.unique(ab, axis=0):
                pairs.add((int(row[0]), int(row[1])))

    if not pairs:
        return 0.0

    # --- Accumulate per-pixel scores ---
    score_sum   = np.zeros((H, W), dtype=np.float64)
    score_count = np.zeros((H, W), dtype=np.int32)

    for i, j in pairs:
        # Crop to bounding box of both labels + band margin
        region = (label_map == i) | (label_map == j)
        ys_r, xs_r = np.where(region)
        r0 = max(int(ys_r.min()) - band - 1, 0)
        r1 = min(int(ys_r.max()) + band + 2, H)
        c0 = max(int(xs_r.min()) - band - 1, 0)
        c1 = min(int(xs_r.max()) + band + 2, W)

        lm_crop = label_map[r0:r1, c0:c1]
        mask_i = lm_crop == i
        mask_j = lm_crop == j

        di_crop = warped_depths[i][r0:r1, c0:c1]
        dj_crop = warped_depths[j][r0:r1, c0:c1]
        overlap_ij = (di_crop > 0) & (dj_crop > 0)

        # Seam band: pixels within `band` of the i↔j boundary
        boundary_ij = (mask_i & binary_dilation(mask_j, struct_1px)) | \
                      (mask_j & binary_dilation(mask_i, struct_1px))
        seam_band_ij = binary_dilation(boundary_ij, struct_band)

        # Fitting region: overlap but outside the seam band
        fitting_mask = overlap_ij & ~seam_band_ij
        n_fit = fitting_mask.sum()
        if n_fit < 10:
            continue

        ys_f, xs_f = np.where(fitting_mask)
        if n_fit > n_fit_samples:
            idx = np.random.choice(n_fit, n_fit_samples, replace=False)
            ys_f, xs_f = ys_f[idx], xs_f[idx]

        di_f = di_crop[ys_f, xs_f].astype(np.float64)
        dj_f = dj_crop[ys_f, xs_f].astype(np.float64)

        # Fit s_ij * d_i + t_ij ≈ d_j  (2-parameter linear regression)
        A_fit = np.column_stack([di_f, np.ones_like(di_f)])
        x, _, _, _ = np.linalg.lstsq(A_fit, dj_f, rcond=None)
        s_ij, t_ij = float(x[0]), float(x[1])

        # Evaluation region: seam band with valid depth from both images
        eval_mask = seam_band_ij & overlap_ij
        if eval_mask.sum() == 0:
            continue

        ys_e, xs_e = np.where(eval_mask)
        if len(ys_e) > n_eval_samples:
            idx = np.random.choice(len(ys_e), n_eval_samples, replace=False)
            ys_e, xs_e = ys_e[idx], xs_e[idx]
        di_e = di_crop[ys_e, xs_e].astype(np.float64)
        dj_e = dj_crop[ys_e, xs_e].astype(np.float64)

        aligned_i = s_ij * di_e + t_ij
        residuals = np.abs(aligned_i - dj_e)

        # Normalize: use the depth range across the evaluation band
        d_lo = np.minimum(aligned_i, dj_e)
        d_hi = np.maximum(aligned_i, dj_e)
        band_range = max(float(d_hi.max() - d_lo.min()), 1e-6)
        normalized = residuals / band_range

        # Accumulate into per-pixel arrays (offset back to full canvas coords)
        np.add.at(score_sum,   (ys_e + r0, xs_e + c0), normalized)
        np.add.at(score_count, (ys_e + r0, xs_e + c0), 1)

    valid = score_count > 0
    if not valid.any():
        return 0.0

    # Uniform average across pairs at each pixel, then mean over all boundary pixels
    pixel_means = score_sum[valid] / score_count[valid]
    return float(pixel_means.mean())


# ---------------------------------------------------------------------------
# Convenience wrappers
# ---------------------------------------------------------------------------

def evaluate_scene(pano, seams, warped_depths=None, label_map=None, band=10):
    """
    Compute all applicable seam quality metrics for a panorama.

    Gradient and color metrics are computed per-seam then averaged.
    Depth metric uses the label_map (topology-agnostic, covers all boundaries).

    Args:
        pano:          (H, W, 3) uint8 composited panorama
        seams:         list of (H,) int arrays (one per adjacent image pair)
        warped_depths: list of N raw warped depth maps (not globally aligned)
        label_map:     (H, W) int array — source image index per pixel
        band:          half-width for all metrics

    Returns:
        dict with keys: "gradient_energy", "color_discontinuity",
                        and optionally "depth_score"
    """
    grad_scores  = [seam_gradient_energy(pano, s, band) for s in seams]
    color_scores = [seam_color_discontinuity(pano, s)   for s in seams]

    result = {
        "gradient_energy":     float(np.mean(grad_scores)),
        "color_discontinuity": float(np.mean(color_scores)),
    }

    if warped_depths is not None and label_map is not None:
        result["depth_score"] = depth_boundary_score(
            label_map, warped_depths, band
        )

    return result
