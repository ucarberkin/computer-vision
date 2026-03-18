"""
seam.py — DP seam finding for panoramic image stitching.

Two public entry points:

  find_seam(warped_a, warped_b, mask_a, mask_b, ...)
      Compute the optimal seam between two warped images and return
      boolean assignment masks.  Implements the center-cut fallback when
      there is no overlap.

  make_depth_seam_fn(aligned_depths, alpha=0.5)
      Factory that closes over a list of globally-aligned depth maps and
      returns a seam_fn compatible with composite_sequential / stitch.
"""

import numpy as np
import cv2


# ---------------------------------------------------------------------------
# Core DP seam finder
# ---------------------------------------------------------------------------

def find_seam(warped_a, warped_b, mask_a, mask_b,
              alpha=0.5, aligned_depth_a=None, aligned_depth_b=None):
    """
    Find the minimum-cost seam through the overlap of two warped images.

    The seam is a 1-pixel-wide path (one column per row) that can shift
    at most one column between adjacent rows (8-connected).  Per-pixel
    cost combines:
      - gradient term  : ||∇I_avg||  at each overlap pixel
      - depth-diff term: |depth_a - depth_b|  (omitted if depths are None)

    Both terms are independently normalized to [0, 1] within the overlap
    before mixing so that alpha is a meaningful weight.

    Parameters
    ----------
    warped_a, warped_b      : (H, W, 3) float64, pixel values in [0, 1]
    mask_a,   mask_b        : (H, W) bool
    alpha                   : weight for gradient term; (1-alpha) for depth term
    aligned_depth_a/b       : (H, W) float64 globally-aligned depth maps,
                              or None to fall back to gradient-only (alpha=1)

    Returns
    -------
    assign_a, assign_b : (H, W) bool arrays partitioning mask_a | mask_b
    """
    H, W = mask_a.shape
    overlap = mask_a & mask_b

    # ------------------------------------------------------------------
    # No overlap → uncontested assignment, no seam needed
    # ------------------------------------------------------------------
    if not overlap.any():
        return mask_a.copy(), mask_b.copy()

    # ------------------------------------------------------------------
    # Overlap bounding box (restrict DP to relevant columns)
    # ------------------------------------------------------------------
    overlap_cols = np.where(overlap.any(axis=0))[0]
    overlap_rows = np.where(overlap.any(axis=1))[0]
    c_lo, c_hi = int(overlap_cols[0]), int(overlap_cols[-1]) + 1  # [c_lo, c_hi)
    r_lo, r_hi = int(overlap_rows[0]), int(overlap_rows[-1]) + 1

    # ------------------------------------------------------------------
    # Cost map over the full canvas (only overlap pixels are meaningful)
    # ------------------------------------------------------------------
    # Gradient term on the average of both images
    avg = ((warped_a + warped_b) / 2.0 * 255).astype(np.float32)
    gray = cv2.cvtColor(avg, cv2.COLOR_RGB2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
    grad_cost = np.sqrt(gx ** 2 + gy ** 2).astype(np.float64)

    # Depth-diff term
    use_depth = (aligned_depth_a is not None) and (aligned_depth_b is not None)
    if use_depth:
        depth_cost = np.abs(aligned_depth_a.astype(np.float64)
                            - aligned_depth_b.astype(np.float64))
    else:
        depth_cost = np.zeros((H, W), dtype=np.float64)
        alpha = 1.0  # gradient-only fallback

    # Normalize each term to [0, 1] within the overlap
    def _norm(arr):
        vals = arr[overlap]
        lo, hi = float(vals.min()), float(vals.max())
        if hi > lo:
            return (arr - lo) / (hi - lo)
        return np.zeros_like(arr)

    grad_cost  = _norm(grad_cost)
    depth_cost = _norm(depth_cost)

    cost = alpha * grad_cost + (1.0 - alpha) * depth_cost  # (H, W)

    # ------------------------------------------------------------------
    # DP over the overlap band [r_lo:r_hi, c_lo:c_hi]
    # ------------------------------------------------------------------
    band_h = r_hi - r_lo
    band_w = c_hi - c_lo
    cost_band = cost[r_lo:r_hi, c_lo:c_hi]       # (band_h, band_w)
    ov_band   = overlap[r_lo:r_hi, c_lo:c_hi]    # valid pixels inside band

    INF = 1e18
    dp  = np.full((band_h, band_w), INF, dtype=np.float64)

    # Seed first row
    dp[0] = np.where(ov_band[0], cost_band[0], INF)

    for r in range(1, band_h):
        prev_l = np.full(band_w, INF)
        prev_s = dp[r - 1]
        prev_r = np.full(band_w, INF)
        prev_l[1:]  = dp[r - 1, :-1]   # predecessor was one column to the left
        prev_r[:-1] = dp[r - 1, 1:]    # predecessor was one column to the right

        best_prev = np.minimum(np.minimum(prev_l, prev_s), prev_r)
        dp[r] = np.where(ov_band[r], cost_band[r] + best_prev, INF)

    # ------------------------------------------------------------------
    # Traceback
    # ------------------------------------------------------------------
    seam_col = np.full(H, -1, dtype=np.int32)
    fallback = int(np.median(overlap_cols))

    last_row_dp = dp[band_h - 1]
    valid_last  = last_row_dp < INF

    if not valid_last.any():
        seam_col[:] = fallback
    else:
        best_bc = int(np.argmin(np.where(valid_last, last_row_dp, INF)))
        seam_col[r_lo + band_h - 1] = c_lo + best_bc

        cur_bc = best_bc
        for r in range(band_h - 2, -1, -1):
            c_l = max(cur_bc - 1, 0)
            c_m = cur_bc
            c_r = min(cur_bc + 1, band_w - 1)
            best_offset = int(np.argmin([dp[r, c_l], dp[r, c_m], dp[r, c_r]])) - 1
            cur_bc = int(np.clip(cur_bc + best_offset, 0, band_w - 1))
            seam_col[r_lo + r] = c_lo + cur_bc

        seam_col[:r_lo] = fallback
        seam_col[r_hi:] = fallback

    # ------------------------------------------------------------------
    # Build assignment masks
    # ------------------------------------------------------------------
    col_idx = np.arange(W)[None, :]   # (1, W)
    seam_2d = seam_col[:, None]       # (H, 1)

    # In the overlap: pixels left of seam → A, pixels at/right of seam → B
    assign_a = mask_a & ((col_idx <  seam_2d) | ~overlap)
    assign_b = mask_b & ((col_idx >= seam_2d) | ~overlap)

    return assign_a, assign_b


# ---------------------------------------------------------------------------
# Factory for use with stitch / composite_sequential
# ---------------------------------------------------------------------------

def make_depth_seam_fn(aligned_depths, alpha=0.5):
    """
    Return a seam_fn that uses globally-aligned depth maps to guide the seam.

    composite_sequential calls seam_fn exactly N-1 times in order (once per
    adjacent pair).  A call counter tracks which pair is being processed so
    the correct depth maps are forwarded to find_seam.

    Parameters
    ----------
    aligned_depths : list of N (H, W) float64 depth maps in canvas frame,
                     as returned by depth_utils.apply_alignment
    alpha          : gradient weight in [0, 1]; (1-alpha) goes to depth term

    Returns
    -------
    seam_fn : callable(warped_a, warped_b, mask_a, mask_b) -> (assign_a, assign_b)
    """
    call_count = [0]

    def seam_fn(warped_a, warped_b, mask_a, mask_b):
        i = call_count[0]
        call_count[0] += 1
        return find_seam(
            warped_a, warped_b, mask_a, mask_b,
            alpha=alpha,
            aligned_depth_a=aligned_depths[i],
            aligned_depth_b=aligned_depths[i + 1],
        )

    def reset():
        """Reset the call counter so the seam_fn can be reused."""
        call_count[0] = 0

    seam_fn.reset = reset
    return seam_fn
