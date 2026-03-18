"""pipeline.py — Panoramic image stitching pipeline.

Phases:
  1. Load images
  2. SIFT detection + BFMatcher with Lowe's ratio test
  3. RANSAC homography estimation for all image pairs
  4. BFS initialization of H_to_ref through the pair graph
  5. Bundle adjustment via Levenberg-Marquardt over all valid pairs
  6. Compute canvas size and warp all images
  7. Composite using a seam function (default: center-cut baseline)

The seam function interface is:
    seam_fn(warped_a, warped_b, mask_a, mask_b) -> (assign_a, assign_b)
where assign_a and assign_b are boolean arrays that partition the union of the two
masks. This interface is shared by the center-cut baseline (below) and the
depth-guided DP seam in seam.py.
"""

import numpy as np
import cv2
from pathlib import Path
from collections import deque
from imageio.v2 import imread
from scipy.optimize import least_squares


# ── Image loading ──────────────────────────────────────────────────────────────

def load_images(folder):
    """Load all .jpg / .JPG images from folder, sorted by filename.

    Returns
    -------
    images : list of H×W×3 uint8 arrays
    names  : list of filenames (for logging)
    """
    folder = Path(folder)
    paths = sorted(folder.glob("*.[jJ][pP][gG]"), key=lambda p: p.name.lower())
    if not paths:
        raise FileNotFoundError(f"No JPEG images found in {folder}")
    images = [imread(str(p)) for p in paths]
    names  = [p.name for p in paths]
    return images, names


def to_gray_u8(img):
    """Convert an RGB (or already-gray) image to uint8 grayscale."""
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img.copy()
    if gray.dtype != np.uint8:
        gray = (gray * 255).astype(np.uint8) if gray.max() <= 1.0 else gray.astype(np.uint8)
    return gray


# ── Feature detection & matching ───────────────────────────────────────────────

_sift = cv2.SIFT_create()


def detect_and_describe(img_gray_u8):
    """Return (keypoints, descriptors) for a uint8 grayscale image."""
    return _sift.detectAndCompute(img_gray_u8, None)


def match_with_ratio_test(desc_a, desc_b, ratio=0.8):
    """BFMatcher + Lowe's ratio test. Returns list of cv2.DMatch."""
    bf = cv2.BFMatcher(cv2.NORM_L2)
    raw = bf.knnMatch(desc_a, desc_b, k=2)
    return [m for m, n in raw if m.distance < ratio * n.distance]


def estimate_homography(kp_a, kp_b, matches, ransac_thresh=5.0):
    """RANSAC homography H such that x_b ≈ H @ x_a (both in homogeneous coords).

    Returns
    -------
    H    : 3×3 float64
    mask : bool array of length len(matches), True = inlier
    """
    pts_a = np.float32([kp_a[m.queryIdx].pt for m in matches])
    pts_b = np.float32([kp_b[m.trainIdx].pt for m in matches])
    H, mask = cv2.findHomography(pts_a, pts_b, cv2.RANSAC,
                                  ransacReprojThreshold=ransac_thresh)
    if H is None or mask is None:
        return None, np.zeros(len(matches), dtype=bool)
    return H, mask.ravel().astype(bool)


# ── All-pairs matching ────────────────────────────────────────────────────────

def compute_all_pairs(images_rgb, ratio=0.8, ransac_thresh=5.0, min_inliers=80):
    """Match every image pair (i, j) and return those with enough RANSAC inliers.

    Parameters
    ----------
    images_rgb    : list of N images
    ratio         : Lowe's ratio test threshold
    ransac_thresh : RANSAC reprojection threshold (pixels)
    min_inliers   : minimum inliers for a pair to be accepted

    Returns
    -------
    pair_indices  : list of (i, j) tuples with i < j
    homographies  : dict {(i, j): H} — H maps image i coords to image j coords
    inlier_pts    : list of (pts_i, pts_j) tuples, each Kx2 float32
    kps           : list of N keypoint lists
    descs         : list of N descriptor arrays
    """
    grays = [to_gray_u8(img) for img in images_rgb]
    kps, descs = zip(*[detect_and_describe(g) for g in grays])
    kps = list(kps)
    descs = list(descs)

    n = len(images_rgb)
    pair_indices = []
    homographies = {}
    inlier_pts = []

    for i in range(n):
        for j in range(i + 1, n):
            good = match_with_ratio_test(descs[i], descs[j], ratio)
            if len(good) < min_inliers:
                continue

            H, mask = estimate_homography(kps[i], kps[j], good, ransac_thresh)
            if H is None:
                continue

            inlier_matches = [m for m, keep in zip(good, mask) if keep]
            if len(inlier_matches) < min_inliers:
                continue

            pts_i = np.float32([kps[i][m.queryIdx].pt for m in inlier_matches])
            pts_j = np.float32([kps[j][m.trainIdx].pt for m in inlier_matches])

            pair_indices.append((i, j))
            homographies[(i, j)] = H
            inlier_pts.append((pts_i, pts_j))
            print(f"  pair {i}→{j}: {len(inlier_matches)} inliers")

    return pair_indices, homographies, inlier_pts, kps, descs


# ── Graph-based homography initialization ─────────────────────────────────────

def init_homographies_bfs(n, ref_idx, pair_indices, homographies):
    """Initialize H_to_ref for each image via BFS through the pair graph.

    Finds shortest path from each image to the reference through valid pairs,
    composing homographies along the path.

    Parameters
    ----------
    n             : number of images
    ref_idx       : reference image index
    pair_indices  : list of (i, j) tuples with i < j
    homographies  : dict {(i, j): H} mapping image i → image j

    Returns
    -------
    H_to_ref     : list of N homographies (None for disconnected images)
    disconnected : list of image indices not reachable from the reference
    """
    # Build adjacency list
    adj = {i: [] for i in range(n)}
    for (i, j) in pair_indices:
        adj[i].append(j)
        adj[j].append(i)

    H_to_ref = [None] * n
    H_to_ref[ref_idx] = np.eye(3)

    visited = {ref_idx}
    queue = deque([ref_idx])

    while queue:
        node = queue.popleft()
        for neighbor in adj[node]:
            if neighbor in visited:
                continue
            visited.add(neighbor)
            queue.append(neighbor)

            # Get H_{neighbor → node}
            if (neighbor, node) in homographies:
                H_nb_to_node = homographies[(neighbor, node)]
            else:
                H_nb_to_node = np.linalg.inv(homographies[(node, neighbor)])

            # H_{neighbor → ref} = H_{node → ref} @ H_{neighbor → node}
            H_to_ref[neighbor] = H_to_ref[node] @ H_nb_to_node

    disconnected = [i for i in range(n) if H_to_ref[i] is None]
    return H_to_ref, disconnected


# ── Bundle adjustment ──────────────────────────────────────────────────────────

def _H_to_params(H):
    """8-vector of free parameters (H normalized so H[2,2]=1)."""
    H = H / H[2, 2]
    return H.ravel()[:8].copy()


def _params_to_H(p):
    """Reconstruct 3×3 homography from 8-vector."""
    return np.append(p, 1.0).reshape(3, 3)


def _project(H_i_to_ref, H_j_to_ref, pts_i):
    """Map pts_i (Nx2) from image i coords to image j coords via the reference frame."""
    H = np.linalg.inv(H_j_to_ref) @ H_i_to_ref   # H_{i→j}
    pts_h = np.hstack([pts_i, np.ones((len(pts_i), 1))])  # Nx3
    proj  = (H @ pts_h.T).T                               # Nx3
    return proj[:, :2] / proj[:, 2:3]                     # Nx2


def bundle_adjust(H_init, pair_indices, inlier_pts, ref_idx):
    """Globally refine H_{i→ref} via Levenberg-Marquardt.

    Parameters
    ----------
    H_init       : initial list of N homographies (from BFS initialization)
    pair_indices : list of (i, j) tuples — all valid pairs
    inlier_pts   : list of (pts_i, pts_j) tuples, each Kx2 float32
    ref_idx      : index of the fixed reference image

    Returns
    -------
    Refined list of N homographies H_{i→ref}.
    """
    n = len(H_init)
    free = [i for i in range(n) if i != ref_idx]   # indices of non-reference images

    def pack(H_list):
        return np.concatenate([_H_to_params(H_list[i]) for i in free])

    def unpack(params):
        H_list = list(H_init)          # copy; reference slot stays as identity
        for k, i in enumerate(free):
            H_list[i] = _params_to_H(params[k * 8:(k + 1) * 8])
        return H_list

    def residuals(params):
        H_list = unpack(params)
        res = []
        for (i, j), (pts_i, pts_j) in zip(pair_indices, inlier_pts):
            proj = _project(H_list[i], H_list[j], pts_i)
            res.append((proj - pts_j).ravel())
        return np.concatenate(res)

    x0 = pack(H_init)
    total_corr = sum(len(p[0]) for p in inlier_pts)
    print(f"  Bundle adjust: {len(free)} free images, "
          f"{total_corr} correspondences across {len(pair_indices)} pairs")

    result = least_squares(residuals, x0, method='lm', verbose=1)
    print(f"  Final cost: {result.cost:.4f}")

    return unpack(result.x)


# ── Canvas layout & warping ────────────────────────────────────────────────────

def _get_corners(img):
    h, w = img.shape[:2]
    return np.float32([[0, 0], [w, 0], [w, h], [0, h]])


def compute_canvas(images, H_to_ref, max_scale=1.5):
    """Compute canvas dimensions and offset translation T.

    T shifts the reference frame origin so all warped images land at non-negative
    coordinates.  Warped corners beyond a reasonable range are clamped to prevent
    an absurdly large canvas (common with looping panoramas where homography
    chaining stretches edge images).  This runs after all homography estimation
    and does not affect the homographies themselves.

    Returns (canvas_w, canvas_h, T_offset).
    """
    total_w = sum(img.shape[1] for img in images)
    max_h   = max(img.shape[0] for img in images)
    max_coord = max(total_w, max_h) * max_scale

    all_corners = []
    for img, H in zip(images, H_to_ref):
        c = _get_corners(img).reshape(-1, 1, 2)
        warped = cv2.perspectiveTransform(c, H).reshape(-1, 2)
        all_corners.append(np.clip(warped, -max_coord, max_coord))
    all_corners = np.vstack(all_corners)

    mn = np.floor(all_corners.min(axis=0)).astype(int)
    mx = np.ceil( all_corners.max(axis=0)).astype(int)
    canvas_w, canvas_h = (mx - mn)[0], (mx - mn)[1]

    T_offset = np.array([[1, 0, -mn[0]],
                          [0, 1, -mn[1]],
                          [0, 0,      1]], dtype=np.float64)
    return canvas_w, canvas_h, T_offset


def warp_all(images, H_to_ref, T_offset, canvas_w, canvas_h):
    """Warp every image into the shared canvas coordinate frame.

    Returns
    -------
    warped : list of H×W×3 float32 arrays (pixel values in [0, 1])
    masks  : list of H×W bool arrays (True where the image covers the canvas)
    """
    dsize = (canvas_w, canvas_h)
    warped, masks = [], []
    for img, H in zip(images, H_to_ref):
        M   = T_offset @ H
        w   = cv2.warpPerspective(img.astype(np.float32) / 255.0, M, dsize)
        m   = cv2.warpPerspective(
                  np.ones(img.shape[:2], dtype=np.float32), M, dsize) > 0.5
        warped.append(w)
        masks.append(m)
    return warped, masks


# ── Seam compositing ───────────────────────────────────────────────────────────

def center_cut_seam(warped_a, warped_b, mask_a, mask_b):
    """Baseline seam: straight vertical cut at the median column of the overlap.

    Parameters
    ----------
    warped_a, warped_b : H×W×3 float32  (used by depth-guided variant; ignored here)
    mask_a, mask_b     : H×W bool

    Returns
    -------
    assign_a, assign_b : H×W bool arrays partitioning mask_a | mask_b
    """
    overlap = mask_a & mask_b
    col_idx = np.arange(mask_a.shape[1])

    assign_a = mask_a & ~overlap                          # uncontested pixels for A
    assign_b = mask_b & ~overlap                          # uncontested pixels for B

    if overlap.any():
        mid = int(np.median(np.where(overlap)[1]))        # median overlap column
        assign_a |= overlap & (col_idx[None, :] <= mid)
        assign_b |= overlap & (col_idx[None, :] >  mid)

    return assign_a, assign_b


def composite_sequential(warped_imgs, masks, seam_fn=None):
    """Composite N warped images left-to-right, applying seam_fn at each step.

    Parameters
    ----------
    warped_imgs : list of H×W×3 float32
    masks       : list of H×W bool
    seam_fn     : callable(warped_a, warped_b, mask_a, mask_b) → (assign_a, assign_b)

    Returns
    -------
    canvas    : H×W×3 float32 (pixel values in [0, 1])
    label_map : H×W int32 — index of source image owning each pixel (-1 = unassigned)
    seams     : list of N-1 int arrays, seams[i][r] = seam column at row r for pair i↔i+1
    """
    if seam_fn is None:
        seam_fn = center_cut_seam

    H, W = warped_imgs[0].shape[:2]
    canvas      = np.zeros((H, W, 3), dtype=np.float32)
    label_map   = np.full((H, W), -1, dtype=np.int32)
    canvas_mask = np.zeros((H, W), dtype=bool)

    canvas[masks[0]]    = warped_imgs[0][masks[0]]
    label_map[masks[0]] = 0
    canvas_mask         = masks[0].copy()

    seams = []
    for i in range(1, len(warped_imgs)):
        prev_canvas_mask = canvas_mask          # capture before update — used for seam extraction
        assign_old, assign_new = seam_fn(
            warped_imgs[i - 1], warped_imgs[i],
            canvas_mask,        masks[i]
        )
        canvas[assign_new]    = warped_imgs[i][assign_new]
        label_map[assign_new] = i
        canvas_mask           = assign_old | assign_new

        # Extract seam path: for each row, first column assigned to the new image
        # within the overlap between the accumulated canvas and the new image's mask
        overlap = prev_canvas_mask & masks[i]
        seam = np.zeros(H, dtype=np.int32)
        for r in range(H):
            new_cols = np.where(assign_new[r] & overlap[r])[0]
            seam[r] = int(new_cols[0]) if len(new_cols) > 0 else W // 2
        seams.append(seam)

    return canvas, label_map, seams


# ── Main entry point ───────────────────────────────────────────────────────────

def stitch(images_rgb, seam_fn=None, use_bundle_adjustment=True,
           ratio=0.8, ransac_thresh=5.0, ref_idx=None, min_inliers=80):
    """Full stitching pipeline for N images.

    Matches all image pairs, initializes homographies via BFS through the
    connectivity graph, then refines with bundle adjustment over all pairs.
    Disconnected images (not reachable from the reference) are dropped
    automatically.

    Parameters
    ----------
    images_rgb           : list of H×W×3 uint8
    seam_fn              : seam function; None → center_cut_seam
    use_bundle_adjustment: if True, refine homographies via Levenberg-Marquardt
    ratio                : Lowe's ratio test threshold
    ransac_thresh        : RANSAC reprojection threshold (pixels)
    ref_idx              : reference image index; None → N//2
    min_inliers          : minimum RANSAC inliers for a pair to be accepted

    Returns
    -------
    panorama : H×W×3 uint8
    meta     : dict with keys H_to_ref, T_offset, warped_imgs, masks, images, ...
               (images contains the kept images — use for depth inference)
    """
    n = len(images_rgb)
    if ref_idx is None:
        ref_idx = n // 2
    print(f"Stitching {n} images  |  reference = {ref_idx}  |  "
          f"bundle_adjust = {use_bundle_adjustment}")

    print("Matching all image pairs...")
    pair_indices, homographies, inlier_pts, kps, descs = compute_all_pairs(
        images_rgb, ratio, ransac_thresh, min_inliers
    )
    print(f"  {len(pair_indices)} valid pairs found")

    print("Initializing homographies via BFS...")
    H_to_ref, disconnected = init_homographies_bfs(
        n, ref_idx, pair_indices, homographies
    )

    # Drop disconnected images
    if disconnected:
        print(f"  WARNING: dropping disconnected images {disconnected}")
        keep = [i for i in range(n) if i not in disconnected]
        # Re-index everything to the kept subset
        old_to_new = {old: new for new, old in enumerate(keep)}
        images_rgb = [images_rgb[i] for i in keep]
        H_to_ref   = [H_to_ref[i]   for i in keep]
        kps        = [kps[i]         for i in keep]
        descs      = [descs[i]       for i in keep]

        new_pairs, new_inlier_pts = [], []
        for (i, j), pts in zip(pair_indices, inlier_pts):
            if i in old_to_new and j in old_to_new:
                new_pairs.append((old_to_new[i], old_to_new[j]))
                new_inlier_pts.append(pts)
        pair_indices = new_pairs
        inlier_pts   = new_inlier_pts

        ref_idx = old_to_new.get(ref_idx, len(keep) // 2)
        n = len(images_rgb)
        print(f"  Kept {n} images, new reference = {ref_idx}")

    if use_bundle_adjustment and len(pair_indices) > 0:
        print("Running bundle adjustment...")
        H_to_ref = bundle_adjust(H_to_ref, pair_indices, inlier_pts, ref_idx)

    print("Computing canvas layout...")
    canvas_w, canvas_h, T_offset = compute_canvas(images_rgb, H_to_ref)
    print(f"  Canvas: {canvas_w} × {canvas_h} px")

    print("Warping images...")
    warped_imgs, masks = warp_all(images_rgb, H_to_ref, T_offset, canvas_w, canvas_h)

    print("Compositing...")
    canvas, label_map, seams = composite_sequential(warped_imgs, masks, seam_fn)

    panorama = np.clip(canvas * 255, 0, 255).astype(np.uint8)

    # Auto-crop to content bounds (remove black border)
    content_mask = np.any(canvas > 0, axis=2)
    if content_mask.any():
        ys, xs = np.where(content_mask)
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        panorama  = panorama[y0:y1, x0:x1]
        label_map = label_map[y0:y1, x0:x1]
        seams     = [s[y0:y1] - x0 for s in seams]
        warped_imgs = [w[y0:y1, x0:x1] for w in warped_imgs]
        masks       = [m[y0:y1, x0:x1] for m in masks]
        canvas_w, canvas_h = x1 - x0, y1 - y0
        # Update T_offset to account for the crop
        T_crop = np.array([[1, 0, -x0],
                           [0, 1, -y0],
                           [0, 0,   1]], dtype=np.float64)
        T_offset = T_crop @ T_offset
        print(f"  Cropped: {canvas_w} × {canvas_h} px")

    meta = dict(H_to_ref=H_to_ref, T_offset=T_offset,
                warped_imgs=warped_imgs, masks=masks,
                label_map=label_map, seams=seams,
                canvas_w=canvas_w, canvas_h=canvas_h,
                images=images_rgb, kps=kps)

    print("Done.")
    return panorama, meta
