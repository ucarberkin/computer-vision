"""
depth_utils.py — depth estimation, warping, and global scale alignment.

Supports:
  - Depth Anything v2 Large  (MODEL_DA2_LARGE)
  - Marigold LCM             (MODEL_MARIGOLD)
"""

import numpy as np
import cv2
from PIL import Image

MODEL_DA2_LARGE = "depth-anything/Depth-Anything-V2-Large-hf"
MODEL_MARIGOLD  = "prs-eth/marigold-lcm-v1-0"


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_depth_model(model_name=MODEL_DA2_LARGE):
    """
    Load a depth estimation model.

    Returns a callable:  infer(pil_image: PIL.Image) -> np.ndarray float32 [0,1]

    Selects MPS (Apple Silicon) if available, otherwise CPU.
    """
    import torch
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    if "marigold" in model_name.lower():
        from diffusers import MarigoldDepthPipeline
        pipe = MarigoldDepthPipeline.from_pretrained(
            model_name, variant="fp16", torch_dtype=torch.float16
        ).to(device)

        def infer(pil_img):
            out = pipe(pil_img, num_inference_steps=4, ensemble_size=1)
            pred = out.prediction
            if hasattr(pred, 'cpu'):
                pred = pred.cpu().numpy()
            else:
                pred = np.asarray(pred)
            depth = pred.squeeze().astype(np.float32)
            return _normalize(depth)

    else:
        from transformers import pipeline as hf_pipeline
        pipe = hf_pipeline("depth-estimation", model=model_name, device=device)

        def infer(pil_img):
            result = pipe(pil_img)
            depth = result["predicted_depth"].squeeze().cpu().numpy().astype(np.float32)
            return _normalize(depth)

    return infer


def _normalize(depth):
    d_min, d_max = depth.min(), depth.max()
    if d_max > d_min:
        return (depth - d_min) / (d_max - d_min)
    return np.zeros_like(depth)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def run_depth(model_fn, images):
    """
    Run depth inference on a list of RGB uint8 numpy arrays.

    Returns a list of float32 depth maps, each normalized to [0,1] and
    resized to match its source image's (H, W).
    """
    depths = []
    for img in images:
        pil = Image.fromarray(img)
        depth = model_fn(pil)
        h, w = img.shape[:2]
        if depth.shape != (h, w):
            depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR)
        depths.append(depth)
    return depths


# ---------------------------------------------------------------------------
# Warping into canvas frame
# ---------------------------------------------------------------------------

def warp_depths(depths, H_to_ref, T_offset, cw, ch):
    """
    Warp each depth map into the panorama canvas frame using the same
    homographies used for RGB warping in pipeline.py.

    Args:
        depths:    list of N float32 depth maps (each Hᵢ × Wᵢ)
        H_to_ref:  list of N 3×3 homography matrices (image i → reference frame)
        T_offset:  3×3 translation matrix (reference frame → canvas origin)
        cw, ch:    canvas width and height

    Returns:
        warped_depths: list of N float32 depth maps, shape (ch, cw)
        depth_masks:   list of N bool masks, True where depth is valid
    """
    dsize = (cw, ch)
    warped_depths, depth_masks = [], []

    for depth, H in zip(depths, H_to_ref):
        M = T_offset @ H
        wd = cv2.warpPerspective(
            depth, M, dsize,
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0.0
        )
        mask = cv2.warpPerspective(
            np.ones_like(depth), M, dsize,
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0.0
        ) > 0.5
        warped_depths.append(wd)
        depth_masks.append(mask)

    return warped_depths, depth_masks


# ---------------------------------------------------------------------------
# Global depth scale alignment
# ---------------------------------------------------------------------------

def global_depth_alignment(warped_depths, depth_masks, ref_idx=None,
                            n_samples=2000, unreliable_thresh=0.30):
    """
    Jointly solve for per-image affine parameters (sᵢ, tᵢ) so that aligned
    depths agree across all overlap regions simultaneously:

        min_{sᵢ,tᵢ}  Σ_{(i,j)} Σ_{p ∈ Ω_ij} (sᵢ·dᵢ(p) + tᵢ − sⱼ·dⱼ(p) − tⱼ)²

    The objective is linear in the unknowns, so the system reduces to an
    overdetermined Ax = b solved via np.linalg.lstsq (no iteration needed).

    Anchor: s_{ref_idx} = 1, t_{ref_idx} = 0  (resolves scale ambiguity).

    Args:
        warped_depths:     list of N float32 depth maps in canvas frame
        depth_masks:       list of N bool masks (valid pixels per image)
        ref_idx:           reference image index (default N//2)
        n_samples:         pixels sampled per overlap pair to build the system
        unreliable_thresh: pairs with normalized residual above this are flagged

    Returns:
        scales:   (N,) float64 array of sᵢ
        shifts:   (N,) float64 array of tᵢ
        reliable: dict {(i, j): bool} — False if pair residual > unreliable_thresh
    """
    N = len(warped_depths)
    if ref_idx is None:
        ref_idx = N // 2

    # Collect all overlapping pairs
    pair_data = []
    for i in range(N):
        for j in range(i + 1, N):
            overlap = depth_masks[i] & depth_masks[j]
            if overlap.sum() > 50:
                pair_data.append((i, j, overlap))

    if not pair_data:
        return np.ones(N), np.zeros(N), {}

    # Free variables: 2*(N-1) unknowns (sᵢ, tᵢ) for all images except ref
    free_imgs = [i for i in range(N) if i != ref_idx]
    var_map = {img: 2 * k for k, img in enumerate(free_imgs)}
    n_vars = 2 * (N - 1)

    A_blocks, b_blocks = [], []

    for i, j, overlap in pair_data:
        ys, xs = np.where(overlap)
        if len(ys) > n_samples:
            idx = np.random.choice(len(ys), n_samples, replace=False)
            ys, xs = ys[idx], xs[idx]

        di = warped_depths[i][ys, xs].astype(np.float64)
        dj = warped_depths[j][ys, xs].astype(np.float64)
        n = len(ys)

        block = np.zeros((n, n_vars))
        rhs   = np.zeros(n)

        # Constraint: sᵢ·dᵢ + tᵢ − sⱼ·dⱼ − tⱼ = 0
        if i != ref_idx:
            block[:, var_map[i]]     = di    # sᵢ coefficient
            block[:, var_map[i] + 1] = 1.0  # tᵢ coefficient
        else:
            rhs -= di   # sᵢ=1, tᵢ=0 anchored → move to RHS as −dᵢ

        if j != ref_idx:
            block[:, var_map[j]]     = -dj   # −sⱼ coefficient
            block[:, var_map[j] + 1] = -1.0  # −tⱼ coefficient
        else:
            rhs += dj   # sⱼ=1, tⱼ=0 anchored → move to RHS as +dⱼ

        A_blocks.append(block)
        b_blocks.append(rhs)

    A = np.vstack(A_blocks)
    b = np.concatenate(b_blocks)

    x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

    scales = np.ones(N,  dtype=np.float64)
    shifts = np.zeros(N, dtype=np.float64)
    for img in free_imgs:
        scales[img] = x[var_map[img]]
        shifts[img] = x[var_map[img] + 1]

    # Per-pair reliability check
    reliable = {}
    for i, j, overlap in pair_data:
        ys, xs = np.where(overlap)
        ai = scales[i] * warped_depths[i][ys, xs] + shifts[i]
        aj = scales[j] * warped_depths[j][ys, xs] + shifts[j]
        residual    = np.mean(np.abs(ai - aj))
        depth_range = np.percentile(np.concatenate([ai, aj]), 95) \
                    - np.percentile(np.concatenate([ai, aj]), 5)
        reliable[(i, j)] = (residual / max(float(depth_range), 1e-6)) < unreliable_thresh

    return scales, shifts, reliable


# ---------------------------------------------------------------------------
# Apply alignment
# ---------------------------------------------------------------------------

def apply_alignment(warped_depths, scales, shifts):
    """Apply sᵢ·dᵢ + tᵢ to each warped depth map. Returns list of float64 arrays."""
    return [float(s) * d + float(t) for d, s, t in zip(warped_depths, scales, shifts)]


# ---------------------------------------------------------------------------
# Depth map persistence
# ---------------------------------------------------------------------------

# Short tag used as a subfolder name under outputs/depth_maps/
_MODEL_TAGS = {
    MODEL_DA2_LARGE: "da2_large",
    MODEL_MARIGOLD:  "marigold",
}


def model_tag(model_name):
    """Return a short filesystem-safe tag for a model name.

    Known models return a fixed tag; unknown models fall back to the last
    path component of the HuggingFace repo ID (e.g. 'my-org/my-model' → 'my-model').
    """
    return _MODEL_TAGS.get(model_name, model_name.split("/")[-1])


def save_depths(depths, path, overwrite=False):
    """
    Save a list of depth maps to a single compressed .npz file.

    Skips saving if the file already exists and overwrite=False, so
    re-running a notebook cell after a previous save is safe.

    Args:
        depths    : list of N float32/float64 numpy arrays
        path      : destination file path
                    e.g. 'outputs/depth_maps/da2_large/cmu1.npz'
                    Parent directories are created automatically.
        overwrite : if False (default), skip if file already exists
    """
    from pathlib import Path
    path = Path(path)
    if path.exists() and not overwrite:
        print(f"Depth maps already saved at {path} — skipping (pass overwrite=True to force).")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(path), **{f"depth_{i}": d.astype(np.float32)
                                      for i, d in enumerate(depths)})
    print(f"Saved {len(depths)} depth maps → {path}")


def load_depths(path):
    """
    Load depth maps saved by save_depths.

    Returns a list of float32 arrays ordered by index.
    """
    data = np.load(str(path))
    n = len(data.files)
    depths = [data[f"depth_{i}"] for i in range(n)]
    print(f"Loaded {n} depth maps ← {path}")
    return depths
