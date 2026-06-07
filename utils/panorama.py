# utils/panorama.py
"""
Panorama-specific helpers: vertical cropping, horizontal (yaw) shifting,
and searching for the best yaw alignment using LightGlue.
"""

from __future__ import annotations

import json
import logging
import math
import os
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from lightglue.utils import rbd

from .tensor_utils import load_image

logger = logging.getLogger(__name__)


# ---------- Basic panorama helpers ----------


def crop_equirectangular_middle(
    image_np: np.ndarray,
    keep_ratio: float = 0.55,
    top_bias: float = 0.0,
) -> np.ndarray:
    """
    Crop a vertical band from an equirectangular image, biased toward the top.

    Args:
        image_np: H x W x C RGB numpy array.
        keep_ratio: Fraction of the height to keep (0–1).
        top_bias:
            0.0 = center crop
            1.0 = shift crop toward top
           -1.0 = shift crop toward bottom

    Returns:
        Cropped Hc x W x C numpy array.
    """
    keep_ratio = float(keep_ratio)
    if keep_ratio >= 1.0:
        return image_np
    if keep_ratio <= 0.0:
        raise ValueError("keep_ratio must be greater than 0")

    h = image_np.shape[0]
    crop_h = max(1, int(round(h * keep_ratio)))
    extra = h - crop_h
    bias = max(-1.0, min(1.0, float(top_bias)))
    start = int(round((extra / 2.0) * (1.0 - bias)))
    start = max(0, min(extra, start))
    return image_np[start : start + crop_h]


def shift_equirectangular(pano_np: np.ndarray, yaw_deg: int) -> np.ndarray:
    """
    Cyclically shift panorama horizontally by yaw_deg.

    Positive yaw rotates the panorama so that content moves left (i.e. we shift
    the pixels to the right).

    Args:
        pano_np: H x W x C numpy array (RGB).
        yaw_deg: Rotation in degrees.

    Returns:
        Shifted panorama with the same shape.
    """
    w = pano_np.shape[1]
    shift_px = int(round((yaw_deg % 360) / 360.0 * w))
    return np.roll(pano_np, -shift_px, axis=1)


def _resize_mask_for_yaw_score(mask: np.ndarray, max_width: int = 1024) -> np.ndarray:
    """Downsample large masks for fast circular yaw scoring."""
    h, w = mask.shape[:2]
    if w <= max_width:
        return mask
    scale = max_width / float(w)
    target = (max_width, max(1, int(round(h * scale))))
    return cv2.resize(mask, target, interpolation=cv2.INTER_NEAREST)


def _semantic_yaw_score(
    left_mask: np.ndarray,
    right_mask: np.ndarray,
    yaw_deg: float,
    ignore_class_ids: Optional[List[int]] = None,
) -> float:
    """
    Score semantic alignment after shifting the right mask by yaw.

    Higher is better. Uses pixel label agreement after optionally ignoring
    dynamic classes such as vehicles and people.
    """
    if left_mask.shape != right_mask.shape:
        right_mask = cv2.resize(
            right_mask,
            (left_mask.shape[1], left_mask.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
    w = right_mask.shape[1]
    shift_px = int(round((yaw_deg % 360) / 360.0 * w))
    shifted = np.roll(right_mask, -shift_px, axis=1)
    valid = np.ones(left_mask.shape, dtype=bool)
    if ignore_class_ids:
        valid &= ~np.isin(left_mask, ignore_class_ids)
        valid &= ~np.isin(shifted, ignore_class_ids)
    if not valid.any():
        return 0.0
    return float(np.mean(left_mask[valid] == shifted[valid]))


def _keypoints_allowed(kp: np.ndarray, valid_mask: Optional[np.ndarray]) -> np.ndarray:
    """Return a boolean vector saying whether each keypoint is allowed."""
    if valid_mask is None:
        return np.ones(len(kp), dtype=bool)
    if len(kp) == 0:
        return np.zeros(0, dtype=bool)
    h, w = valid_mask.shape[:2]
    xs = np.clip(np.rint(kp[:, 0]).astype(np.int32), 0, w - 1)
    ys = np.clip(np.rint(kp[:, 1]).astype(np.int32), 0, h - 1)
    return valid_mask[ys, xs].astype(bool)


def rerank_yaw_by_semantic_masks(
    left_mask: np.ndarray,
    right_mask: np.ndarray,
    initial_yaw: int,
    search_radius: int = 8,
    step: int = 2,
    ignore_class_ids: Optional[List[int]] = None,
) -> Tuple[int, float, List[Dict[str, float]]]:
    """
    Rerank yaw candidates near an existing keypoint yaw using semantic masks.

    This is intended as a cheap accuracy pass after LightGlue has already
    narrowed the alignment to a small local window.
    """
    left_small = _resize_mask_for_yaw_score(left_mask)
    right_small = _resize_mask_for_yaw_score(right_mask, max_width=left_small.shape[1])
    step = max(1, int(step))
    radius = max(0, int(search_radius))

    best_yaw = int(initial_yaw) % 360
    best_score = -1.0
    scores: List[Dict[str, float]] = []
    for delta in range(-radius, radius + 1, step):
        yaw = int(round(initial_yaw + delta)) % 360
        score = _semantic_yaw_score(
            left_small,
            right_small,
            yaw,
            ignore_class_ids=ignore_class_ids,
        )
        scores.append({"yaw": float(yaw), "delta": float(delta), "score": float(score)})
        if score > best_score:
            best_score = score
            best_yaw = yaw
    return best_yaw, float(best_score), scores


def equirectangular_to_perspective(
    pano_np: np.ndarray,
    yaw_deg: float,
    pitch_deg: float = 0.0,
    fov_deg: float = 90.0,
    out_size: Tuple[int, int] = (768, 768),
) -> np.ndarray:
    """
    Project an equirectangular panorama into a rectilinear perspective view.

    Args:
        pano_np:   H x W x C RGB equirectangular image.
        yaw_deg:   Horizontal camera direction in degrees.
        pitch_deg: Vertical camera direction in degrees.
        fov_deg:   Horizontal field of view in degrees.
        out_size:  Output (width, height).

    Returns:
        Hout x Wout x C RGB perspective image.
    """
    pano_h, pano_w = pano_np.shape[:2]
    out_w, out_h = out_size
    fov_rad = math.radians(float(fov_deg))
    focal = (out_w / 2.0) / math.tan(fov_rad / 2.0)

    xs, ys = np.meshgrid(
        np.arange(out_w, dtype=np.float32),
        np.arange(out_h, dtype=np.float32),
    )
    x = (xs - (out_w - 1) / 2.0) / focal
    y = -((ys - (out_h - 1) / 2.0) / focal)
    z = np.ones_like(x)

    dirs = np.stack([x, y, z], axis=-1)
    dirs /= np.linalg.norm(dirs, axis=-1, keepdims=True)

    yaw = math.radians(float(yaw_deg))
    pitch = math.radians(float(pitch_deg))
    rot_yaw = np.array(
        [
            [math.cos(yaw), 0.0, math.sin(yaw)],
            [0.0, 1.0, 0.0],
            [-math.sin(yaw), 0.0, math.cos(yaw)],
        ],
        dtype=np.float32,
    )
    rot_pitch = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, math.cos(pitch), -math.sin(pitch)],
            [0.0, math.sin(pitch), math.cos(pitch)],
        ],
        dtype=np.float32,
    )
    dirs_world = dirs @ (rot_pitch.T @ rot_yaw.T)

    lon = np.arctan2(dirs_world[..., 0], dirs_world[..., 2])
    lat = np.arcsin(np.clip(dirs_world[..., 1], -1.0, 1.0))

    map_x = ((lon / (2.0 * math.pi)) + 0.5) * pano_w
    map_y = (0.5 - (lat / math.pi)) * pano_h

    return cv2.remap(
        pano_np,
        map_x.astype(np.float32),
        map_y.astype(np.float32),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_WRAP,
    )


def _estimate_horizon_line(view_rgb: np.ndarray) -> Optional[Tuple[float, float]]:
    """
    Estimate horizon angle and centre-y position in a rectilinear preview.

    Returns:
        (angle_deg, y_at_center), or None if no stable horizontal line is found.
    """
    gray = cv2.cvtColor(view_rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    min_line_len = max(40, int(view_rgb.shape[1] * 0.18))
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=80,
        minLineLength=min_line_len,
        maxLineGap=12,
    )
    if lines is None:
        return None

    center_x = view_rgb.shape[1] / 2.0
    angles: List[float] = []
    y_values: List[float] = []
    weights: List[float] = []

    for x1, y1, x2, y2 in lines[:, 0]:
        dx = float(x2 - x1)
        dy = float(y2 - y1)
        if abs(dx) < 1e-6:
            continue
        angle = math.degrees(math.atan2(dy, dx))
        if not -20.0 <= angle <= 20.0:
            continue
        length = float(math.hypot(dx, dy))
        slope = dy / dx
        y_at_center = float(y1) + slope * (center_x - float(x1))
        angles.append(angle)
        y_values.append(y_at_center)
        weights.append(length)

    if not angles:
        return None

    weights_np = np.array(weights, dtype=np.float32)
    return (
        float(np.average(np.array(angles, dtype=np.float32), weights=weights_np)),
        float(np.average(np.array(y_values, dtype=np.float32), weights=weights_np)),
    )


def align_perspective_horizon(view_rgb: np.ndarray) -> np.ndarray:
    """
    Rotate and translate a perspective preview so the detected horizon is level
    and vertically centred. If no stable horizon is found, returns the input.
    """
    horizon = _estimate_horizon_line(view_rgb)
    if horizon is None:
        return view_rgb

    angle_deg, y_at_center = horizon
    h, w = view_rgb.shape[:2]
    center = (w / 2.0, h / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    matrix[1, 2] += (h / 2.0) - y_at_center
    return cv2.warpAffine(
        view_rgb,
        matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )


def export_perspective_views(
    pano_np: np.ndarray,
    output_dir: str,
    prefix: str,
    yaws: List[float],
    pitch_deg: float = 0.0,
    fov_deg: float = 90.0,
    out_size: Tuple[int, int] = (768, 768),
    align_horizon: bool = False,
    write_360_strip: bool = True,
) -> List[str]:
    """Save perspective previews for a panorama and return written file paths."""
    os.makedirs(output_dir, exist_ok=True)
    written: List[str] = []
    strip_views: List[np.ndarray] = []
    for yaw in yaws:
        view = equirectangular_to_perspective(
            pano_np,
            yaw_deg=yaw,
            pitch_deg=pitch_deg,
            fov_deg=fov_deg,
            out_size=out_size,
        )
        if align_horizon:
            view = align_perspective_horizon(view)
        strip_views.append(view)
        yaw_label = int(round(yaw)) % 360
        pitch_label = int(round(pitch_deg))
        path = os.path.join(
            output_dir,
            f"{prefix}_yaw_{yaw_label:03d}_pitch_{pitch_label:+03d}.png",
        )
        Image.fromarray(view).save(path)
        written.append(path)
    if write_360_strip and strip_views:
        strip_path = os.path.join(output_dir, f"{prefix}_360_strip.png")
        Image.fromarray(np.hstack(strip_views)).save(strip_path)
        written.append(strip_path)
    return written


# ---------- Yaw search ----------


def _evaluate_yaw(
    yaw_deg: int,
    right_crop: np.ndarray,
    feats_left,
    device: torch.device,
    extractor,
    matcher,
    left_keypoint_mask: Optional[np.ndarray] = None,
    right_keypoint_mask: Optional[np.ndarray] = None,
) -> Tuple[float, int, float]:
    """
    Evaluate a single yaw candidate using pre-extracted left features.

    Returns:
        score (lower is better), num_matches, avg_distance between matched keypoints.
    """
    kp1 = feats_left["keypoints"][0].detach().cpu().numpy()

    shifted_rgb = shift_equirectangular(right_crop, yaw_deg)
    right_gray = np.array(Image.fromarray(shifted_rgb).convert("L"))
    right_tensor = load_image(right_gray, device)

    with torch.no_grad():
        feats_right = extractor.extract(right_tensor)
        matches_dict = matcher({"image0": feats_left, "image1": feats_right})
        matches = rbd(matches_dict)["matches"]

    kp2 = feats_right["keypoints"][0].detach().cpu().numpy()
    matches_np = matches.detach().cpu().numpy()
    shifted_right_mask = (
        shift_equirectangular(right_keypoint_mask, yaw_deg)
        if right_keypoint_mask is not None
        else None
    )
    kp1_allowed = _keypoints_allowed(kp1, left_keypoint_mask)
    kp2_allowed = _keypoints_allowed(kp2, shifted_right_mask)

    valid_pairs: List[Tuple[int, int]] = [
        (int(i), int(j))
        for i, j in matches_np
        if (
            i != -1
            and j != -1
            and i < len(kp1)
            and j < len(kp2)
            and kp1_allowed[int(i)]
            and kp2_allowed[int(j)]
        )
    ]

    num_matches = len(valid_pairs)
    if num_matches == 0:
        return float("inf"), 0, float("inf")

    pts1 = np.float32([kp1[i] for i, _ in valid_pairs])
    pts2 = np.float32([kp2[j] for _, j in valid_pairs])
    dists = np.linalg.norm(pts1 - pts2, axis=1)
    avg_dist = float(dists.mean()) if dists.size > 0 else float("inf")

    # Lower score = smaller distances + more matches
    alpha = 0.3
    score = avg_dist / (1.0 + alpha * math.log(1.0 + num_matches))

    return float(score), num_matches, avg_dist


def _circular_distance_deg(a: np.ndarray | float, b: float) -> np.ndarray | float:
    """Smallest absolute circular distance between angle(s) a and b in degrees."""
    return np.abs(((a - b + 180.0) % 360.0) - 180.0)


def _circular_mean_deg(angles: np.ndarray) -> float:
    """Circular mean of angles in degrees, returned in [0, 360)."""
    radians = np.deg2rad(angles)
    mean_sin = float(np.sin(radians).mean())
    mean_cos = float(np.cos(radians).mean())
    return float(np.rad2deg(np.arctan2(mean_sin, mean_cos)) % 360.0)


def _estimate_initial_yaw_from_matches(
    right_crop: np.ndarray,
    feats_left,
    device: torch.device,
    extractor,
    matcher,
    yaw_step: int,
    min_support: int = 8,
    min_support_ratio: float = 0.25,
    left_keypoint_mask: Optional[np.ndarray] = None,
    right_keypoint_mask: Optional[np.ndarray] = None,
) -> Optional[Dict[str, float]]:
    """
    Estimate panorama yaw from one unshifted LightGlue match pass.

    For equirectangular panoramas, horizontal keypoint displacement maps to yaw:
    shifting the right panorama by yaw aligns x_right - yaw_px with x_left.
    """
    right_gray = np.array(Image.fromarray(right_crop).convert("L"))
    right_tensor = load_image(right_gray, device)

    with torch.no_grad():
        feats_right = extractor.extract(right_tensor)
        matches_dict = matcher({"image0": feats_left, "image1": feats_right})
        matches = rbd(matches_dict)["matches"]

    kp1 = feats_left["keypoints"][0].detach().cpu().numpy()
    kp2 = feats_right["keypoints"][0].detach().cpu().numpy()
    matches_np = matches.detach().cpu().numpy()
    kp1_allowed = _keypoints_allowed(kp1, left_keypoint_mask)
    kp2_allowed = _keypoints_allowed(kp2, right_keypoint_mask)

    valid_pairs: List[Tuple[int, int]] = [
        (int(i), int(j))
        for i, j in matches_np
        if (
            i != -1
            and j != -1
            and i < len(kp1)
            and j < len(kp2)
            and kp1_allowed[int(i)]
            and kp2_allowed[int(j)]
        )
    ]
    if len(valid_pairs) < min_support:
        return None

    width = float(right_crop.shape[1])
    yaw_offsets = np.array(
        [((kp2[j][0] - kp1[i][0]) / width * 360.0) % 360.0 for i, j in valid_pairs],
        dtype=np.float32,
    )

    bin_width = max(1.0, float(yaw_step))
    bins = np.arange(0.0, 360.0 + bin_width, bin_width)
    hist, edges = np.histogram(yaw_offsets, bins=bins)
    if hist.size == 0:
        return None

    best_bin = int(hist.argmax())
    bin_center = float((edges[best_bin] + edges[best_bin + 1]) / 2.0) % 360.0
    support_window = max(bin_width * 1.5, 8.0)
    support_mask = _circular_distance_deg(yaw_offsets, bin_center) <= support_window
    support = int(support_mask.sum())
    support_ratio = float(support / len(yaw_offsets))

    if support < min_support or support_ratio < min_support_ratio:
        return None

    yaw_estimate = _circular_mean_deg(yaw_offsets[support_mask])
    concentration = 1.0 - float(
        _circular_distance_deg(yaw_offsets[support_mask], yaw_estimate).mean() / 180.0
    )

    return {
        "yaw": yaw_estimate,
        "matches": float(len(valid_pairs)),
        "support": float(support),
        "support_ratio": support_ratio,
        "support_window": float(support_window),
        "concentration": concentration,
    }


def estimate_yaw_from_keypoints_once(
    left_rgb: np.ndarray,
    right_rgb: np.ndarray,
    device,
    extractor,
    matcher,
    yaw_step: int = 10,
    crop_keep: float = 0.55,
    crop_top_bias: float = 0.0,
    left_keypoint_mask: Optional[np.ndarray] = None,
    right_keypoint_mask: Optional[np.ndarray] = None,
) -> Tuple[int, str]:
    """
    Estimate panorama yaw from one unshifted LightGlue match pass.

    This is the fastest panorama alignment path: no coarse/fine yaw search.
    It matches keypoints once, estimates horizontal equirectangular offset from
    matched x-coordinates, and returns that yaw. If support is weak, it returns
    yaw 0 with a method label that makes the fallback explicit.
    """
    left_crop = crop_equirectangular_middle(
        left_rgb, keep_ratio=crop_keep, top_bias=crop_top_bias
    )
    right_crop = crop_equirectangular_middle(
        right_rgb, keep_ratio=crop_keep, top_bias=crop_top_bias
    )

    left_gray = np.array(Image.fromarray(left_crop).convert("L"))
    left_tensor = load_image(left_gray, device)
    with torch.no_grad():
        feats_left = extractor.extract(left_tensor)

    initial = _estimate_initial_yaw_from_matches(
        right_crop,
        feats_left,
        device,
        extractor,
        matcher,
        yaw_step=yaw_step,
        left_keypoint_mask=left_keypoint_mask,
        right_keypoint_mask=right_keypoint_mask,
    )
    if initial is None:
        return 0, "keypoint_one_shot_failed_yaw0"
    return int(round(initial["yaw"])) % 360, "keypoint_one_shot"


def estimate_best_yaw(
    left_rgb: np.ndarray,
    right_rgb: np.ndarray,
    device,
    extractor,
    matcher,
    yaw_step: int = 10,
    crop_keep: float = 0.55,
    crop_top_bias: float = 0.0,
    original_yaw_deg: int = 0,
    debug_scores_path: Optional[str] = None,
    left_keypoint_mask: Optional[np.ndarray] = None,
    right_keypoint_mask: Optional[np.ndarray] = None,
) -> Tuple[int, str]:
    """
    Find best yaw for the RIGHT panorama using a hybrid keypoint+search strategy.

    Strategy:
        1. Crop vertically (to remove sky/ground) using `crop_equirectangular_middle`.
        2. Extract left features once (shared across all yaw candidates).
        3. Try a one-shot keypoint yaw estimate from unshifted panorama matches.
        4. If the estimate has enough support, refine locally around it.
        5. Otherwise, evaluate LightGlue-based score at a coarse grid of angles
           and refine around the best coarse yaw.

    The score per yaw combines average keypoint distance and number of matches.
    Lower score is better.

    Args:
        left_rgb, right_rgb: full-resolution RGB panoramas (np.uint8, HxWx3).
        device: torch.device.
        extractor: keypoint+descriptor extractor (LightGlue backbone).
        matcher: LightGlue matcher.
        yaw_step: coarse step in degrees (e.g. 10).
        crop_keep: vertical height fraction to keep.
        crop_top_bias: vertical crop bias.
        original_yaw_deg: yaw you started with (0 if unknown).
        debug_scores_path: if set, write a JSON file with per-yaw scores.

    Returns:
        (best_yaw, alignment_method), where best_yaw is an int degree in
        [0, 360), and alignment_method describes which search path was used:
        "keypoint_one_shot_refine" or "coarse_fine_search".
    """
    left_crop = crop_equirectangular_middle(
        left_rgb, keep_ratio=crop_keep, top_bias=crop_top_bias
    )
    right_crop = crop_equirectangular_middle(
        right_rgb, keep_ratio=crop_keep, top_bias=crop_top_bias
    )

    # Extract left features once — reused for every yaw candidate
    left_gray = np.array(Image.fromarray(left_crop).convert("L"))
    left_tensor = load_image(left_gray, device)
    with torch.no_grad():
        feats_left = extractor.extract(left_tensor)

    yaw_debug: Dict[int, List[Dict[str, float]]] = {}

    best_yaw = original_yaw_deg % 360
    best_score = float("inf")
    alignment_method = "coarse_fine_search"
    fine_step = max(1, yaw_step // 4)
    fine_range = 2 * yaw_step

    initial = _estimate_initial_yaw_from_matches(
        right_crop,
        feats_left,
        device,
        extractor,
        matcher,
        yaw_step=yaw_step,
        left_keypoint_mask=left_keypoint_mask,
        right_keypoint_mask=right_keypoint_mask,
    )

    if initial is not None:
        initial_yaw = int(round(initial["yaw"])) % 360
        yaw_debug.setdefault(initial_yaw, []).append(
            {
                "stage": "initial_estimate",
                "score": float("nan"),
                "num_matches": int(initial["matches"]),
                "avg_dist": float("nan"),
                "support": int(initial["support"]),
                "support_ratio": float(initial["support_ratio"]),
                "support_window": float(initial["support_window"]),
                "concentration": float(initial["concentration"]),
            }
        )

        local_yaws: List[int] = []
        for delta in range(-fine_range, fine_range + 1, fine_step):
            y = (initial_yaw + delta) % 360
            if y not in local_yaws:
                local_yaws.append(y)

        for yaw in local_yaws:
            score, num_matches, avg_dist = _evaluate_yaw(
                yaw,
                right_crop,
                feats_left,
                device,
                extractor,
                matcher,
                left_keypoint_mask=left_keypoint_mask,
                right_keypoint_mask=right_keypoint_mask,
            )
            yaw_debug.setdefault(yaw, []).append(
                {
                    "stage": "initial_refine",
                    "score": float(score),
                    "num_matches": int(num_matches),
                    "avg_dist": float(avg_dist),
                }
            )

            if score < best_score:
                best_score = score
                best_yaw = yaw
                alignment_method = "keypoint_one_shot_refine"

    if not np.isfinite(best_score):
        alignment_method = "coarse_fine_search"
        coarse_yaws: List[int] = list(range(0, 360, yaw_step))
        if original_yaw_deg % 360 not in coarse_yaws:
            coarse_yaws.insert(0, original_yaw_deg % 360)
        else:
            coarse_yaws = [original_yaw_deg % 360] + [
                y for y in coarse_yaws if y != (original_yaw_deg % 360)
            ]

        # ---- coarse search fallback ----
        for yaw in coarse_yaws:
            score, num_matches, avg_dist = _evaluate_yaw(
                yaw,
                right_crop,
                feats_left,
                device,
                extractor,
                matcher,
                left_keypoint_mask=left_keypoint_mask,
                right_keypoint_mask=right_keypoint_mask,
            )
            yaw_debug.setdefault(yaw, []).append(
                {
                    "stage": "coarse",
                    "score": float(score),
                    "num_matches": int(num_matches),
                    "avg_dist": float(avg_dist),
                }
            )

            if score < best_score:
                best_score = score
                best_yaw = yaw

        # ---- fine search around coarse best ----
        fine_yaws: List[int] = []
        for delta in range(-fine_range, fine_range + 1, fine_step):
            y = (best_yaw + delta) % 360
            if y not in yaw_debug:
                fine_yaws.append(y)

        for yaw in fine_yaws:
            score, num_matches, avg_dist = _evaluate_yaw(
                yaw,
                right_crop,
                feats_left,
                device,
                extractor,
                matcher,
                left_keypoint_mask=left_keypoint_mask,
                right_keypoint_mask=right_keypoint_mask,
            )
            yaw_debug.setdefault(yaw, []).append(
                {
                    "stage": "fine",
                    "score": float(score),
                    "num_matches": int(num_matches),
                    "avg_dist": float(avg_dist),
                }
            )

            if score < best_score:
                best_score = score
                best_yaw = yaw

    if debug_scores_path is not None:
        try:
            dir_name = os.path.dirname(debug_scores_path)
            if dir_name:
                os.makedirs(dir_name, exist_ok=True)
            with open(debug_scores_path, "w", encoding="utf-8") as f:
                json.dump(
                    {int(k): v for k, v in yaw_debug.items()},
                    f,
                    indent=2,
                )
        except Exception as e:
            logger.warning("Failed to write yaw debug file: %s", e)

    return int(best_yaw % 360), alignment_method


# ---------- FOV alignment ----------


def _robust_axis_mask(values: np.ndarray, iqr_multiplier: float = 1.5) -> np.ndarray:
    """Return a mask that excludes isolated 1-D outliers using Tukey fences."""
    if values.size < 4:
        return np.ones(values.shape, dtype=bool)

    q1, q3 = np.percentile(values, [25, 75])
    iqr = q3 - q1
    if iqr <= 1e-6:
        return np.ones(values.shape, dtype=bool)

    lower = q1 - iqr_multiplier * iqr
    upper = q3 + iqr_multiplier * iqr
    return (values >= lower) & (values <= upper)


def _robust_pair_mask(pts1: np.ndarray, pts2: np.ndarray) -> np.ndarray:
    """
    Keep matched pairs that are spatially plausible in both images.

    A single far-away match should not force either crop to include a mostly
    unmatched region, so outlier checks are applied independently to x/y in
    both images and then combined at the match-pair level.
    """
    mask = np.ones(len(pts1), dtype=bool)
    for pts in (pts1, pts2):
        mask &= _robust_axis_mask(pts[:, 0])
        mask &= _robust_axis_mask(pts[:, 1])
    return mask


def _crop_box_for_points(
    image_shape: tuple[int, int],
    pts: np.ndarray,
    padding_fraction: float = 0.02,
) -> tuple[int, int, int, int]:
    """Return x1, y1, x2, y2 crop bounds for pts, with small relative padding."""
    h, w = image_shape
    x1 = float(pts[:, 0].min())
    x2 = float(pts[:, 0].max())
    y1 = float(pts[:, 1].min())
    y2 = float(pts[:, 1].max())

    pad_x = max(1.0, (x2 - x1) * padding_fraction)
    pad_y = max(1.0, (y2 - y1) * padding_fraction)

    crop_x1 = max(0, int(math.floor(x1 - pad_x)))
    crop_x2 = min(w, int(math.ceil(x2 + pad_x)))
    crop_y1 = max(0, int(math.floor(y1 - pad_y)))
    crop_y2 = min(h, int(math.ceil(y2 + pad_y)))

    return crop_x1, crop_y1, crop_x2, crop_y2


def _crop_to_points(
    image_rgb: np.ndarray,
    pts: np.ndarray,
    padding_fraction: float = 0.02,
) -> Tuple[np.ndarray, float]:
    """
    Crop image to the bounding box covered by pts, with small relative padding.

    Returns the cropped image and retained-area fraction.
    """
    cropped, retained_fraction, _ = _crop_to_points_with_box(
        image_rgb, pts, padding_fraction=padding_fraction
    )
    return cropped, retained_fraction


def _crop_to_points_with_box(
    image_np: np.ndarray,
    pts: np.ndarray,
    padding_fraction: float = 0.02,
) -> Tuple[np.ndarray, float, tuple[int, int, int, int]]:
    """
    Crop an array to the bounding box covered by pts.

    Returns the cropped array, retained-area fraction, and crop box
    (x1, y1, x2, y2). The same box can be applied to aligned masks.
    """
    h, w = image_np.shape[:2]
    crop_x1, crop_y1, crop_x2, crop_y2 = _crop_box_for_points(
        (h, w), pts, padding_fraction=padding_fraction
    )

    if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
        return image_np, 1.0, (0, 0, w, h)

    retained_fraction = (crop_y2 - crop_y1) * (crop_x2 - crop_x1) / (h * w)
    return (
        image_np[crop_y1:crop_y2, crop_x1:crop_x2],
        float(retained_fraction),
        (crop_x1, crop_y1, crop_x2, crop_y2),
    )


def _homography_distortion_score(
    homography: np.ndarray,
    source_shape: tuple[int, int],
    valid_fraction: float,
) -> float:
    """Lower means the homography changes the source image geometry less."""
    h_src, w_src = source_shape
    corners = np.float32(
        [[0, 0], [w_src, 0], [w_src, h_src], [0, h_src]]
    ).reshape(-1, 1, 2)
    warped = cv2.perspectiveTransform(corners, homography).reshape(4, 2)

    source_area = float(w_src * h_src)
    warped_area = abs(float(cv2.contourArea(warped.astype(np.float32))))
    if source_area <= 0 or warped_area <= 1e-6:
        return float("inf")

    src_edges = np.array([w_src, h_src, w_src, h_src], dtype=np.float32)
    warped_edges = np.array(
        [
            np.linalg.norm(warped[(i + 1) % 4] - warped[i])
            for i in range(4)
        ],
        dtype=np.float32,
    )
    if np.any(warped_edges <= 1e-6):
        return float("inf")

    edge_ratios = warped_edges / src_edges
    area_scale = warped_area / source_area

    log_area = abs(math.log(max(area_scale, 1e-6)))
    log_edge = float(np.mean(np.abs(np.log(np.maximum(edge_ratios, 1e-6)))))
    anisotropy = float(np.std(np.log(np.maximum(edge_ratios, 1e-6))))

    # Reward usable overlap a little, but keep the score mostly about distortion.
    overlap_penalty = 0.25 * (1.0 - max(0.0, min(1.0, valid_fraction)))
    return log_area + log_edge + anisotropy + overlap_penalty


def _warp_candidate(
    *,
    warp_side: str,
    source_rgb: np.ndarray,
    target_rgb: np.ndarray,
    source_pts: np.ndarray,
    target_pts: np.ndarray,
    min_matches: int,
    padding_fraction: float = 0.02,
    source_aux: Optional[np.ndarray] = None,
    target_aux: Optional[np.ndarray] = None,
) -> (
    tuple[
        float,
        Tuple[np.ndarray, np.ndarray, float, float, float],
        np.ndarray,
        np.ndarray,
        Dict[str, object],
    ]
    | None
):
    """Build and score one homography warp candidate."""
    h_source, w_source = source_rgb.shape[:2]
    h_target, w_target = target_rgb.shape[:2]

    homography, inlier_mask = cv2.findHomography(
        source_pts.reshape(-1, 1, 2),
        target_pts.reshape(-1, 1, 2),
        cv2.RANSAC,
        5.0,
    )
    if homography is None or inlier_mask is None:
        logger.debug("fov_crop_align: reprojection homography failed for %s", warp_side)
        return None

    inliers = inlier_mask.ravel().astype(bool)
    if int(inliers.sum()) < min_matches:
        logger.debug(
            "fov_crop_align: only %d homography inliers for %s",
            int(inliers.sum()),
            warp_side,
        )
        return None

    warped_source = cv2.warpPerspective(source_rgb, homography, (w_target, h_target))
    source_mask = np.ones((h_source, w_source), dtype=np.uint8) * 255
    warped_mask = cv2.warpPerspective(source_mask, homography, (w_target, h_target))
    warped_source_aux = None
    if source_aux is not None:
        warped_source_aux = cv2.warpPerspective(
            source_aux,
            homography,
            (w_target, h_target),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=255,
        )

    inlier_target_pts = target_pts[inliers]
    crop_x1, crop_y1, crop_x2, crop_y2 = _crop_box_for_points(
        (h_target, w_target), inlier_target_pts, padding_fraction=padding_fraction
    )
    if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
        logger.debug("fov_crop_align: degenerate reprojection crop box for %s", warp_side)
        return None

    cropped_mask = warped_mask[crop_y1:crop_y2, crop_x1:crop_x2]
    if not np.any(cropped_mask):
        logger.debug("fov_crop_align: reprojection crop has no valid warped pixels for %s", warp_side)
        return None

    cropped_target = target_rgb[crop_y1:crop_y2, crop_x1:crop_x2]
    cropped_source = warped_source[crop_y1:crop_y2, crop_x1:crop_x2]
    cropped_valid = cropped_mask > 0

    crop_area = (crop_y2 - crop_y1) * (crop_x2 - crop_x1)
    target_fraction = crop_area / (h_target * w_target)
    source_fraction = float(np.count_nonzero(cropped_mask)) / (h_source * w_source)
    fov_crop_fraction = min(source_fraction, target_fraction)
    score = _homography_distortion_score(
        homography,
        (h_source, w_source),
        valid_fraction=float(np.count_nonzero(cropped_mask)) / max(1, crop_area),
    )

    logger.debug(
        "fov_crop_align: %s candidate score=%.3f with %d/%d inliers",
        warp_side,
        score,
        int(inliers.sum()),
        len(inliers),
    )

    if warp_side == "left_to_right":
        result = (
            cropped_source,
            cropped_target,
            fov_crop_fraction,
            source_fraction,
            float(target_fraction),
        )
        reprojected_left = warped_source
        reprojected_right = target_rgb
        reprojected_left_aux = warped_source_aux
        reprojected_right_aux = target_aux
        cropped_left_aux = (
            warped_source_aux[crop_y1:crop_y2, crop_x1:crop_x2]
            if warped_source_aux is not None
            else None
        )
        cropped_right_aux = (
            target_aux[crop_y1:crop_y2, crop_x1:crop_x2]
            if target_aux is not None
            else None
        )
    elif warp_side == "right_to_left":
        result = (
            cropped_target,
            cropped_source,
            fov_crop_fraction,
            float(target_fraction),
            source_fraction,
        )
        reprojected_left = target_rgb
        reprojected_right = warped_source
        reprojected_left_aux = target_aux
        reprojected_right_aux = warped_source_aux
        cropped_left_aux = (
            target_aux[crop_y1:crop_y2, crop_x1:crop_x2]
            if target_aux is not None
            else None
        )
        cropped_right_aux = (
            warped_source_aux[crop_y1:crop_y2, crop_x1:crop_x2]
            if warped_source_aux is not None
            else None
        )
    else:
        raise ValueError(f"Unknown warp side: {warp_side}")

    metadata: Dict[str, object] = {
        "reprojected": True,
        "warp_side": warp_side,
        "homography": homography,
        "crop_box": (crop_x1, crop_y1, crop_x2, crop_y2),
        "reprojected_left_aux": reprojected_left_aux,
        "reprojected_right_aux": reprojected_right_aux,
        "left_aux": cropped_left_aux,
        "right_aux": cropped_right_aux,
        "reprojected_valid_mask": warped_mask > 0,
        "valid_mask": cropped_valid,
    }

    return score, result, reprojected_left, reprojected_right, metadata


def _warp_least_distorted_and_crop(
    left_rgb: np.ndarray,
    right_rgb: np.ndarray,
    pts1: np.ndarray,
    pts2: np.ndarray,
    min_matches: int,
    padding_fraction: float = 0.02,
    debug_dir: Optional[str] = None,
    left_aux: Optional[np.ndarray] = None,
    right_aux: Optional[np.ndarray] = None,
) -> tuple[Tuple[np.ndarray, np.ndarray, float, float, float], Dict[str, object]] | None:
    """Try both warp directions and return the less distorted reprojection."""
    candidates = [
        _warp_candidate(
            warp_side="right_to_left",
            source_rgb=right_rgb,
            target_rgb=left_rgb,
            source_pts=pts2,
            target_pts=pts1,
            min_matches=min_matches,
            padding_fraction=padding_fraction,
            source_aux=right_aux,
            target_aux=left_aux,
        ),
        _warp_candidate(
            warp_side="left_to_right",
            source_rgb=left_rgb,
            target_rgb=right_rgb,
            source_pts=pts1,
            target_pts=pts2,
            min_matches=min_matches,
            padding_fraction=padding_fraction,
            source_aux=left_aux,
            target_aux=right_aux,
        ),
    ]
    candidates = [candidate for candidate in candidates if candidate is not None]
    if not candidates:
        return None

    best_score, best_result, reprojected_left, reprojected_right, metadata = min(
        candidates, key=lambda candidate: candidate[0]
    )
    logger.debug("fov_crop_align: selected reprojection candidate score=%.3f", best_score)

    if debug_dir is not None:
        try:
            os.makedirs(debug_dir, exist_ok=True)
            Image.fromarray(reprojected_left).save(
                os.path.join(debug_dir, "left_reprojected.png")
            )
            Image.fromarray(reprojected_right).save(
                os.path.join(debug_dir, "right_reprojected.png")
            )
        except Exception as e:
            logger.warning("Failed to write reprojection debug images: %s", e)

    return best_result, metadata


def fov_crop_align(
    left_rgb: np.ndarray,
    right_rgb: np.ndarray,
    device: torch.device,
    extractor,
    matcher,
    min_matches: int = 4,
    reproject: bool = False,
    reproject_debug_dir: Optional[str] = None,
    left_aux: Optional[np.ndarray] = None,
    right_aux: Optional[np.ndarray] = None,
    return_metadata: bool = False,
) -> Tuple[np.ndarray, np.ndarray, float, float, float] | tuple[
    np.ndarray,
    np.ndarray,
    float,
    float,
    float,
    Dict[str, object],
]:
    """
    Align two same-spot images by cropping both to their robust matched area.

    Algorithm:
        1. Match keypoints at native resolution.
        2. Remove isolated outlier matches using robust x/y bounds in both images.
        3. Optionally estimate homographies in both directions with RANSAC and
           use the warp direction with the lower geometric distortion.
        4. Crop each image to the bounding box covered by the remaining matched
           keypoints, or to the same inlier area when reprojecting.

    Returns:
        (best_left_rgb, best_right_rgb, fov_crop_fraction,
         left_retained_fraction, right_retained_fraction)

        Both returned images may be cropped.
        fov_crop_fraction is the smaller retained-area fraction of the two crops
        (1.0 when there are too few keypoints or the crop is effectively unchanged).
    """
    left_gray = np.array(Image.fromarray(left_rgb).convert("L"))
    right_gray = np.array(Image.fromarray(right_rgb).convert("L"))

    with torch.no_grad():
        feats_left = extractor.extract(load_image(left_gray, device))
        feats_right = extractor.extract(load_image(right_gray, device))
        matches_dict = matcher({"image0": feats_left, "image1": feats_right})
        matches = rbd(matches_dict)["matches"]

    kp1 = feats_left["keypoints"][0].detach().cpu().numpy()   # (N, 2) x,y pixel coords
    kp2 = feats_right["keypoints"][0].detach().cpu().numpy()  # (M, 2)
    matches_np = matches.detach().cpu().numpy()

    valid: List[Tuple[int, int]] = [
        (int(i), int(j))
        for i, j in matches_np
        if i != -1 and j != -1 and i < len(kp1) and j < len(kp2)
    ]

    metadata: Dict[str, object] = {
        "reprojected": False,
        "warp_side": None,
        "crop_box_left": (0, 0, left_rgb.shape[1], left_rgb.shape[0]),
        "crop_box_right": (0, 0, right_rgb.shape[1], right_rgb.shape[0]),
        "left_aux": left_aux,
        "right_aux": right_aux,
        "reprojected_left_aux": None,
        "reprojected_right_aux": None,
        "valid_mask": None,
        "reprojected_valid_mask": None,
    }

    if len(valid) < min_matches:
        logger.debug("fov_crop_align: only %d matches, skipping crop", len(valid))
        result = (left_rgb, right_rgb, 1.0, 1.0, 1.0)
        return (*result, metadata) if return_metadata else result

    pts1 = np.array([kp1[i] for i, _ in valid], dtype=np.float32)  # (N, 2) x,y
    pts2 = np.array([kp2[j] for _, j in valid], dtype=np.float32)

    robust_mask = _robust_pair_mask(pts1, pts2)
    if int(robust_mask.sum()) < min_matches:
        logger.debug(
            "fov_crop_align: only %d robust matches after outlier removal, skipping crop",
            int(robust_mask.sum()),
        )
        result = (left_rgb, right_rgb, 1.0, 1.0, 1.0)
        return (*result, metadata) if return_metadata else result

    robust_pts1 = pts1[robust_mask]
    robust_pts2 = pts2[robust_mask]

    if reproject:
        reprojected = _warp_least_distorted_and_crop(
            left_rgb,
            right_rgb,
            robust_pts1,
            robust_pts2,
            min_matches=min_matches,
            debug_dir=reproject_debug_dir,
            left_aux=left_aux,
            right_aux=right_aux,
        )
        if reprojected is not None:
            result, metadata = reprojected
            return (*result, metadata) if return_metadata else result

    cropped_left, left_fraction, crop_box_left = _crop_to_points_with_box(
        left_rgb, robust_pts1
    )
    cropped_right, right_fraction, crop_box_right = _crop_to_points_with_box(
        right_rgb, robust_pts2
    )
    fov_crop_fraction = min(left_fraction, right_fraction)
    lx1, ly1, lx2, ly2 = crop_box_left
    rx1, ry1, rx2, ry2 = crop_box_right
    if left_aux is not None:
        metadata["left_aux"] = left_aux[ly1:ly2, lx1:lx2]
    if right_aux is not None:
        metadata["right_aux"] = right_aux[ry1:ry2, rx1:rx2]
    metadata["crop_box_left"] = crop_box_left
    metadata["crop_box_right"] = crop_box_right

    logger.debug(
        "fov_crop_align: matches=%d robust_matches=%d left_fraction=%.3f right_fraction=%.3f crop_fraction=%.3f",
        len(valid),
        int(robust_mask.sum()),
        left_fraction,
        right_fraction,
        fov_crop_fraction,
    )

    result = (cropped_left, cropped_right, fov_crop_fraction, left_fraction, right_fraction)
    return (*result, metadata) if return_metadata else result
