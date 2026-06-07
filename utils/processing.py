# utils/processing.py
import json
import logging
import os
import math
import time
from typing import Optional, List

import numpy as np
from PIL import Image
import cv2

from .metrics import (
    estimate_horizon_angle,
    compute_image_quality_metrics,
    resize_to_smallest,
)
from .io_utils import ensure_dir

logger = logging.getLogger(__name__)


def _lightglue_runtime():
    import torch
    from lightglue.utils import rbd

    from .tensor_utils import load_image

    return torch, rbd, load_image


def _pair_output_dir(root: str, id_left: str, id_right: str) -> str:
    """Return sharded pair output path: root/first2/id_left_id_right."""
    shard = str(id_left)[:2] or "no_id"
    return os.path.join(root, shard, f"{id_left}_{id_right}")


class PairTimer:
    """Small per-pair stage timer used only when profiling is enabled."""

    def __init__(self, enabled: bool):
        self.enabled = enabled
        self._last = time.perf_counter()
        self._total_start = self._last
        self.rows: list[tuple[str, float]] = []

    def mark(self, stage: str) -> None:
        if not self.enabled:
            return
        now = time.perf_counter()
        self.rows.append((stage, now - self._last))
        self._last = now

    def finish(self) -> list[tuple[str, float]]:
        if not self.enabled:
            return []
        now = time.perf_counter()
        self.rows.append(("total", now - self._total_start))
        return self.rows


def _match_keypoints(
    gray_left: np.ndarray,
    gray_right: np.ndarray,
    device,
    extractor,
    matcher,
) -> tuple[np.ndarray, np.ndarray]:
    """Return matched keypoint coordinates as (left_pts, right_pts)."""
    torch, rbd, load_image = _lightglue_runtime()
    left_tensor = load_image(gray_left, device)
    right_tensor = load_image(gray_right, device)

    with torch.no_grad():
        feats_left = extractor.extract(left_tensor)
        feats_right = extractor.extract(right_tensor)
        matches_dict = matcher({"image0": feats_left, "image1": feats_right})
        feats_left, feats_right, matches_dict = [
            rbd(x) for x in (feats_left, feats_right, matches_dict)
        ]

    kp_left = feats_left["keypoints"].detach().cpu().numpy()
    kp_right = feats_right["keypoints"].detach().cpu().numpy()
    matches_np = matches_dict["matches"].detach().cpu().numpy()
    pairs = [
        (int(i), int(j))
        for i, j in matches_np
        if i != -1 and j != -1 and i < len(kp_left) and j < len(kp_right)
    ]
    if not pairs:
        return np.empty((0, 2), dtype=np.float32), np.empty((0, 2), dtype=np.float32)
    left_pts = np.float32([kp_left[i] for i, _ in pairs])
    right_pts = np.float32([kp_right[j] for _, j in pairs])
    return left_pts, right_pts


def _build_keypoint_valid_mask(
    seg_mask: np.ndarray,
    ignore_class_ids: List[int],
    sky_boundary_px: int = 10,
) -> np.ndarray:
    """
    Return a boolean mask where keypoints are allowed.

    Pixels outside ignored classes are always allowed. Ignored pixels are allowed
    only when they are close to a non-ignored class, preserving useful boundaries.
    """
    if not ignore_class_ids:
        return np.ones(seg_mask.shape, dtype=bool)
    ignored = np.isin(seg_mask, ignore_class_ids)
    if not ignored.any():
        return np.ones(seg_mask.shape, dtype=bool)
    if ignored.all():
        return np.zeros(seg_mask.shape, dtype=bool)

    boundary_px = max(0, int(sky_boundary_px))
    if boundary_px == 0:
        return ~ignored
    dist_to_non_sky = cv2.distanceTransform(
        ignored.astype(np.uint8),
        cv2.DIST_L2,
        3,
    )
    return (~ignored) | (dist_to_non_sky <= float(boundary_px))


def _estimate_sky_mask_heuristic(rgb: np.ndarray) -> np.ndarray:
    """
    Fast approximate sky mask for keypoint filtering.

    This intentionally avoids semantic segmentation. It only searches the upper
    part of the panorama and marks bright/blue/low-texture pixels connected to
    the top border as sky.
    """
    h, w = rgb.shape[:2]
    upper_h = max(1, int(round(h * 0.65)))
    upper = rgb[:upper_h]
    hsv = cv2.cvtColor(upper, cv2.COLOR_RGB2HSV)
    gray = cv2.cvtColor(upper, cv2.COLOR_RGB2GRAY)
    hue = hsv[..., 0]
    sat = hsv[..., 1]
    val = hsv[..., 2]
    blueish = (hue >= 85) & (hue <= 135) & (sat >= 20) & (val >= 80)
    bright_low_sat = (sat <= 70) & (val >= 145)
    edges = cv2.Canny(gray, 50, 150)
    low_texture = edges < 1
    sky_candidate = (blueish | bright_low_sat) & low_texture

    candidate_u8 = sky_candidate.astype(np.uint8)
    num_labels, labels = cv2.connectedComponents(candidate_u8, connectivity=8)
    sky_upper = np.zeros((upper_h, w), dtype=bool)
    top_labels = np.unique(labels[0, :])
    for label in top_labels:
        if label != 0:
            sky_upper |= labels == label

    sky = np.zeros((h, w), dtype=bool)
    sky[:upper_h] = sky_upper
    return sky


def _build_keypoint_valid_mask_from_sky(
    sky: np.ndarray,
    sky_boundary_px: int = 10,
) -> np.ndarray:
    """Allow non-sky and sky pixels close to non-sky boundaries."""
    if not sky.any():
        return np.ones(sky.shape, dtype=bool)
    if sky.all():
        return np.zeros(sky.shape, dtype=bool)
    boundary_px = max(0, int(sky_boundary_px))
    if boundary_px == 0:
        return ~sky
    dist_to_non_sky = cv2.distanceTransform(
        sky.astype(np.uint8),
        cv2.DIST_L2,
        3,
    )
    return (~sky) | (dist_to_non_sky <= float(boundary_px))


def _non_sky_coverage_mask(
    rgb: np.ndarray,
    seg_mask: Optional[np.ndarray] = None,
    sky_class_ids: Optional[List[int]] = None,
    valid_pixels: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Return pixels counted as coverable area for keypoint coverage."""
    if seg_mask is not None and sky_class_ids:
        coverable = ~np.isin(seg_mask, sky_class_ids)
    else:
        coverable = ~_estimate_sky_mask_heuristic(rgb)
    if valid_pixels is not None:
        if valid_pixels.shape != coverable.shape:
            valid_pixels = _resize_seg_mask(
                valid_pixels.astype(np.uint8),
                coverable.shape,
            ).astype(bool)
        coverable = coverable & valid_pixels
    if not coverable.any():
        return np.ones(rgb.shape[:2], dtype=bool)
    return coverable


def _robust_keypoint_points(
    points: np.ndarray,
    min_points: int = 4,
) -> Optional[np.ndarray]:
    """Return matched keypoints after removing isolated spatial outliers."""
    if points.shape[0] < min_points:
        return None

    pts = np.asarray(points, dtype=np.float32)
    keep = np.ones(len(pts), dtype=bool)
    for axis in (0, 1):
        values = pts[:, axis]
        q1, q3 = np.percentile(values, [25, 75])
        iqr = q3 - q1
        if iqr <= 0:
            continue
        low = q1 - 1.5 * iqr
        high = q3 + 1.5 * iqr
        keep &= (values >= low) & (values <= high)

    filtered = pts[keep]
    if filtered.shape[0] < min_points:
        filtered = pts
    if filtered.shape[0] < min_points:
        return None
    return filtered


def _matched_keypoint_coverage(
    points: np.ndarray,
    image_shape: tuple[int, int],
    coverable_mask: Optional[np.ndarray] = None,
) -> float:
    """
    Fraction of coverable image area inside the robust matched-keypoint hull.

    Sky is excluded by passing a non-sky coverable mask. Isolated spatial
    outliers are removed before the convex hull is measured.
    """
    h, w = image_shape
    filtered = _robust_keypoint_points(points)
    if filtered is None:
        return 0.0
    if coverable_mask is None:
        coverable_mask = np.ones((h, w), dtype=bool)
    elif coverable_mask.shape != (h, w):
        coverable_mask = _resize_seg_mask(
            coverable_mask.astype(np.uint8),
            (h, w),
        ).astype(bool)

    denom = int(coverable_mask.sum())
    if denom <= 0:
        denom = h * w
        coverable_mask = np.ones((h, w), dtype=bool)

    hull_points = _robust_keypoint_hull_points(points, (h, w))
    if hull_points is None:
        return 0.0

    inside = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(inside, hull_points, 1)
    inside = inside.astype(bool)
    return float((inside & coverable_mask).sum() / denom)


def _matched_keypoint_hull_iou(
    left_points: np.ndarray,
    right_points: np.ndarray,
    image_shape: tuple[int, int],
) -> float:
    """IoU between same-size robust convex hulls from matched keypoints."""
    h, w = image_shape
    left_hull = _robust_keypoint_hull_points(left_points, (h, w))
    right_hull = _robust_keypoint_hull_points(right_points, (h, w))
    if left_hull is None or right_hull is None:
        return 0.0

    left_mask = np.zeros((h, w), dtype=np.uint8)
    right_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(left_mask, left_hull, 1)
    cv2.fillConvexPoly(right_mask, right_hull, 1)
    intersection = int(np.logical_and(left_mask, right_mask).sum())
    union = int(np.logical_or(left_mask, right_mask).sum())
    return float(intersection / union) if union > 0 else 0.0


def _robust_keypoint_hull_points(
    points: np.ndarray,
    image_shape: tuple[int, int],
) -> Optional[np.ndarray]:
    """Return clipped convex-hull vertices for robust matched keypoints."""
    h, w = image_shape
    filtered = _robust_keypoint_points(points)
    if filtered is None:
        return None
    hull = cv2.convexHull(filtered.reshape(-1, 1, 2).astype(np.float32))
    hull_points = np.rint(hull.reshape(-1, 2)).astype(np.int32)
    hull_points[:, 0] = np.clip(hull_points[:, 0], 0, max(0, w - 1))
    hull_points[:, 1] = np.clip(hull_points[:, 1], 0, max(0, h - 1))
    if len(np.unique(hull_points, axis=0)) < 3:
        return None
    return hull_points


def _resize_seg_mask(mask: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    """Resize a segmentation mask to (height, width) with nearest-neighbour labels."""
    target_h, target_w = target_shape
    if mask.shape[:2] == (target_h, target_w):
        return mask
    return cv2.resize(mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)


def _segment_image_cached(
    segmenter,
    image: Image.Image,
    cache_path: Optional[str] = None,
    max_width: Optional[int] = None,
) -> np.ndarray:
    """Segment an image, optionally caching the label mask as a .npy file."""
    if cache_path and os.path.exists(cache_path):
        try:
            return np.load(cache_path)
        except Exception:
            logger.warning("Failed to read segmentation cache: %s", cache_path)
    image_for_seg = image
    if max_width and image.width > max_width:
        scale = float(max_width) / float(image.width)
        image_for_seg = image.resize(
            (int(max_width), max(1, int(round(image.height * scale)))),
            Image.Resampling.BILINEAR,
        )
    mask = segmenter.segment_image(image_for_seg)
    if cache_path:
        try:
            cache_dir = os.path.dirname(cache_path)
            if cache_dir:
                os.makedirs(cache_dir, exist_ok=True)
            np.save(cache_path, mask)
        except Exception as e:
            logger.warning("Failed to write segmentation cache %s: %s", cache_path, e)
    return mask


_SEG_INVALID_CLASS_ID = 255


def _empty_seg_summary() -> tuple[str, str, str, str, str, str, str, str]:
    """Default semantic summary tuple for disabled/unavailable segmentation."""
    return "NA", "NA", "{}", "{}", "{}", "{}", "{}", "NA"


def _crop_seg_inputs(
    left_mask: np.ndarray,
    right_mask: np.ndarray,
    seg_crop_top_frac: float,
    seg_crop_bottom_frac: float,
    valid_pixels: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Apply the same optional vertical segmentation crop to masks and validity."""
    if left_mask.shape != right_mask.shape:
        right_mask = _resize_seg_mask(right_mask, left_mask.shape)
    if valid_pixels is not None and valid_pixels.shape != left_mask.shape:
        valid_pixels = _resize_seg_mask(valid_pixels.astype(np.uint8), left_mask.shape).astype(bool)

    if seg_crop_top_frac <= 0.0 and seg_crop_bottom_frac <= 0.0:
        return left_mask, right_mask, valid_pixels

    h = left_mask.shape[0]
    top_frac = max(0.0, min(1.0, seg_crop_top_frac))
    bottom_frac = max(0.0, min(1.0, seg_crop_bottom_frac))
    start_row = int(round(h * top_frac))
    end_row = int(round(h * (1.0 - bottom_frac)))
    if end_row <= start_row:
        return left_mask, right_mask, valid_pixels

    left_mask = left_mask[start_row:end_row, :]
    right_mask = right_mask[start_row:end_row, :]
    if valid_pixels is not None:
        valid_pixels = valid_pixels[start_row:end_row, :]
    return left_mask, right_mask, valid_pixels


def _summarize_seg_pair(
    segmenter,
    left_mask: np.ndarray,
    right_mask: np.ndarray,
    seg_crop_top_frac: float = 0.0,
    seg_crop_bottom_frac: float = 0.0,
    valid_pixels: Optional[np.ndarray] = None,
) -> tuple[object, object, str, str, str, str, str, object]:
    """
    Summarize one aligned segmentation-mask pair.

    Returns IoU metrics plus class-share JSON before and after cross-image
    temporary-object masking. Pixels with label 255 or valid_pixels=False are
    excluded from all summaries.
    """
    left_use, right_use, valid_use = _crop_seg_inputs(
        left_mask,
        right_mask,
        seg_crop_top_frac,
        seg_crop_bottom_frac,
        valid_pixels=valid_pixels,
    )
    valid = (left_use != _SEG_INVALID_CLASS_ID) & (right_use != _SEG_INVALID_CLASS_ID)
    if valid_use is not None:
        valid &= valid_use

    if not valid.any():
        return _empty_seg_summary()

    if valid.all():
        left_eval = left_use
        right_eval = right_use
    else:
        left_eval = left_use[valid]
        right_eval = right_use[valid]

    mean_all, mean_road, per_class_json = segmenter.summarize_iou(left_eval, right_eval)
    (
        props_left_before_json,
        props_right_before_json,
        props_left_temp_masked_json,
        props_right_temp_masked_json,
        temp_union_fraction,
    ) = segmenter.summarize_class_proportions(left_eval, right_eval)

    return (
        float(mean_all) if mean_all is not None else "NA",
        float(mean_road) if mean_road is not None else "NA",
        per_class_json,
        props_left_before_json,
        props_right_before_json,
        props_left_temp_masked_json,
        props_right_temp_masked_json,
        float(temp_union_fraction),
    )


def _round_seg_fraction(value: object) -> object:
    if value == "NA":
        return "NA"
    return round(float(value), 6)


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


def _align_right_view_to_left_by_keypoints(
    left_view: np.ndarray,
    right_view: np.ndarray,
    device,
    extractor,
    matcher,
    min_matches: int = 8,
) -> tuple[np.ndarray, str, int]:
    """
    Align a projected right panorama view to the left view using matched
    keypoints. Returns (aligned_right, method, match_count).
    """
    left_gray = cv2.cvtColor(left_view, cv2.COLOR_RGB2GRAY)
    right_gray = cv2.cvtColor(right_view, cv2.COLOR_RGB2GRAY)
    left_pts, right_pts = _match_keypoints(
        left_gray, right_gray, device, extractor, matcher
    )
    match_count = int(len(left_pts))
    if match_count < min_matches:
        return right_view, "not_enough_matches", match_count

    matrix, inliers = cv2.estimateAffinePartial2D(
        right_pts,
        left_pts,
        method=cv2.RANSAC,
        ransacReprojThreshold=5.0,
        maxIters=2000,
        confidence=0.99,
    )
    if matrix is None:
        return right_view, "affine_failed", match_count

    h, w = left_view.shape[:2]
    aligned = cv2.warpAffine(
        right_view,
        matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    inlier_count = int(inliers.sum()) if inliers is not None else 0
    return aligned, f"affine_keypoints_{inlier_count}_inliers", match_count


def _export_pair_perspective_debug(
    *,
    left_rgb: np.ndarray,
    right_rgb: np.ndarray,
    preview_dir: str,
    yaws: List[float],
    pitch_deg: float,
    fov_deg: float,
    out_size: tuple[int, int],
    align_horizon: bool,
    keypoint_align: bool,
    device,
    extractor,
    matcher,
) -> None:
    """Export projected panorama views and optional keypoint-aligned right views."""
    from .panorama import export_perspective_views, equirectangular_to_perspective

    export_perspective_views(
        left_rgb,
        preview_dir,
        "left",
        yaws,
        pitch_deg=pitch_deg,
        fov_deg=fov_deg,
        out_size=out_size,
        align_horizon=align_horizon,
    )
    export_perspective_views(
        right_rgb,
        preview_dir,
        "right",
        yaws,
        pitch_deg=pitch_deg,
        fov_deg=fov_deg,
        out_size=out_size,
        align_horizon=align_horizon,
    )
    if not keypoint_align:
        return

    aligned_views: List[np.ndarray] = []
    metadata: list[dict] = []
    for yaw in yaws:
        left_view = equirectangular_to_perspective(
            left_rgb, yaw_deg=yaw, pitch_deg=pitch_deg, fov_deg=fov_deg, out_size=out_size
        )
        right_view = equirectangular_to_perspective(
            right_rgb, yaw_deg=yaw, pitch_deg=pitch_deg, fov_deg=fov_deg, out_size=out_size
        )
        aligned, method, matches = _align_right_view_to_left_by_keypoints(
            left_view,
            right_view,
            device,
            extractor,
            matcher,
        )
        aligned_views.append(aligned)
        yaw_label = int(round(yaw)) % 360
        path = os.path.join(
            preview_dir,
            f"right_keypoint_aligned_yaw_{yaw_label:03d}_pitch_{int(round(pitch_deg)):+03d}.png",
        )
        Image.fromarray(aligned).save(path)
        metadata.append(
            {
                "yaw": float(yaw),
                "pitch": float(pitch_deg),
                "method": method,
                "matches": matches,
                "path": os.path.basename(path),
            }
        )

    if aligned_views:
        Image.fromarray(np.hstack(aligned_views)).save(
            os.path.join(preview_dir, "right_keypoint_aligned_360_strip.png")
        )
    with open(os.path.join(preview_dir, "keypoint_alignment.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def compute_lightglue_score(
    img1_pil,
    img2_pil,
    img1_np,
    img2_np,
    gray1_np,
    gray2_np,
    device,
    extractor,
    matcher,
    left_keypoint_mask: Optional[np.ndarray] = None,
    right_keypoint_mask: Optional[np.ndarray] = None,
    left_coverage_mask: Optional[np.ndarray] = None,
    right_coverage_mask: Optional[np.ndarray] = None,
):
    """
    Compute LightGlue matching stats between two images.
    Returns:
        match_ratio, avg_distance, inliers, total, inlier_ratio,
        coverage_left, coverage_right, coverage_min, hull_iou
    """
    torch, rbd, load_image = _lightglue_runtime()
    img1_tensor = load_image(gray1_np, device)
    img2_tensor = load_image(gray2_np, device)
    match_ratio = 0.0
    avg_distance = float("inf")
    inliers = total = 0
    inlier_ratio = 0.0
    coverage_left = 0.0
    coverage_right = 0.0
    hull_iou = 0.0

    with torch.no_grad():
        feats1 = extractor.extract(img1_tensor)
        feats2 = extractor.extract(img2_tensor)
        matches_dict = matcher({"image0": feats1, "image1": feats2})
        feats1, feats2, matches_dict = [rbd(x) for x in (feats1, feats2, matches_dict)]
        matches = matches_dict["matches"]

        kp1 = feats1["keypoints"].detach().cpu().numpy()
        kp2 = feats2["keypoints"].detach().cpu().numpy()
        matches_np = matches.detach().cpu().numpy()
        kp1_allowed = _keypoints_allowed(kp1, left_keypoint_mask)
        kp2_allowed = _keypoints_allowed(kp2, right_keypoint_mask)

        valid = [
            (i, j)
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
        allowed_left_count = int(kp1_allowed.sum())
        allowed_right_count = int(kp2_allowed.sum())
        if allowed_left_count > 0 and allowed_right_count > 0:
            total_possible = min(allowed_left_count, allowed_right_count)
            match_ratio = len(valid) / total_possible if total_possible > 0 else 0.0

        if valid:
            h, w = img1_pil.size[1], img1_pil.size[0]
            diag = float(np.hypot(w, h))
            dists = [float(np.linalg.norm(kp1[i] - kp2[j]) / diag) for i, j in valid]
            if dists:
                avg_distance = float(np.mean(dists))
            left_points = np.float32([kp1[i] for i, _ in valid])
            right_points = np.float32([kp2[j] for _, j in valid])
            coverage_left = _matched_keypoint_coverage(
                left_points,
                (h, w),
                left_coverage_mask,
            )
            coverage_right = _matched_keypoint_coverage(
                right_points,
                (img2_pil.size[1], img2_pil.size[0]),
                right_coverage_mask,
            )
            hull_iou = _matched_keypoint_hull_iou(left_points, right_points, (h, w))
            if len(valid) >= 4:
                src_pts = np.float32([kp1[i] for i, _ in valid]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp2[j] for _, j in valid]).reshape(-1, 1, 2)
                _, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                if mask is not None:
                    inliers = int(mask.sum())
                    total = len(valid)
                    inlier_ratio = float(inliers / total) if total > 0 else 0.0

    coverage_min = min(coverage_left, coverage_right)
    return (
        match_ratio,
        avg_distance,
        inliers,
        total,
        inlier_ratio,
        coverage_left,
        coverage_right,
        coverage_min,
        hull_iou,
    )


def generate_lightglue_visualization(
    left_rgb: np.ndarray,
    right_rgb: np.ndarray,
    device,
    extractor,
    matcher,
    viz_path: str,
    left_keypoint_mask: Optional[np.ndarray] = None,
    right_keypoint_mask: Optional[np.ndarray] = None,
):
    """Save a side-by-side match visualization to viz_path.

    Green circles = valid matched keypoints, red = unmatched or masked-out
    keypoints, cyan lines = matches, magenta polygons = robust matched-keypoint
    convex hulls used by the coverage metric.
    Images are resized to the same height for side-by-side display but are
    otherwise unmodified by this helper.
    """
    torch, rbd, load_image = _lightglue_runtime()
    img1_pil = Image.fromarray(left_rgb)
    img2_pil = Image.fromarray(right_rgb)
    img1_pil, img2_pil, _ = resize_to_smallest(img1_pil, img2_pil)
    left_np = np.array(img1_pil)
    right_np = np.array(img2_pil)
    left_gray = np.array(img1_pil.convert("L"))
    right_gray = np.array(img2_pil.convert("L"))

    img1_tensor = load_image(left_gray, device)
    img2_tensor = load_image(right_gray, device)

    with torch.no_grad():
        feats1 = extractor.extract(img1_tensor)
        feats2 = extractor.extract(img2_tensor)
        matches_dict = matcher({"image0": feats1, "image1": feats2})
        feats1, feats2, matches_dict = [rbd(x) for x in (feats1, feats2, matches_dict)]
        matches = matches_dict["matches"]

        kp1 = feats1["keypoints"].detach().cpu().numpy()
        kp2 = feats2["keypoints"].detach().cpu().numpy()
        matches_np = matches.detach().cpu().numpy()

    kp1_allowed = _keypoints_allowed(kp1, left_keypoint_mask)
    kp2_allowed = _keypoints_allowed(kp2, right_keypoint_mask)
    matched_left: set = set()
    matched_right: set = set()
    valid_pairs = []
    if matches_np.ndim == 2:
        for row in matches_np:
            i, j = int(row[0]), int(row[1])
            if (
                i != -1
                and j != -1
                and i < len(kp1)
                and j < len(kp2)
                and kp1_allowed[i]
                and kp2_allowed[j]
            ):
                valid_pairs.append((i, j))
                matched_left.add(i)
                matched_right.add(j)

    vis = np.hstack([left_np.copy(), right_np.copy()])
    w_left = left_np.shape[1]

    for i, j in valid_pairs:
        pt1 = tuple(map(int, kp1[i]))
        pt2 = (int(kp2[j][0]) + w_left, int(kp2[j][1]))
        cv2.line(vis, pt1, pt2, (0, 255, 255), 1)

    if valid_pairs:
        left_points = np.float32([kp1[i] for i, _ in valid_pairs])
        right_points = np.float32([kp2[j] for _, j in valid_pairs])
        left_hull = _robust_keypoint_hull_points(left_points, left_np.shape[:2])
        right_hull = _robust_keypoint_hull_points(right_points, right_np.shape[:2])
        if left_hull is not None:
            cv2.polylines(vis, [left_hull], isClosed=True, color=(255, 0, 255), thickness=3)
        if right_hull is not None:
            shifted_right_hull = right_hull.copy()
            shifted_right_hull[:, 0] += w_left
            cv2.polylines(
                vis,
                [shifted_right_hull],
                isClosed=True,
                color=(255, 0, 255),
                thickness=3,
            )

    for idx, kp in enumerate(kp1):
        color = (0, 200, 0) if idx in matched_left else (200, 0, 0)
        cv2.circle(vis, tuple(map(int, kp)), 4, color, -1)

    for idx, kp in enumerate(kp2):
        pt = (int(kp[0]) + w_left, int(kp[1]))
        color = (0, 200, 0) if idx in matched_right else (200, 0, 0)
        cv2.circle(vis, pt, 4, color, -1)

    vis_bgr = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
    cv2.imwrite(viz_path, vis_bgr)


# ---------- Main per-pair API ----------


def process_one_pair(
    *,
    id_left,
    date_left,
    id_right,
    date_right,
    left_path,
    right_path,
    writer,
    csvfile,
    device,
    extractor,
    matcher,
    panorama: bool = False,
    scale_search: bool = False,
    cache: bool = False,
    yaw_step: int = 10,
    crop_keep: float = 0.55,
    crop_top_bias: float = 0.0,
    scale_reproject: bool = False,
    include_image_metrics: bool = True,
    panorama_perspective_preview: bool = False,
    perspective_output_dir: Optional[str] = None,
    perspective_yaws: Optional[List[float]] = None,
    perspective_pitch: float = 0.0,
    perspective_fov: float = 90.0,
    perspective_size: int = 768,
    perspective_align_horizon: bool = False,
    perspective_keypoint_align: bool = False,
    panorama_semantic_rerank: bool = False,
    semantic_rerank_radius: int = 8,
    semantic_rerank_step: int = 2,
    ignore_sky_keypoints: bool = False,
    sky_keypoint_source: str = "segmentation",
    sky_keypoint_boundary_px: int = 10,
    panorama_fast_yaw: bool = False,
    extra_tail: Optional[List] = None,
    segmenter=None,
    seg_output_root: Optional[str] = None,
    seg_crop_top_frac: float = 0.0,
    seg_crop_bottom_frac: float = 0.0,
    profile_timing: bool = False,
    save_debug_images: bool = True,
    segmentation_cache_dir: Optional[str] = None,
    segmentation_max_width: Optional[int] = None,
):
    """
    Process a single pair and write one row to the CSV.

    Common behaviour for both panorama and non-panorama:
        - Compute LightGlue metrics on the "best" pair (possibly yaw-adjusted).
        - Compute quality metrics for both images.
        - Compute horizon angles and their difference.
        - Write one base row to CSV.

    If panorama=True:
        - Compute original metrics (unrotated panoramas).
        - Find best yaw for the RIGHT pano using coarse+fine search.
        - Rotate right pano by best_yaw.
        - Compute "best" metrics on yaw-aligned pair.
        - Append original metrics + best_yaw_deg to the row.
    """
    try:
        from .panorama import (
            estimate_best_yaw,
            estimate_yaw_from_keypoints_once,
            shift_equirectangular,
            crop_equirectangular_middle,
            fov_crop_align,
            rerank_yaw_by_semantic_masks,
        )

        timer = PairTimer(profile_timing)
        left_img = Image.open(left_path).convert("RGB")
        right_img = Image.open(right_path).convert("RGB")
        left_rgb = np.array(left_img)
        right_rgb = np.array(right_img)
        timer.mark("load_images")

        best_yaw = 0
        best_left_rgb = left_rgb
        best_right_rgb = right_rgb
        fov_crop_fraction = 1.0
        fov_left_retained_fraction = 1.0
        fov_right_retained_fraction = 1.0

        orig_match_ratio = None
        orig_avg_dist = None
        orig_inl = None
        orig_total = None
        orig_ratio = None
        orig_coverage_left = 0.0
        orig_coverage_right = 0.0
        orig_coverage_min = 0.0
        orig_hull_iou = 0.0
        orig_seg_mean_iou = "NA"
        orig_seg_road_iou = "NA"
        orig_seg_per_class_json = "{}"
        orig_seg_props_left_before_json = "{}"
        orig_seg_props_right_before_json = "{}"
        orig_seg_props_left_temp_masked_json = "{}"
        orig_seg_props_right_temp_masked_json = "{}"
        orig_seg_temp_union_fraction = "NA"
        scale_reproject_seg_mean_iou = "NA"
        scale_reproject_seg_road_iou = "NA"
        scale_reproject_seg_per_class_json = "{}"
        scale_reproject_seg_props_left_before_json = "{}"
        scale_reproject_seg_props_right_before_json = "{}"
        scale_reproject_seg_props_left_temp_masked_json = "{}"
        scale_reproject_seg_props_right_temp_masked_json = "{}"
        scale_reproject_seg_temp_union_fraction = "NA"
        orig_left_seg = None
        orig_right_seg = None
        scale_left_seg = None
        scale_right_seg = None
        scale_valid_mask = None
        full_left_seg = None
        full_right_seg = None
        crop_left_seg = None
        crop_right_seg = None
        seg_cache_suffix = (
            f"w{int(segmentation_max_width)}"
            if segmentation_max_width
            else "full"
        )
        left_seg_cache_path = (
            os.path.join(segmentation_cache_dir, f"{id_left}_{seg_cache_suffix}.npy")
            if segmentation_cache_dir
            else None
        )
        right_seg_cache_path = (
            os.path.join(segmentation_cache_dir, f"{id_right}_{seg_cache_suffix}.npy")
            if segmentation_cache_dir
            else None
        )
        use_heuristic_sky_keypoints = (
            ignore_sky_keypoints and sky_keypoint_source == "heuristic"
        )
        sky_class_ids = (
            getattr(segmenter, "sky_class_ids", [])
            if (
                segmenter is not None
                and ignore_sky_keypoints
                and sky_keypoint_source == "segmentation"
            )
            else []
        )
        coverage_sky_class_ids = (
            getattr(segmenter, "sky_class_ids", [])
            if segmenter is not None
            else []
        )
        temporary_keypoint_class_ids = (
            getattr(segmenter, "temporary_class_ids", [])
            if (
                segmenter is not None
                and ignore_sky_keypoints
                and sky_keypoint_source == "segmentation"
            )
            else []
        )
        keypoint_ignore_class_ids = sorted(
            set(sky_class_ids) | set(temporary_keypoint_class_ids)
        )

        if panorama and panorama_perspective_preview:
            preview_root = (
                perspective_output_dir
                if perspective_output_dir
                else os.path.join(os.path.dirname(csvfile.name), "panorama_perspective_views")
            )
            preview_dir = _pair_output_dir(preview_root, id_left, id_right)
            ensure_dir(preview_dir)
            yaws = perspective_yaws or [0.0, 60.0, 120.0, 180.0, 240.0, 300.0]
            out_size = (int(perspective_size), int(perspective_size))
            _export_pair_perspective_debug(
                left_rgb=left_rgb,
                right_rgb=right_rgb,
                preview_dir=preview_dir,
                yaws=yaws,
                pitch_deg=perspective_pitch,
                fov_deg=perspective_fov,
                out_size=out_size,
                align_horizon=perspective_align_horizon,
                keypoint_align=perspective_keypoint_align,
                device=device,
                extractor=extractor,
                matcher=matcher,
            )
            logger.info(
                "Saved panorama perspective previews for %s vs %s to %s",
                id_left,
                id_right,
                preview_dir,
            )
            timer.mark("panorama_perspective_preview")

        # ------------------------------------------------------------------
        # 1) PANORAMA: compute original metrics, then find best yaw
        # ------------------------------------------------------------------
        if panorama:
            left_crop_rgb = crop_equirectangular_middle(
                left_rgb, keep_ratio=crop_keep, top_bias=crop_top_bias
            )
            right_crop_rgb = crop_equirectangular_middle(
                right_rgb, keep_ratio=crop_keep, top_bias=crop_top_bias
            )
            orig_L_pil, orig_R_pil, _ = resize_to_smallest(
                Image.fromarray(left_crop_rgb),
                Image.fromarray(right_crop_rgb),
            )
            orig_left_np = np.array(orig_L_pil)
            orig_right_np = np.array(orig_R_pil)
            orig_left_gray = np.array(orig_L_pil.convert("L"))
            orig_right_gray = np.array(orig_R_pil.convert("L"))
            orig_left_keypoint_mask = None
            orig_right_keypoint_mask = None
            panorama_left_keypoint_mask = None
            panorama_right_keypoint_mask = None

            if keypoint_ignore_class_ids:
                full_left_seg = _segment_image_cached(
                    segmenter,
                    left_img,
                    left_seg_cache_path,
                    max_width=segmentation_max_width,
                )
                full_right_seg = _segment_image_cached(
                    segmenter,
                    right_img,
                    right_seg_cache_path,
                    max_width=segmentation_max_width,
                )
                crop_left_seg = _resize_seg_mask(
                    crop_equirectangular_middle(
                        full_left_seg,
                        keep_ratio=crop_keep,
                        top_bias=crop_top_bias,
                    ),
                    left_crop_rgb.shape[:2],
                )
                crop_right_seg = _resize_seg_mask(
                    crop_equirectangular_middle(
                        full_right_seg,
                        keep_ratio=crop_keep,
                        top_bias=crop_top_bias,
                    ),
                    right_crop_rgb.shape[:2],
                )
                orig_left_seg = _resize_seg_mask(crop_left_seg, orig_left_gray.shape)
                orig_right_seg = _resize_seg_mask(crop_right_seg, orig_right_gray.shape)
                orig_left_keypoint_mask = _build_keypoint_valid_mask(
                    orig_left_seg,
                    keypoint_ignore_class_ids,
                    sky_boundary_px=sky_keypoint_boundary_px,
                )
                orig_right_keypoint_mask = _build_keypoint_valid_mask(
                    orig_right_seg,
                    keypoint_ignore_class_ids,
                    sky_boundary_px=sky_keypoint_boundary_px,
                )
                panorama_left_keypoint_mask = _build_keypoint_valid_mask(
                    crop_left_seg,
                    keypoint_ignore_class_ids,
                    sky_boundary_px=sky_keypoint_boundary_px,
                )
                panorama_right_keypoint_mask = _build_keypoint_valid_mask(
                    crop_right_seg,
                    keypoint_ignore_class_ids,
                    sky_boundary_px=sky_keypoint_boundary_px,
                )
                timer.mark("semantic_keypoint_masks")
            elif use_heuristic_sky_keypoints:
                orig_left_keypoint_mask = _build_keypoint_valid_mask_from_sky(
                    _resize_seg_mask(
                        _estimate_sky_mask_heuristic(left_crop_rgb).astype(np.uint8),
                        orig_left_gray.shape,
                    ).astype(bool),
                    sky_keypoint_boundary_px,
                )
                orig_right_keypoint_mask = _build_keypoint_valid_mask_from_sky(
                    _resize_seg_mask(
                        _estimate_sky_mask_heuristic(right_crop_rgb).astype(np.uint8),
                        orig_right_gray.shape,
                    ).astype(bool),
                    sky_keypoint_boundary_px,
                )
                panorama_left_keypoint_mask = _build_keypoint_valid_mask_from_sky(
                    _estimate_sky_mask_heuristic(left_crop_rgb),
                    sky_keypoint_boundary_px,
                )
                panorama_right_keypoint_mask = _build_keypoint_valid_mask_from_sky(
                    _estimate_sky_mask_heuristic(right_crop_rgb),
                    sky_keypoint_boundary_px,
                )
                timer.mark("heuristic_sky_keypoint_masks")

            (
                orig_match_ratio,
                orig_avg_dist,
                orig_inl,
                orig_total,
                orig_ratio,
                orig_coverage_left,
                orig_coverage_right,
                orig_coverage_min,
                orig_hull_iou,
            ) = compute_lightglue_score(
                orig_L_pil,
                orig_R_pil,
                orig_left_np,
                orig_right_np,
                orig_left_gray,
                orig_right_gray,
                device,
                extractor,
                matcher,
                left_keypoint_mask=orig_left_keypoint_mask,
                right_keypoint_mask=orig_right_keypoint_mask,
                left_coverage_mask=_non_sky_coverage_mask(
                    orig_left_np,
                    orig_left_seg,
                    coverage_sky_class_ids,
                ),
                right_coverage_mask=_non_sky_coverage_mask(
                    orig_right_np,
                    orig_right_seg,
                    coverage_sky_class_ids,
                ),
            )
            timer.mark("panorama_original_lightglue")

            if panorama_fast_yaw:
                best_yaw, panorama_alignment_method = estimate_yaw_from_keypoints_once(
                    left_rgb,
                    right_rgb,
                    device,
                    extractor,
                    matcher,
                    yaw_step=yaw_step,
                    crop_keep=crop_keep,
                    crop_top_bias=crop_top_bias,
                    left_keypoint_mask=panorama_left_keypoint_mask,
                    right_keypoint_mask=panorama_right_keypoint_mask,
                )
            else:
                yaw_debug_path = None
                if save_debug_images:
                    yaw_debug_path = os.path.join(
                        _pair_output_dir(
                            os.path.join(os.path.dirname(csvfile.name), "pano_debug"),
                            id_left,
                            id_right,
                        ),
                        "yaw_scores.json",
                    )
                best_yaw, panorama_alignment_method = estimate_best_yaw(
                    left_rgb,
                    right_rgb,
                    device,
                    extractor,
                    matcher,
                    yaw_step=yaw_step,
                    crop_keep=crop_keep,
                    crop_top_bias=crop_top_bias,
                    original_yaw_deg=0,
                    debug_scores_path=yaw_debug_path,
                    left_keypoint_mask=panorama_left_keypoint_mask,
                    right_keypoint_mask=panorama_right_keypoint_mask,
                )
            timer.mark("panorama_yaw_search")

            if panorama_semantic_rerank and segmenter is not None:
                if crop_left_seg is None or crop_right_seg is None:
                    if full_left_seg is None:
                        full_left_seg = _segment_image_cached(
                            segmenter,
                            left_img,
                            left_seg_cache_path,
                            max_width=segmentation_max_width,
                        )
                    if full_right_seg is None:
                        full_right_seg = _segment_image_cached(
                            segmenter,
                            right_img,
                            right_seg_cache_path,
                            max_width=segmentation_max_width,
                        )
                    crop_left_seg = _resize_seg_mask(
                        crop_equirectangular_middle(
                            full_left_seg,
                            keep_ratio=crop_keep,
                            top_bias=crop_top_bias,
                        ),
                        crop_equirectangular_middle(
                            left_rgb,
                            keep_ratio=crop_keep,
                            top_bias=crop_top_bias,
                        ).shape[:2],
                    )
                    crop_right_seg = _resize_seg_mask(
                        crop_equirectangular_middle(
                            full_right_seg,
                            keep_ratio=crop_keep,
                            top_bias=crop_top_bias,
                        ),
                        crop_equirectangular_middle(
                            right_rgb,
                            keep_ratio=crop_keep,
                            top_bias=crop_top_bias,
                        ).shape[:2],
                    )
                ignore_ids = (
                    getattr(segmenter, "temporary_class_ids", [])
                    if getattr(segmenter, "ignore_temporary", False)
                    else []
                )
                semantic_yaw, semantic_score, semantic_scores = rerank_yaw_by_semantic_masks(
                    crop_left_seg,
                    crop_right_seg,
                    best_yaw,
                    search_radius=semantic_rerank_radius,
                    step=semantic_rerank_step,
                    ignore_class_ids=ignore_ids,
                )
                semantic_debug_dir = _pair_output_dir(
                    os.path.join(os.path.dirname(csvfile.name), "pano_debug"),
                    id_left,
                    id_right,
                )
                ensure_dir(semantic_debug_dir)
                with open(
                    os.path.join(semantic_debug_dir, "semantic_yaw_scores.json"),
                    "w",
                    encoding="utf-8",
                ) as f:
                    json.dump(
                        {
                            "keypoint_yaw": int(best_yaw),
                            "semantic_yaw": int(semantic_yaw),
                            "semantic_score": float(semantic_score),
                            "scores": semantic_scores,
                        },
                        f,
                        indent=2,
                    )
                best_yaw = semantic_yaw
                panorama_alignment_method = f"{panorama_alignment_method}+semantic_rerank"
                timer.mark("panorama_semantic_rerank")

            best_right_rgb = shift_equirectangular(right_rgb, best_yaw)
            timer.mark("panorama_shift")

            if save_debug_images:
                debug_root = os.path.join(os.path.dirname(csvfile.name), "pano_debug")
                pair_dir = _pair_output_dir(debug_root, id_left, id_right)
                ensure_dir(pair_dir)

                Image.fromarray(left_rgb).save(os.path.join(pair_dir, "left.png"))

                best_pano_path = os.path.join(pair_dir, f"right_best_yaw_{best_yaw}.png")
                Image.fromarray(best_right_rgb).save(best_pano_path)

                viz_path = os.path.join(pair_dir, f"matches_best_yaw_{best_yaw}.png")
                generate_lightglue_visualization(
                    left_rgb, best_right_rgb, device, extractor, matcher, viz_path
                )
                timer.mark("panorama_debug_outputs")

            best_left_rgb = crop_equirectangular_middle(
                left_rgb, keep_ratio=crop_keep, top_bias=crop_top_bias
            )
            best_right_rgb = crop_equirectangular_middle(
                best_right_rgb, keep_ratio=crop_keep, top_bias=crop_top_bias
            )
            timer.mark("panorama_analysis_crop")

        # ------------------------------------------------------------------
        # 1b) SCALE SEARCH: compute original metrics, then find best scale
        # ------------------------------------------------------------------
        elif scale_search:
            orig_L_pil, orig_R_pil, _ = resize_to_smallest(left_img, right_img)
            orig_left_np = np.array(orig_L_pil)
            orig_right_np = np.array(orig_R_pil)
            orig_left_gray = np.array(orig_L_pil.convert("L"))
            orig_right_gray = np.array(orig_R_pil.convert("L"))
            (
                orig_match_ratio,
                orig_avg_dist,
                orig_inl,
                orig_total,
                orig_ratio,
                orig_coverage_left,
                orig_coverage_right,
                orig_coverage_min,
                orig_hull_iou,
            ) = compute_lightglue_score(
                orig_L_pil, orig_R_pil,
                orig_left_np, orig_right_np,
                orig_left_gray, orig_right_gray,
                device, extractor, matcher,
                left_coverage_mask=_non_sky_coverage_mask(
                    orig_left_np,
                    None,
                    coverage_sky_class_ids,
                ),
                right_coverage_mask=_non_sky_coverage_mask(
                    orig_right_np,
                    None,
                    coverage_sky_class_ids,
                ),
            )
            timer.mark("scale_original_lightglue")

            if include_image_metrics:
                (
                    orig_b_left,
                    orig_mb_left,
                    orig_df_left,
                    orig_c_left,
                    orig_s_left,
                    orig_n_left,
                ) = compute_image_quality_metrics(orig_left_gray)
                (
                    orig_b_right,
                    orig_mb_right,
                    orig_df_right,
                    orig_c_right,
                    orig_s_right,
                    orig_n_right,
                ) = compute_image_quality_metrics(orig_right_gray)

                orig_ha_left = estimate_horizon_angle(orig_left_gray)
                orig_ha_right = estimate_horizon_angle(orig_right_gray)
                orig_ha_diff = (
                    abs(orig_ha_left - orig_ha_right)
                    if orig_ha_left is not None and orig_ha_right is not None
                    else "NA"
                )
                timer.mark("scale_original_image_metrics")

            if segmenter is not None:
                full_left_seg = _resize_seg_mask(
                    _segment_image_cached(
                        segmenter,
                        left_img,
                        left_seg_cache_path,
                        max_width=segmentation_max_width,
                    ),
                    left_rgb.shape[:2],
                )
                full_right_seg = _resize_seg_mask(
                    _segment_image_cached(
                        segmenter,
                        right_img,
                        right_seg_cache_path,
                        max_width=segmentation_max_width,
                    ),
                    right_rgb.shape[:2],
                )
                orig_left_mask = _resize_seg_mask(full_left_seg, orig_left_gray.shape)
                orig_right_mask = _resize_seg_mask(full_right_seg, orig_right_gray.shape)
                (
                    orig_seg_mean_iou,
                    orig_seg_road_iou,
                    orig_seg_per_class_json,
                    orig_seg_props_left_before_json,
                    orig_seg_props_right_before_json,
                    orig_seg_props_left_temp_masked_json,
                    orig_seg_props_right_temp_masked_json,
                    orig_seg_temp_union_fraction,
                ) = _summarize_seg_pair(
                    segmenter,
                    orig_left_mask,
                    orig_right_mask,
                    seg_crop_top_frac=seg_crop_top_frac,
                    seg_crop_bottom_frac=seg_crop_bottom_frac,
                )
                timer.mark("scale_original_segmentation")

            pair_dir = None
            if save_debug_images:
                scale_debug_root = os.path.join(os.path.dirname(csvfile.name), "scale_debug")
                pair_dir = _pair_output_dir(scale_debug_root, id_left, id_right)
                ensure_dir(pair_dir)

            (
                best_left_rgb,
                best_right_rgb,
                fov_crop_fraction,
                fov_left_retained_fraction,
                fov_right_retained_fraction,
                scale_alignment_metadata,
            ) = fov_crop_align(
                left_rgb,
                right_rgb,
                device,
                extractor,
                matcher,
                reproject=scale_reproject,
                reproject_debug_dir=pair_dir if scale_reproject and save_debug_images else None,
                left_aux=full_left_seg,
                right_aux=full_right_seg,
                return_metadata=True,
            )
            if segmenter is not None:
                scale_left_seg = scale_alignment_metadata.get("left_aux")
                scale_right_seg = scale_alignment_metadata.get("right_aux")
                scale_valid_from_metadata = scale_alignment_metadata.get("valid_mask")
                scale_valid_mask = (
                    scale_valid_from_metadata
                    if isinstance(scale_valid_from_metadata, np.ndarray)
                    else None
                )
                if scale_left_seg is not None and scale_right_seg is not None:
                    scale_left_seg = _resize_seg_mask(scale_left_seg, best_left_rgb.shape[:2])
                    scale_right_seg = _resize_seg_mask(scale_right_seg, best_right_rgb.shape[:2])
                    if scale_valid_mask is not None:
                        scale_valid_mask = _resize_seg_mask(
                            scale_valid_mask.astype(np.uint8),
                            best_left_rgb.shape[:2],
                        ).astype(bool)

                reproject_left_seg = scale_alignment_metadata.get("reprojected_left_aux")
                reproject_right_seg = scale_alignment_metadata.get("reprojected_right_aux")
                if (
                    scale_reproject
                    and reproject_left_seg is not None
                    and reproject_right_seg is not None
                ):
                    reproject_valid_mask = scale_alignment_metadata.get(
                        "reprojected_valid_mask"
                    )
                    (
                        scale_reproject_seg_mean_iou,
                        scale_reproject_seg_road_iou,
                        scale_reproject_seg_per_class_json,
                        scale_reproject_seg_props_left_before_json,
                        scale_reproject_seg_props_right_before_json,
                        scale_reproject_seg_props_left_temp_masked_json,
                        scale_reproject_seg_props_right_temp_masked_json,
                        scale_reproject_seg_temp_union_fraction,
                    ) = _summarize_seg_pair(
                        segmenter,
                        reproject_left_seg,
                        reproject_right_seg,
                        seg_crop_top_frac=seg_crop_top_frac,
                        seg_crop_bottom_frac=seg_crop_bottom_frac,
                        valid_pixels=reproject_valid_mask
                        if isinstance(reproject_valid_mask, np.ndarray)
                        else None,
                    )
            timer.mark("scale_fov_alignment")

            if save_debug_images and scale_reproject:
                crop_left_rgb, crop_right_rgb, _, _, _ = fov_crop_align(
                    left_rgb,
                    right_rgb,
                    device,
                    extractor,
                    matcher,
                    reproject=False,
                )
                Image.fromarray(crop_left_rgb).save(
                    os.path.join(pair_dir, "left_cropped.png")
                )
                Image.fromarray(crop_right_rgb).save(
                    os.path.join(pair_dir, "right_cropped.png")
                )
                Image.fromarray(best_left_rgb).save(
                    os.path.join(pair_dir, "left_reprojected_cropped.png")
                )
                Image.fromarray(best_right_rgb).save(
                    os.path.join(pair_dir, "right_reprojected_cropped.png")
                )
                timer.mark("scale_debug_outputs")
            elif save_debug_images:
                Image.fromarray(best_left_rgb).save(os.path.join(pair_dir, "left_cropped.png"))
                Image.fromarray(best_right_rgb).save(os.path.join(pair_dir, "right_cropped.png"))
                timer.mark("scale_debug_outputs")

        # ------------------------------------------------------------------
        # 2) COMMON ANALYSIS: same analysis for panorama, scale-search, and plain
        # ------------------------------------------------------------------
        L_pil, R_pil, _ = resize_to_smallest(
            Image.fromarray(best_left_rgb), Image.fromarray(best_right_rgb)
        )
        left_np = np.array(L_pil)
        right_np = np.array(R_pil)
        left_gray = np.array(L_pil.convert("L"))
        right_gray = np.array(R_pil.convert("L"))
        left_keypoint_mask = None
        right_keypoint_mask = None
        left_mask = None
        right_mask = None
        if segmenter is not None:
            if panorama:
                if full_left_seg is None:
                    full_left_seg = _segment_image_cached(
                        segmenter,
                        left_img,
                        left_seg_cache_path,
                        max_width=segmentation_max_width,
                    )
                if full_right_seg is None:
                    full_right_seg = _segment_image_cached(
                        segmenter,
                        right_img,
                        right_seg_cache_path,
                        max_width=segmentation_max_width,
                    )
                left_mask = crop_equirectangular_middle(
                    full_left_seg,
                    keep_ratio=crop_keep,
                    top_bias=crop_top_bias,
                )
                right_mask = crop_equirectangular_middle(
                    shift_equirectangular(full_right_seg, best_yaw),
                    keep_ratio=crop_keep,
                    top_bias=crop_top_bias,
                )
                left_mask = _resize_seg_mask(left_mask, left_gray.shape)
                right_mask = _resize_seg_mask(right_mask, right_gray.shape)
            elif scale_search and scale_left_seg is not None and scale_right_seg is not None:
                left_mask = _resize_seg_mask(scale_left_seg, left_gray.shape)
                right_mask = _resize_seg_mask(scale_right_seg, right_gray.shape)
            else:
                left_mask = _resize_seg_mask(
                    _segment_image_cached(
                        segmenter,
                        L_pil,
                        left_seg_cache_path,
                        max_width=segmentation_max_width,
                    ),
                    left_gray.shape,
                )
                right_mask = _resize_seg_mask(
                    _segment_image_cached(
                        segmenter,
                        R_pil,
                        right_seg_cache_path,
                        max_width=segmentation_max_width,
                    ),
                    right_gray.shape,
                )
            timer.mark("aligned_segmentation_inference")

        if keypoint_ignore_class_ids:
            left_keypoint_mask = _build_keypoint_valid_mask(
                left_mask,
                keypoint_ignore_class_ids,
                sky_boundary_px=sky_keypoint_boundary_px,
            )
            right_keypoint_mask = _build_keypoint_valid_mask(
                right_mask,
                keypoint_ignore_class_ids,
                sky_boundary_px=sky_keypoint_boundary_px,
            )
            timer.mark("aligned_semantic_keypoint_masks")
        elif use_heuristic_sky_keypoints:
            left_keypoint_mask = _build_keypoint_valid_mask_from_sky(
                _estimate_sky_mask_heuristic(left_np),
                sky_keypoint_boundary_px,
            )
            right_keypoint_mask = _build_keypoint_valid_mask_from_sky(
                _estimate_sky_mask_heuristic(right_np),
                sky_keypoint_boundary_px,
            )
            timer.mark("aligned_heuristic_sky_keypoint_masks")

        (
            lg_match_ratio,
            lg_avg_dist,
            lg_inl,
            lg_total,
            lg_ratio,
            lg_coverage_left,
            lg_coverage_right,
            lg_coverage_min,
            lg_hull_iou,
        ) = compute_lightglue_score(
            L_pil,
            R_pil,
            left_np,
            right_np,
            left_gray,
            right_gray,
            device,
            extractor,
            matcher,
            left_keypoint_mask=left_keypoint_mask,
            right_keypoint_mask=right_keypoint_mask,
            left_coverage_mask=_non_sky_coverage_mask(
                left_np,
                left_mask,
                coverage_sky_class_ids,
                valid_pixels=scale_valid_mask if scale_search else None,
            ),
            right_coverage_mask=_non_sky_coverage_mask(
                right_np,
                right_mask,
                coverage_sky_class_ids,
                valid_pixels=scale_valid_mask if scale_search else None,
            ),
        )
        timer.mark("aligned_lightglue")

        if scale_search:
            lg_coverage_left = orig_coverage_left
            lg_coverage_right = orig_coverage_right
            lg_coverage_min = orig_coverage_min
            lg_hull_iou = orig_hull_iou

        # -------------------------------------------------------------
        # 2b) Semantic segmentation & overlap
        # -------------------------------------------------------------
        seg_mean_iou = "NA"
        seg_road_iou = "NA"
        seg_per_class_json = "{}"
        seg_props_left_before_json = "{}"
        seg_props_right_before_json = "{}"
        seg_props_left_temp_masked_json = "{}"
        seg_props_right_temp_masked_json = "{}"
        seg_temp_union_fraction = "NA"

        if segmenter is not None:
            if seg_output_root is not None:
                pair_dir = _pair_output_dir(seg_output_root, id_left, id_right)
                ensure_dir(pair_dir)

                Image.fromarray(left_mask).save(
                    os.path.join(pair_dir, "left_seg_labels.png")
                )
                Image.fromarray(right_mask).save(
                    os.path.join(pair_dir, "right_seg_labels.png")
                )

                left_color = segmenter.colorize_mask(left_mask)
                right_color = segmenter.colorize_mask(right_mask)
                Image.fromarray(left_color).save(
                    os.path.join(pair_dir, "left_seg_color.png")
                )
                Image.fromarray(right_color).save(
                    os.path.join(pair_dir, "right_seg_color.png")
                )

            (
                seg_mean_iou,
                seg_road_iou,
                seg_per_class_json,
                seg_props_left_before_json,
                seg_props_right_before_json,
                seg_props_left_temp_masked_json,
                seg_props_right_temp_masked_json,
                seg_temp_union_fraction,
            ) = _summarize_seg_pair(
                segmenter,
                left_mask,
                right_mask,
                seg_crop_top_frac=seg_crop_top_frac,
                seg_crop_bottom_frac=seg_crop_bottom_frac,
                valid_pixels=scale_valid_mask if scale_search else None,
            )
            timer.mark("aligned_segmentation")

        if not panorama and not scale_search:
            orig_match_ratio = lg_match_ratio
            orig_avg_dist = lg_avg_dist
            orig_inl = lg_inl
            orig_total = lg_total
            orig_ratio = lg_ratio
            orig_coverage_left = lg_coverage_left
            orig_coverage_right = lg_coverage_right
            orig_coverage_min = lg_coverage_min
            orig_hull_iou = lg_hull_iou

        image_metric_values = []
        if include_image_metrics:
            (
                b_left,
                mb_left,
                df_left,
                c_left,
                s_left,
                n_left,
            ) = compute_image_quality_metrics(left_gray)
            (
                b_right,
                mb_right,
                df_right,
                c_right,
                s_right,
                n_right,
            ) = compute_image_quality_metrics(right_gray)

            ha_left = estimate_horizon_angle(left_gray)
            ha_right = estimate_horizon_angle(right_gray)
            image_metric_values = [
                round(b_left, 2),
                round(mb_left, 2),
                round(df_left, 4),
                round(c_left, 2),
                round(s_left, 2),
                round(n_left, 2),
                round(b_right, 2),
                round(mb_right, 2),
                round(df_right, 4),
                round(c_right, 2),
                round(s_right, 2),
                round(n_right, 2),
                round(ha_left, 2) if ha_left is not None else "NA",
                round(ha_right, 2) if ha_right is not None else "NA",
                round(abs(ha_left - ha_right), 2)
                if ha_left is not None and ha_right is not None
                else "NA",
            ]
            timer.mark("image_metrics")

        # ------------------------------------------------------------------
        # 2b) Keypoint match visualization.
        # ------------------------------------------------------------------
        if save_debug_images:
            viz_root = os.path.join(os.path.dirname(csvfile.name), "match_viz")
            viz_dir = _pair_output_dir(viz_root, id_left, id_right)
            ensure_dir(viz_dir)
            viz_left_rgb = left_rgb if scale_search else best_left_rgb
            viz_right_rgb = right_rgb if scale_search else best_right_rgb
            viz_left_keypoint_mask = None if scale_search else left_keypoint_mask
            viz_right_keypoint_mask = None if scale_search else right_keypoint_mask
            generate_lightglue_visualization(
                viz_left_rgb,
                viz_right_rgb,
                device,
                extractor,
                matcher,
                os.path.join(viz_dir, "matches.png"),
                left_keypoint_mask=viz_left_keypoint_mask,
                right_keypoint_mask=viz_right_keypoint_mask,
            )
            timer.mark("match_visualization")

        # ------------------------------------------------------------------
        # 3) Build CSV row
        # ------------------------------------------------------------------
        row_out = [
            id_left,
            date_left,
            id_right,
            date_right,
            round(lg_match_ratio, 4),
            round(lg_avg_dist, 4) if np.isfinite(lg_avg_dist) else "inf",
            round(lg_coverage_left, 4),
            round(lg_coverage_right, 4),
            round(lg_coverage_min, 4),
            round(lg_hull_iou, 4),
            *image_metric_values,
            int(round(lg_inl)),
            int(round(lg_total)),
            round(lg_ratio, 4) if lg_total > 0 else 0.0,
            seg_mean_iou,
            seg_road_iou,
            seg_per_class_json,
            seg_props_left_before_json,
            seg_props_right_before_json,
            seg_props_left_temp_masked_json,
            seg_props_right_temp_masked_json,
            _round_seg_fraction(seg_temp_union_fraction),
        ]

        if panorama:
            row_out.extend(
                [
                    round(orig_match_ratio, 4),
                    round(orig_avg_dist, 4) if np.isfinite(orig_avg_dist) else "inf",
                    round(orig_coverage_left, 4),
                    round(orig_coverage_right, 4),
                    round(orig_coverage_min, 4),
                    round(orig_hull_iou, 4),
                    int(round(orig_inl)),
                    int(round(orig_total)),
                    round(orig_ratio, 4) if orig_total > 0 else 0.0,
                    int(best_yaw),
                    panorama_alignment_method,
                ]
            )
        elif scale_search:
            scale_values = [
                round(orig_match_ratio, 4),
                round(orig_avg_dist, 4) if np.isfinite(orig_avg_dist) else "inf",
                round(orig_coverage_left, 4),
                round(orig_coverage_right, 4),
                round(orig_coverage_min, 4),
                round(orig_hull_iou, 4),
            ]
            if include_image_metrics:
                scale_values.extend(
                    [
                    round(orig_b_left, 2),
                    round(orig_mb_left, 2),
                    round(orig_df_left, 4),
                    round(orig_c_left, 2),
                    round(orig_s_left, 2),
                    round(orig_n_left, 2),
                    round(orig_b_right, 2),
                    round(orig_mb_right, 2),
                    round(orig_df_right, 4),
                    round(orig_c_right, 2),
                    round(orig_s_right, 2),
                    round(orig_n_right, 2),
                    round(orig_ha_left, 2) if orig_ha_left is not None else "NA",
                    round(orig_ha_right, 2) if orig_ha_right is not None else "NA",
                    round(orig_ha_diff, 2) if orig_ha_diff != "NA" else "NA",
                    ]
                )
            scale_values.extend(
                [
                    int(round(orig_inl)),
                    int(round(orig_total)),
                    round(orig_ratio, 4) if orig_total > 0 else 0.0,
                    orig_seg_mean_iou,
                    orig_seg_road_iou,
                    orig_seg_per_class_json,
                    orig_seg_props_left_before_json,
                    orig_seg_props_right_before_json,
                    orig_seg_props_left_temp_masked_json,
                    orig_seg_props_right_temp_masked_json,
                    _round_seg_fraction(orig_seg_temp_union_fraction),
                    scale_reproject_seg_mean_iou,
                    scale_reproject_seg_road_iou,
                    scale_reproject_seg_per_class_json,
                    scale_reproject_seg_props_left_before_json,
                    scale_reproject_seg_props_right_before_json,
                    scale_reproject_seg_props_left_temp_masked_json,
                    scale_reproject_seg_props_right_temp_masked_json,
                    _round_seg_fraction(scale_reproject_seg_temp_union_fraction),
                    round(fov_crop_fraction, 4),
                    round(fov_left_retained_fraction, 4),
                    round(fov_right_retained_fraction, 4),
                    round(1.0 - fov_left_retained_fraction, 4),
                    round(1.0 - fov_right_retained_fraction, 4),
                ]
            )
            row_out.extend(scale_values)

        if extra_tail:
            row_out.extend(extra_tail)

        writer.writerow(row_out)
        timing_rows = timer.finish()
        if timing_rows:
            slowest_stage, slowest_seconds = max(
                (row for row in timing_rows if row[0] != "total"),
                key=lambda row: row[1],
                default=("none", 0.0),
            )
            logger.info(
                "Timing %s vs %s: total=%.3fs, slowest=%s %.3fs",
                id_left,
                id_right,
                dict(timing_rows).get("total", 0.0),
                slowest_stage,
                slowest_seconds,
            )
        return {"left": date_left, "right": date_right, "timing": timing_rows}

    except Exception as e:
        import traceback, sys

        tb = traceback.extract_tb(sys.exc_info()[2])[-1]
        logger.error(
            "Error processing %s vs %s: %s (line %d in %s)",
            id_left, id_right, e, tb.lineno, tb.filename,
        )
        return {"left": date_left, "right": date_right}
