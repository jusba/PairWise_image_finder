import logging
import os
import csv
import pandas as pd
from typing import Iterator, Dict

logger = logging.getLogger(__name__)

SEG_CLASS_PROPORTION_COLUMNS = [
    "seg_class_props_left_before_json",
    "seg_class_props_right_before_json",
    "seg_class_props_left_temp_masked_json",
    "seg_class_props_right_temp_masked_json",
    "seg_temp_union_fraction",
]

ORIG_SEG_CLASS_PROPORTION_COLUMNS = [
    f"orig_{col}" for col in SEG_CLASS_PROPORTION_COLUMNS
]

SCALE_REPROJECT_SEG_CLASS_PROPORTION_COLUMNS = [
    f"scale_reproject_{col}" for col in SEG_CLASS_PROPORTION_COLUMNS
]

def case_insensitive_join(dirpath: str, name: str):
    """Return the full path of a file in dirpath matching name, case-insensitive."""
    if not os.path.isdir(dirpath):
        return None
    target = name.lower()
    for f in os.listdir(dirpath):
        if f.lower() == target:
            return os.path.join(dirpath, f)
    return None


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def ensure_manifest(path: str):
    """Create a manifest CSV with header if missing."""
    if not os.path.exists(path) or os.stat(path).st_size == 0:
        with open(path, "w", newline="") as f:
            csv.writer(f).writerow(["id", "path"])


CSV_HEADER = [
    "id_left", "date_left", "id_right", "date_right",
    "lightglue_match_ratio", "lightglue_avg_distance",
    "lightglue_keypoint_coverage_left",
    "lightglue_keypoint_coverage_right",
    "lightglue_keypoint_coverage_min",
    "lightglue_keypoint_hull_iou",
    "brightness_left", "median_brightness_left", "dark_fraction_left",
    "contrast_left", "sharpness_left", "noise_left",
    "brightness_right", "median_brightness_right", "dark_fraction_right",
    "contrast_right", "sharpness_right", "noise_right",
    "horizon_angle_left", "horizon_angle_right", "horizon_alignment_diff",
    "lightglue_homography_inliers", "lightglue_homography_total",
    "lightglue_homography_ratio",
    "seg_overlap_mean_iou",
    "seg_overlap_road_iou",
    "seg_overlap_per_class_json",
    *SEG_CLASS_PROPORTION_COLUMNS,
    "index",
]

IMAGE_METRIC_COLUMNS = [
    "brightness_left", "median_brightness_left", "dark_fraction_left",
    "contrast_left", "sharpness_left", "noise_left",
    "brightness_right", "median_brightness_right", "dark_fraction_right",
    "contrast_right", "sharpness_right", "noise_right",
    "horizon_angle_left", "horizon_angle_right",
]

_IMAGE_METRIC_COLUMN_SET = set(IMAGE_METRIC_COLUMNS) | {"horizon_alignment_diff"}

_ORIG_IMAGE_METRIC_COLUMN_SET = {
    f"orig_{col}" for col in _IMAGE_METRIC_COLUMN_SET
}


def with_optional_image_metrics(header: list[str], include_image_metrics: bool) -> list[str]:
    """Return a CSV header with image-quality columns included or removed."""
    if include_image_metrics:
        return list(header)
    return [
        col for col in header
        if col not in _IMAGE_METRIC_COLUMN_SET
        and col not in _ORIG_IMAGE_METRIC_COLUMN_SET
    ]

CSV_HEADER_PANORAMA = [
    "id_left", "date_left", "id_right", "date_right",
    # Best-yaw metrics
    "lightglue_match_ratio", "lightglue_avg_distance",
    "lightglue_keypoint_coverage_left",
    "lightglue_keypoint_coverage_right",
    "lightglue_keypoint_coverage_min",
    "lightglue_keypoint_hull_iou",
    "brightness_left", "median_brightness_left", "dark_fraction_left",
    "contrast_left", "sharpness_left", "noise_left",
    "brightness_right", "median_brightness_right", "dark_fraction_right",
    "contrast_right", "sharpness_right", "noise_right",
    "horizon_angle_left", "horizon_angle_right", "horizon_alignment_diff",
    "lightglue_homography_inliers", "lightglue_homography_total",
    "lightglue_homography_ratio",
    # Segmentation (written before panorama extras to match row construction order)
    "seg_overlap_mean_iou",
    "seg_overlap_road_iou",
    "seg_overlap_per_class_json",
    *SEG_CLASS_PROPORTION_COLUMNS,
    # Original (unrotated) panorama metrics
    "orig_lightglue_match_ratio", "orig_lightglue_avg_distance",
    "orig_lightglue_keypoint_coverage_left",
    "orig_lightglue_keypoint_coverage_right",
    "orig_lightglue_keypoint_coverage_min",
    "orig_lightglue_keypoint_hull_iou",
    "orig_lightglue_homography_inliers", "orig_lightglue_homography_total",
    "orig_lightglue_homography_ratio",
    "best_yaw_deg",
    "panorama_alignment_method",
    "index",
]

CSV_HEADER_SCALE = [
    "id_left", "date_left", "id_right", "date_right",
    # Cropped scale-search metrics
    "lightglue_match_ratio", "lightglue_avg_distance",
    "lightglue_keypoint_coverage_left",
    "lightglue_keypoint_coverage_right",
    "lightglue_keypoint_coverage_min",
    "lightglue_keypoint_hull_iou",
    "brightness_left", "median_brightness_left", "dark_fraction_left",
    "contrast_left", "sharpness_left", "noise_left",
    "brightness_right", "median_brightness_right", "dark_fraction_right",
    "contrast_right", "sharpness_right", "noise_right",
    "horizon_angle_left", "horizon_angle_right", "horizon_alignment_diff",
    "lightglue_homography_inliers", "lightglue_homography_total",
    "lightglue_homography_ratio",
    "seg_overlap_mean_iou",
    "seg_overlap_road_iou",
    "seg_overlap_per_class_json",
    *SEG_CLASS_PROPORTION_COLUMNS,
    # Original/full-image metrics
    "orig_lightglue_match_ratio", "orig_lightglue_avg_distance",
    "orig_lightglue_keypoint_coverage_left",
    "orig_lightglue_keypoint_coverage_right",
    "orig_lightglue_keypoint_coverage_min",
    "orig_lightglue_keypoint_hull_iou",
    "orig_brightness_left", "orig_median_brightness_left", "orig_dark_fraction_left",
    "orig_contrast_left", "orig_sharpness_left", "orig_noise_left",
    "orig_brightness_right", "orig_median_brightness_right", "orig_dark_fraction_right",
    "orig_contrast_right", "orig_sharpness_right", "orig_noise_right",
    "orig_horizon_angle_left", "orig_horizon_angle_right", "orig_horizon_alignment_diff",
    "orig_lightglue_homography_inliers", "orig_lightglue_homography_total",
    "orig_lightglue_homography_ratio",
    "orig_seg_overlap_mean_iou",
    "orig_seg_overlap_road_iou",
    "orig_seg_overlap_per_class_json",
    *ORIG_SEG_CLASS_PROPORTION_COLUMNS,
    # Full reprojected metrics before the final scale crop.
    "scale_reproject_seg_overlap_mean_iou",
    "scale_reproject_seg_overlap_road_iou",
    "scale_reproject_seg_overlap_per_class_json",
    *SCALE_REPROJECT_SEG_CLASS_PROPORTION_COLUMNS,
    # Crop amounts
    "fov_crop_fraction",
    "fov_left_retained_fraction",
    "fov_right_retained_fraction",
    "fov_left_cropped_fraction",
    "fov_right_cropped_fraction",
    "index",
]

# Extra columns appended in area mode (after "index")
_AREA_EXTRA = [
    "dist_m",
    "angle_diff_deg",
    "is_pano_left",
    "is_pano_right",
    "lat_left",
    "lon_left",
    "lat_right",
    "lon_right",
]

CSV_HEADER_AREA = CSV_HEADER + _AREA_EXTRA
CSV_HEADER_PANORAMA_AREA = CSV_HEADER_PANORAMA + _AREA_EXTRA
CSV_HEADER_SCALE_AREA = CSV_HEADER_SCALE + _AREA_EXTRA





def load_completed_pairs_from_manifest(csv_path: str) -> set[tuple[str, str]]:
    completed: set[tuple[str, str]] = set()
    if not os.path.exists(csv_path):
        return completed
    try:
        df = pd.read_csv(csv_path, on_bad_lines='skip')
        if {"id_left", "id_right", "left_path", "right_path"}.issubset(df.columns):
            for _, row in df.iterrows():
                lp, rp = row.get("left_path"), row.get("right_path")
                if isinstance(lp, str) and isinstance(rp, str) and os.path.exists(lp) and os.path.exists(rp):
                    completed.add((str(row["id_left"]), str(row["id_right"])))
    except Exception as e:
        logger.warning("manifest read error: %s", e)
    return completed

def completed_pairs_set(output_csv: str) -> set[tuple[str, str]]:
    done: set[tuple[str, str]] = set()
    if not os.path.exists(output_csv) or os.stat(output_csv).st_size == 0:
        return done
    try:
        df = pd.read_csv(output_csv, on_bad_lines='skip')
        if {"id_left", "id_right"}.issubset(df.columns):
            for _, r in df.iterrows():
                done.add((str(r["id_left"]), str(r["id_right"])))
    except Exception as e:
        logger.warning("completed_pairs_set error: %s", e)
    return done

def iter_manifest_rows(manifest_csv: str) -> Iterator[Dict]:
    df = pd.read_csv(manifest_csv, on_bad_lines='skip')
    required = {"id_left", "date_left", "id_right", "date_right", "left_path", "right_path"}
    if not required.issubset(df.columns):
        raise ValueError(f"Manifest missing columns: {required}")
    for _, r in df.iterrows():
        yield r.to_dict()

def iter_folder_rows(input_dir: str) -> Iterator[Dict]:
    for root, _, files in os.walk(input_dir):
        pngs = [f for f in files if f.lower().endswith(".png")]
        if len(pngs) < 2:
            continue
        folder = os.path.basename(root)
        if "_" in folder:
            id_left, id_right = folder.split("_", 1)
            left_name = f"{id_left}.png"
            right_name = f"{id_right}.png"
            left_path = os.path.join(root, left_name) if left_name in pngs else None
            right_path = os.path.join(root, right_name) if right_name in pngs else None
            if left_path and right_path and os.path.exists(left_path) and os.path.exists(right_path):
                yield {
                    "id_left": id_left, "date_left": "N/A",
                    "id_right": id_right, "date_right": "N/A",
                    "left_path": left_path, "right_path": right_path
                }
