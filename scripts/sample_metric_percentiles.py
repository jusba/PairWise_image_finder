#!/usr/bin/env python
"""
Sample panorama pairs from results.csv by metric deciles and save aligned previews.

Designed so it can be run either from this project directory or from the parent
folder that contains Pairwise_image_finder.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests
from PIL import Image, ImageDraw


def _find_project_root() -> Path:
    here = Path(__file__).resolve()
    candidates = [
        Path.cwd(),
        Path.cwd() / "Pairwise_image_finder",
        here.parents[1],
    ]
    for candidate in candidates:
        if (candidate / "utils").is_dir() and (candidate / "scripts").is_dir():
            return candidate.resolve()
    raise RuntimeError("Could not find Pairwise_image_finder project root.")


PROJECT_ROOT = _find_project_root()
sys.path.insert(0, str(PROJECT_ROOT))

from utils.fetcher import fetch_and_cache_image  # noqa: E402
from utils.models import init_models, pick_device  # noqa: E402
from utils.panorama import (  # noqa: E402
    estimate_yaw_from_keypoints_once,
    shift_equirectangular,
)


logger = logging.getLogger(__name__)

DEFAULT_METRICS = [
    "lightglue_avg_distance",
    "lightglue_match_ratio",
    "lightglue_keypoint_coverage_min",
    "lightglue_keypoint_hull_iou",
    "seg_overlap_road_iou",
]


def _to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.replace({"NA": np.nan, "inf": np.inf}), errors="coerce")


def _safe_name(value: object) -> str:
    text = str(value)
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in text)


def _jsonable(value: object) -> object:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, np.generic):
        return value.item()
    return value


def _load_rgb(path: str) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))


def _resize_to_height(image: Image.Image, height: int) -> Image.Image:
    if image.height == height:
        return image
    width = max(1, int(round(image.width * (height / image.height))))
    return image.resize((width, height), Image.Resampling.BILINEAR)


def _make_side_by_side(left_rgb: np.ndarray, right_rgb: np.ndarray, label: str) -> Image.Image:
    left = Image.fromarray(left_rgb)
    right = Image.fromarray(right_rgb)
    target_h = min(left.height, right.height, 720)
    left = _resize_to_height(left, target_h)
    right = _resize_to_height(right, target_h)

    label_h = 34
    canvas = Image.new("RGB", (left.width + right.width, target_h + label_h), "white")
    canvas.paste(left, (0, label_h))
    canvas.paste(right, (left.width, label_h))
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 9), label, fill=(0, 0, 0))
    draw.text((10, label_h + 10), "left", fill=(255, 255, 255))
    draw.text((left.width + 10, label_h + 10), "right aligned", fill=(255, 255, 255))
    return canvas


def _make_mosaic(
    image_paths: list[Path],
    output_path: Path,
    *,
    columns: int = 2,
    thumb_height: int = 360,
) -> None:
    if not image_paths:
        return
    columns = max(1, int(columns))
    thumb_height = max(1, int(thumb_height))

    thumbs: list[Image.Image] = []
    for path in image_paths:
        img = Image.open(path).convert("RGB")
        thumbs.append(_resize_to_height(img, thumb_height))

    cell_w = max(img.width for img in thumbs)
    cell_h = max(img.height for img in thumbs)
    rows = int(np.ceil(len(thumbs) / columns))
    mosaic = Image.new("RGB", (columns * cell_w, rows * cell_h), "white")

    for idx, img in enumerate(thumbs):
        row = idx // columns
        col = idx % columns
        x = col * cell_w
        y = row * cell_h
        mosaic.paste(img, (x, y))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    mosaic.save(output_path)


def _cache_path(cache_dir: Optional[str], image_id: str, segmentation_max_width: Optional[int]) -> Optional[Path]:
    if not cache_dir:
        return None
    suffix = f"w{int(segmentation_max_width)}" if segmentation_max_width else "full"
    return Path(cache_dir) / f"{image_id}_{suffix}.npy"


def _maybe_save_cached_masks(
    *,
    output_dir: Path,
    id_left: str,
    id_right: str,
    yaw: int,
    cache_dir: Optional[str],
    segmentation_max_width: Optional[int],
) -> None:
    left_path = _cache_path(cache_dir, id_left, segmentation_max_width)
    right_path = _cache_path(cache_dir, id_right, segmentation_max_width)
    if not left_path or not right_path or not left_path.exists() or not right_path.exists():
        return
    left_mask = np.load(left_path)
    right_mask = shift_equirectangular(np.load(right_path), yaw)
    np.save(output_dir / "left_seg_cache.npy", left_mask)
    np.save(output_dir / "right_seg_cache_aligned.npy", right_mask)


def _sample_metric_deciles(
    df: pd.DataFrame,
    metric: str,
    samples_per_decile: int,
    rng: np.random.Generator,
) -> list[tuple[str, pd.DataFrame]]:
    values = _to_numeric(df[metric])
    valid = df.loc[values.replace([np.inf, -np.inf], np.nan).notna()].copy()
    if valid.empty:
        logger.warning("No valid rows for metric %s", metric)
        return []

    valid["_metric_value"] = _to_numeric(valid[metric])
    try:
        valid["_decile"] = pd.qcut(
            valid["_metric_value"],
            q=10,
            labels=False,
            duplicates="drop",
        )
    except ValueError:
        logger.warning("Could not make deciles for metric %s", metric)
        return []

    out: list[tuple[str, pd.DataFrame]] = []
    max_decile = int(valid["_decile"].max())
    for decile in range(max_decile + 1):
        group = valid[valid["_decile"] == decile]
        if group.empty:
            continue
        take = min(samples_per_decile, len(group))
        random_state = int(rng.integers(0, np.iinfo(np.int32).max))
        sample = group.sample(n=take, random_state=random_state)
        start = decile * 10
        end = (decile + 1) * 10
        out.append((f"p{start:02d}_{end:03d}", sample))
    return out


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--results-csv",
        default=str(PROJECT_ROOT / "results.csv"),
        help="Input results CSV (default: Pairwise_image_finder/results.csv).",
    )
    ap.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT.parent / "metric_percentile_samples"),
        help="Directory for sampled preview folders.",
    )
    ap.add_argument("--access-token", required=True, help="Mapillary API token.")
    ap.add_argument(
        "--metrics",
        nargs="+",
        default=DEFAULT_METRICS,
        help=f"Metric columns to sample (default: {' '.join(DEFAULT_METRICS)}).",
    )
    ap.add_argument("--samples-per-decile", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--image-size", choices=["256", "1024", "2048", "original"], default="1024")
    ap.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    ap.add_argument(
        "--use-csv-yaw",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use best_yaw_deg from results.csv for exact previous alignment (default: true).",
    )
    ap.add_argument("--crop-keep", type=float, default=0.6)
    ap.add_argument("--crop-top-bias", type=float, default=0.0)
    ap.add_argument("--yaw-step", type=int, default=10)
    ap.add_argument("--image-cache-dir", default=None)
    ap.add_argument("--keep-downloaded-images", action="store_true")
    ap.add_argument("--segmentation-cache-dir", default=str(PROJECT_ROOT / "segmentation_cache"))
    ap.add_argument("--segmentation-max-width", type=int, default=1024)
    ap.add_argument(
        "--save-cached-segmentation-masks",
        action="store_true",
        help="If cached .npy masks exist, save aligned mask arrays next to each preview.",
    )
    ap.add_argument(
        "--mosaic-columns",
        type=int,
        default=2,
        help="Number of columns in each metric/decile mosaic image (default: 2).",
    )
    ap.add_argument(
        "--mosaic-thumb-height",
        type=int,
        default=360,
        help="Height in pixels for each side-by-side preview inside mosaics (default: 360).",
    )
    ap.add_argument("--mapillary-request-delay", type=float, default=0.1)
    ap.add_argument("--mapillary-max-retries", type=int, default=5)
    ap.add_argument("--mapillary-retry-delay", type=float, default=60.0)
    return ap.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    args = parse_args()
    if args.samples_per_decile <= 0:
        raise ValueError("--samples-per-decile must be > 0")
    if args.mosaic_columns <= 0:
        raise ValueError("--mosaic-columns must be > 0")
    if args.mosaic_thumb_height <= 0:
        raise ValueError("--mosaic-thumb-height must be > 0")
    if args.segmentation_max_width < 0:
        raise ValueError("--segmentation-max-width must be >= 0")
    segmentation_max_width = args.segmentation_max_width or None

    results_path = Path(args.results_csv)
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(results_path, on_bad_lines="skip")

    required = {"id_left", "id_right"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns in {results_path}: {missing}")
    for metric in args.metrics:
        if metric not in df.columns:
            raise ValueError(f"Metric column not found in {results_path}: {metric}")

    rng = np.random.default_rng(args.seed)
    samples: list[tuple[str, str, pd.DataFrame]] = []
    for metric in args.metrics:
        for decile_name, sample in _sample_metric_deciles(
            df, metric, args.samples_per_decile, rng
        ):
            samples.append((metric, decile_name, sample))

    total_rows = sum(len(sample) for _, _, sample in samples)
    logger.info("Selected %d rows across %d metric/decile groups", total_rows, len(samples))

    device = None
    extractor = None
    matcher = None
    if not args.use_csv_yaw:
        device = pick_device(args.device)
        logger.info("Recomputing yaw on device: %s", device)
        extractor, matcher = init_models(device)

    cache_ctx = None
    if args.image_cache_dir:
        image_cache_dir = Path(args.image_cache_dir)
        image_cache_dir.mkdir(parents=True, exist_ok=True)
    elif args.keep_downloaded_images:
        image_cache_dir = output_root / "_downloaded_images"
        image_cache_dir.mkdir(parents=True, exist_ok=True)
    else:
        cache_ctx = tempfile.TemporaryDirectory(prefix="metric_sample_images_")
        image_cache_dir = Path(cache_ctx.name)

    manifest_rows: list[dict] = []
    try:
        with requests.Session() as session:
            for metric, decile_name, sample in samples:
                group_dir = output_root / _safe_name(metric) / decile_name
                group_dir.mkdir(parents=True, exist_ok=True)
                mosaic_inputs: list[Path] = []
                for rank, (_, row) in enumerate(sample.reset_index(drop=True).iterrows(), start=1):
                    id_left = str(row["id_left"])
                    id_right = str(row["id_right"])
                    pair_dir = group_dir / f"{rank:02d}_{id_left}_{id_right}"
                    pair_dir.mkdir(parents=True, exist_ok=True)

                    left_path = fetch_and_cache_image(
                        id_left,
                        args.access_token,
                        str(image_cache_dir),
                        session=session,
                        size=args.image_size,
                        request_delay=args.mapillary_request_delay,
                        max_retries=args.mapillary_max_retries,
                        retry_delay=args.mapillary_retry_delay,
                    )
                    right_path = fetch_and_cache_image(
                        id_right,
                        args.access_token,
                        str(image_cache_dir),
                        session=session,
                        size=args.image_size,
                        request_delay=args.mapillary_request_delay,
                        max_retries=args.mapillary_max_retries,
                        retry_delay=args.mapillary_retry_delay,
                    )
                    if not left_path or not right_path:
                        logger.warning("Skipping %s/%s: image download failed", id_left, id_right)
                        continue

                    left_rgb = _load_rgb(left_path)
                    right_rgb = _load_rgb(right_path)
                    if args.use_csv_yaw and "best_yaw_deg" in row and pd.notna(row["best_yaw_deg"]):
                        yaw = int(round(float(row["best_yaw_deg"]))) % 360
                        method = str(row.get("panorama_alignment_method", "csv_best_yaw"))
                    else:
                        yaw, method = estimate_yaw_from_keypoints_once(
                            left_rgb,
                            right_rgb,
                            device,
                            extractor,
                            matcher,
                            yaw_step=args.yaw_step,
                            crop_keep=args.crop_keep,
                            crop_top_bias=args.crop_top_bias,
                        )
                    right_aligned = shift_equirectangular(right_rgb, yaw)

                    metric_value = row.get(metric, "NA")
                    label = (
                        f"{metric}={metric_value} | decile={decile_name} | "
                        f"yaw={yaw} | {id_left} vs {id_right}"
                    )
                    Image.fromarray(left_rgb).save(pair_dir / "left.png")
                    Image.fromarray(right_aligned).save(pair_dir / f"right_aligned_yaw_{yaw}.png")
                    side_by_side_path = pair_dir / "side_by_side.png"
                    _make_side_by_side(left_rgb, right_aligned, label).save(side_by_side_path)
                    mosaic_inputs.append(side_by_side_path)

                    if args.save_cached_segmentation_masks:
                        _maybe_save_cached_masks(
                            output_dir=pair_dir,
                            id_left=id_left,
                            id_right=id_right,
                            yaw=yaw,
                            cache_dir=args.segmentation_cache_dir,
                            segmentation_max_width=segmentation_max_width,
                        )

                    metadata = {
                        "metric": metric,
                        "decile": decile_name,
                        "metric_value": _jsonable(metric_value),
                        "id_left": id_left,
                        "id_right": id_right,
                        "best_yaw_deg": yaw,
                        "alignment_method": method,
                        "results_row": {
                            key: _jsonable(value)
                            for key, value in row.to_dict().items()
                        },
                    }
                    with open(pair_dir / "metadata.json", "w", encoding="utf-8") as f:
                        json.dump(metadata, f, indent=2)
                    manifest_rows.append(metadata)

                _make_mosaic(
                    mosaic_inputs,
                    group_dir / "mosaic.png",
                    columns=args.mosaic_columns,
                    thumb_height=args.mosaic_thumb_height,
                )
                _make_mosaic(
                    mosaic_inputs,
                    output_root / _safe_name(metric) / f"{decile_name}_mosaic.png",
                    columns=args.mosaic_columns,
                    thumb_height=args.mosaic_thumb_height,
                )

        with open(output_root / "sample_manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest_rows, f, indent=2)
        logger.info("Wrote %d sampled previews to %s", len(manifest_rows), output_root)
    finally:
        if cache_ctx is not None:
            cache_ctx.cleanup()


if __name__ == "__main__":
    main()
