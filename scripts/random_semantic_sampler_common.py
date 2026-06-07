#!/usr/bin/env python
"""Random Mapillary image semantic sampling helpers."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, TYPE_CHECKING

import numpy as np
import pandas as pd
import requests
from PIL import Image
from tqdm import tqdm


def _find_project_root() -> Path:
    here = Path(__file__).resolve()
    candidates = [Path.cwd(), Path.cwd() / "Pairwise_image_finder"]
    for parent in [Path.cwd(), *Path.cwd().parents, *here.parents]:
        candidates.extend([parent, parent / "Pairwise_image_finder"])
    for candidate in candidates:
        if (candidate / "utils").is_dir() and (candidate / "scripts").is_dir():
            return candidate.resolve()
    raise RuntimeError("Could not find Pairwise_image_finder project root.")


PROJECT_ROOT = _find_project_root()
sys.path.insert(0, str(PROJECT_ROOT))

from utils.area_pairing import fetch_images_in_area  # noqa: E402
from utils.config import add_config_argument, parse_args_with_config  # noqa: E402
from utils.fetcher import fetch_image  # noqa: E402

if TYPE_CHECKING:
    from utils.segmentation import Segmenter


logger = logging.getLogger(__name__)
SEASON_MONTHS = {
    "spring": {3, 4, 5},
    "summer": {6, 7, 8},
    "autumn": {9, 10, 11},
    "winter": {12, 1, 2},
}


def default_config_path() -> Path:
    cwd_config = Path.cwd() / "config.toml"
    if cwd_config.exists():
        return cwd_config
    return PROJECT_ROOT / "config.toml"


def pick_device(prefer: Optional[str] = None):
    import torch

    if prefer == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested with --device cuda, but PyTorch cannot use CUDA. "
                "Check the NVIDIA driver, CUDA-compatible PyTorch build, and GPU visibility."
            )
        return torch.device("cuda")
    if prefer == "mps":
        if not (getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()):
            raise RuntimeError(
                "MPS was requested with --device mps, but PyTorch cannot use MPS on this machine."
            )
        return torch.device("mps")
    if prefer == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def parse_captured_year(value: object) -> Optional[int]:
    dt = parse_captured_datetime(value)
    return dt.year if dt is not None else None


def parse_captured_datetime(value: object) -> Optional[datetime]:
    if value is None or pd.isna(value):
        return None
    try:
        return datetime.fromtimestamp(int(value) / 1000, tz=timezone.utc)
    except (TypeError, ValueError, OSError):
        return None


def season_from_month(month: int) -> str:
    for season, months in SEASON_MONTHS.items():
        if int(month) in months:
            return season
    raise ValueError(f"Invalid month: {month}")


def daytime_label(captured_at: object, lat: object, lon: object) -> Optional[str]:
    dt = parse_captured_datetime(captured_at)
    if dt is None:
        return None
    try:
        from astral import LocationInfo
        from astral.sun import sun
    except ImportError:
        raise ImportError(
            'astral is required for --time-filter time. '
            'Install area dependencies with: pip install ".[area]"'
        )

    loc = LocationInfo(latitude=float(lat), longitude=float(lon))
    s = sun(loc.observer, date=dt.date(), tzinfo=timezone.utc)
    return "day" if s["sunrise"] <= dt <= s["sunset"] else "night"


def crop_vertical_middle(
    image: Image.Image,
    keep_ratio: float,
    top_bias: float = 0.0,
) -> Image.Image:
    """Crop a central vertical band. keep_ratio=1.0 returns the original image."""
    keep_ratio = float(keep_ratio)
    if keep_ratio >= 1.0:
        return image
    if keep_ratio <= 0.0:
        raise ValueError("--crop-keep must be > 0")

    width, height = image.size
    crop_h = max(1, int(round(height * keep_ratio)))
    extra = height - crop_h
    bias = max(-1.0, min(1.0, float(top_bias)))
    y1 = int(round((extra / 2.0) * (1.0 + bias)))
    y1 = max(0, min(extra, y1))
    y2 = y1 + crop_h
    return image.crop((0, y1, width, y2))


def resize_for_segmentation(image: Image.Image, max_width: Optional[int]) -> Image.Image:
    if max_width and image.width > max_width:
        scale = float(max_width) / float(image.width)
        return image.resize(
            (int(max_width), max(1, int(round(image.height * scale)))),
            Image.Resampling.BILINEAR,
        )
    return image


def segment_cached(
    segmenter: "Segmenter",
    image: Image.Image,
    cache_path: Optional[Path],
    max_width: Optional[int],
) -> np.ndarray:
    if cache_path and cache_path.exists():
        try:
            return np.load(cache_path)
        except Exception:
            logger.warning("Failed to read segmentation cache: %s", cache_path)

    image_for_seg = resize_for_segmentation(image, max_width)
    mask = segmenter.segment_image(image_for_seg)

    if cache_path:
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(cache_path, mask)
        except Exception as exc:
            logger.warning("Failed to write segmentation cache %s: %s", cache_path, exc)

    return mask


def json_props(props: dict[int, float]) -> str:
    return json.dumps({int(k): float(v) for k, v in props.items()}, sort_keys=True)


def named_props(props: dict[int, float], id2label: dict[int, str]) -> str:
    return json.dumps(
        {id2label.get(int(k), f"class_{int(k)}"): float(v) for k, v in props.items()},
        sort_keys=True,
    )


def safe_label(text: str) -> str:
    return re.sub(r"[^0-9a-zA-Z]+", "_", text.strip().lower()).strip("_") or "class"


def build_arg_parser(
    default_image_type: str,
    default_output: str,
    default_crop_keep: float,
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Randomly sample Mapillary images inside a WKT area from two year groups, "
            "segment each image, and write per-image semantic class proportions."
        )
    )
    add_config_argument(parser)
    parser.add_argument("--area-wkt", help="WKT polygon/multipolygon in lon/lat coordinates.")
    parser.add_argument(
        "--access-token",
        help=(
            "Mapillary API token. Prefer access_token_env in config.toml "
            "with the real token stored in local .env."
        ),
    )
    parser.add_argument("--output", default=default_output, help=f"Output CSV (default: {default_output}).")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Fetch metadata and report available/sample counts without downloading "
            "images, loading segmentation models, or writing output."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Start a fresh output CSV. By default, an existing output CSV is "
            "treated as a resume file and already-written image IDs are skipped."
        ),
    )
    parser.add_argument(
        "--samples-per-group",
        "-x",
        type=int,
        help="Number of random images to sample from each year group.",
    )
    parser.add_argument(
        "--image-type",
        choices=["flat", "panorama"],
        default=default_image_type,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--time-filter",
        nargs="+",
        choices=["any", "year", "month", "season", "same-season", "time"],
        default=["year"],
        help=(
            "Sampling balance mode. any=random images, year=old/new year groups, "
            "month=equal samples per month, season=equal samples per season, "
            "same-season=old/new samples within each season, time=day/night "
            '(requires: pip install ".[area]").'
        ),
    )
    parser.add_argument("--year-group-old", nargs="+", type=int, default=[2016, 2017, 2018])
    parser.add_argument("--year-group-new", nargs="+", type=int, default=[2024, 2025, 2026])
    parser.add_argument("--old-label", default="old")
    parser.add_argument("--new-label", default="new")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducible sampling.")
    parser.add_argument("--allow-short", action="store_true", help="Sample all available images if a year group has fewer than X.")
    parser.add_argument("--image-size", choices=["256", "1024", "2048", "original"], default="1024")
    parser.add_argument("--device", choices=["cuda", "mps", "cpu"], default=None)
    parser.add_argument("--seg-dataset", choices=["cityscapes", "ade20k"], default="cityscapes")
    parser.add_argument(
        "--segmentation-max-width",
        type=int,
        default=1024,
        help="Resize images to this max width before segmentation; use 0 for full resolution.",
    )
    parser.add_argument(
        "--segmentation-cache-dir",
        default=None,
        help="Optional directory for cached .npy segmentation masks.",
    )
    parser.add_argument(
        "--save-images-dir",
        default=None,
        help="Optional directory to save downloaded sampled images. By default images are not saved.",
    )
    parser.add_argument(
        "--crop-keep",
        type=float,
        default=default_crop_keep,
        help=(
            "Vertical keep ratio before segmentation, useful for panorama central "
            f"crops (default: {default_crop_keep})."
        ),
    )
    parser.add_argument("--crop-top-bias", type=float, default=0.0)
    parser.add_argument("--mapillary-request-delay", type=float, default=0.1)
    parser.add_argument("--mapillary-max-retries", type=int, default=5)
    parser.add_argument("--mapillary-retry-delay", type=float, default=60.0)
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if not args.area_wkt:
        raise ValueError("--area-wkt is required")
    if not args.access_token:
        raise ValueError(
            "A Mapillary token is required; set it in .env/config or pass --access-token."
        )
    if args.samples_per_group is None:
        raise ValueError("--samples-per-group is required")
    if args.samples_per_group <= 0:
        raise ValueError("--samples-per-group must be > 0")
    if args.segmentation_max_width < 0:
        raise ValueError("--segmentation-max-width must be >= 0")
    if args.crop_keep <= 0:
        raise ValueError("--crop-keep must be > 0")
    if args.mapillary_request_delay < 0:
        raise ValueError("--mapillary-request-delay must be >= 0")
    if args.mapillary_max_retries < 0:
        raise ValueError("--mapillary-max-retries must be >= 0")
    if args.mapillary_retry_delay < 0:
        raise ValueError("--mapillary-retry-delay must be >= 0")
    sampler_mode(args)


def _load_resume_state(
    output_path: Path,
    *,
    overwrite: bool,
    labels: list[str],
) -> tuple[set[str], dict[str, int]]:
    counts = {label: 0 for label in labels}
    if overwrite or not output_path.exists() or output_path.stat().st_size == 0:
        return set(), counts

    try:
        existing = pd.read_csv(output_path, dtype=str)
    except Exception as exc:
        raise ValueError(
            f"Could not read existing output CSV for resume: {output_path}"
        ) from exc

    if "id" not in existing.columns:
        raise ValueError(
            f"Existing output CSV {output_path} has no 'id' column. "
            "Use a different --output or --overwrite."
        )

    existing_ids = set(existing["id"].dropna().astype(str))
    if "year_group" in existing.columns:
        value_counts = existing["year_group"].value_counts()
        for label in labels:
            counts[label] = int(value_counts.get(label, 0))

    return existing_ids, counts


def sampler_mode(args: argparse.Namespace) -> str:
    active = [value for value in (args.time_filter or ["any"]) if value != "any"]
    if not active:
        return "any"
    if len(active) > 1:
        raise ValueError(
            "Random semantic sampling supports one active --time-filter at a time. "
            "Use one of: any, year, month, season, same-season, time."
        )
    return active[0]


def _add_time_columns(images: pd.DataFrame, mode: str) -> pd.DataFrame:
    images = images.copy()
    images["captured_dt"] = images["captured_at"].map(parse_captured_datetime)
    images = images[images["captured_dt"].notna()].copy()
    images["year"] = images["captured_dt"].map(lambda dt: int(dt.year))
    images["month"] = images["captured_dt"].map(lambda dt: int(dt.month))
    images["season"] = images["month"].map(season_from_month)
    if mode == "time":
        images["daytime_group"] = images.apply(
            lambda row: daytime_label(row["captured_at"], row["lat"], row["lon"]),
            axis=1,
        )
        images = images[images["daytime_group"].notna()].copy()
    return images


def _build_sample_groups(
    images: pd.DataFrame,
    args: argparse.Namespace,
) -> dict[str, pd.DataFrame]:
    mode = sampler_mode(args)
    if mode == "any":
        return {"any": images.copy()}
    if mode == "year":
        return {
            args.old_label: images[images["year"].isin(args.year_group_old)].copy(),
            args.new_label: images[images["year"].isin(args.year_group_new)].copy(),
        }
    if mode == "month":
        return {
            f"month_{month:02d}": images[images["month"] == month].copy()
            for month in range(1, 13)
        }
    if mode == "season":
        return {
            season: images[images["season"] == season].copy()
            for season in SEASON_MONTHS
        }
    if mode == "same-season":
        groups: dict[str, pd.DataFrame] = {}
        old = images[images["year"].isin(args.year_group_old)]
        new = images[images["year"].isin(args.year_group_new)]
        for season in SEASON_MONTHS:
            groups[f"{args.old_label}_{season}"] = old[old["season"] == season].copy()
            groups[f"{args.new_label}_{season}"] = new[new["season"] == season].copy()
        return groups
    if mode == "time":
        return {
            "day": images[images["daytime_group"] == "day"].copy(),
            "night": images[images["daytime_group"] == "night"].copy(),
        }
    raise ValueError(f"Unsupported sampler mode: {mode}")


def _shuffled_group_candidates(
    group: pd.DataFrame,
    *,
    label: str,
    existing_ids: set[str],
    rng: np.random.Generator,
) -> list[dict[str, object]]:
    group = group[~group["id"].astype(str).isin(existing_ids)].copy()
    if group.empty:
        return []
    group["year_group"] = label
    random_state = int(rng.integers(0, np.iinfo(np.int32).max))
    return group.sample(frac=1.0, random_state=random_state).to_dict("records")


def _make_output_fields(segmenter: "Segmenter") -> tuple[list[str], list[int], list[str], list[str]]:
    id2label = getattr(segmenter, "id2label", {}) or {}
    colors = getattr(segmenter, "colors", {}) or {}
    class_ids = sorted(set(map(int, id2label.keys())) | set(map(int, colors.keys())))

    base_fields = [
        "id",
        "captured_at",
        "year",
        "year_group",
        "lat",
        "lon",
        "compass_angle",
        "is_pano",
        "image_type",
        "source_width",
        "source_height",
        "analysis_width",
        "analysis_height",
        "seg_width",
        "seg_height",
        "seg_dataset",
        "crop_keep",
        "crop_top_bias",
        "class_props_json",
        "class_props_named_json",
        "class_props_no_temporary_json",
        "class_props_no_temporary_named_json",
        "temporary_fraction",
    ]
    prop_fields = [
        f"prop_{class_id:03d}_{safe_label(id2label.get(class_id, f'class_{class_id}'))}"
        for class_id in class_ids
    ]
    no_temp_fields = [f"no_temp_{field}" for field in prop_fields]
    return [*base_fields, *prop_fields, *no_temp_fields], class_ids, prop_fields, no_temp_fields


def _check_existing_header(
    output_path: Path,
    *,
    fieldnames: list[str],
    overwrite: bool,
) -> None:
    if overwrite or not output_path.exists() or output_path.stat().st_size == 0:
        return
    with output_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            return
    if header != fieldnames:
        raise ValueError(
            f"Existing output CSV {output_path} has a different header. "
            "Use a new --output path or --overwrite."
        )


def _log_dry_run_plan(
    args: argparse.Namespace,
    groups: dict[str, pd.DataFrame],
) -> None:
    labels = list(groups)
    output_path = Path(args.output)
    existing_ids, existing_counts = _load_resume_state(
        output_path,
        overwrite=args.overwrite,
        labels=labels,
    )

    targets = {
        label: args.samples_per_group
        for label in labels
    }
    available = {label: len(groups[label]) for label in labels}
    for label, group in groups.items():
        if len(group) < targets[label] and not args.allow_short:
            raise ValueError(
                f"Only found {len(group)} image(s) for {label}; "
                f"need {targets[label]}. Use --allow-short to sample all available."
            )
        if args.allow_short:
            targets[label] = min(targets[label], len(group))

    logger.info("Dry run: sampling mode: %s", sampler_mode(args))
    if existing_ids:
        logger.info("Dry run: existing output rows that would be treated as resume state: %d", len(existing_ids))
    for label in labels:
        candidates = groups[label][~groups[label]["id"].astype(str).isin(existing_ids)]
        remaining = max(0, targets[label] - existing_counts.get(label, 0))
        logger.info(
            "Dry run: %s available=%d existing=%d target=%d remaining=%d unprocessed_candidates=%d",
            label,
            available[label],
            existing_counts.get(label, 0),
            targets[label],
            remaining,
            len(candidates),
        )
        if len(candidates) < remaining and not args.allow_short:
            raise ValueError(
                f"Only {len(candidates)} unprocessed candidate(s) remain for {label}; "
                f"need {remaining}. Use --allow-short or lower --samples-per-group."
            )

    logger.info("Dry run: no images downloaded, segmentation models loaded, output CSVs written, or caches saved.")


def run(
    default_image_type: str,
    default_output: str,
    default_crop_keep: float = 1.0,
    argv: Optional[list[str]] = None,
) -> None:
    parser = build_arg_parser(default_image_type, default_output, default_crop_keep)
    args = parse_args_with_config(
        parser,
        argv=argv,
        default_config_path=default_config_path(),
        script_sections=("sampler", "semantic_sampler", "random_semantic_sampler"),
    )
    validate_args(args)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )

    segmentation_max_width = args.segmentation_max_width or None
    rng = np.random.default_rng(args.seed)

    logger.info("Fetching %s image metadata inside WKT area", args.image_type)
    images = fetch_images_in_area(
        args.area_wkt,
        args.access_token,
        image_type=args.image_type,
        request_delay=args.mapillary_request_delay,
        max_retries=args.mapillary_max_retries,
        retry_delay=args.mapillary_retry_delay,
    )
    if images.empty:
        raise RuntimeError(f"No {args.image_type} images found inside WKT area.")

    mode = sampler_mode(args)
    images = _add_time_columns(images, mode)
    if images.empty:
        raise RuntimeError("No images with usable capture timestamps remain for sampling.")
    groups = _build_sample_groups(images, args)
    logger.info("Sampling mode: %s", mode)
    for label, group in groups.items():
        logger.info("Available %s images for %s: %d", args.image_type, label, len(group))

    if args.dry_run:
        _log_dry_run_plan(args, groups)
        return

    device = pick_device(args.device)
    logger.info("Importing segmentation module")
    from utils.segmentation import Segmenter

    logger.info("Initializing %s segmenter on %s", args.seg_dataset, device)
    segmenter = Segmenter(model_type=args.seg_dataset, device=device)
    id2label = getattr(segmenter, "id2label", {}) or {}

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True) if output_path.parent != Path(".") else None
    existing_ids, existing_counts = _load_resume_state(
        output_path,
        overwrite=args.overwrite,
        labels=list(groups),
    )
    if existing_ids:
        logger.info(
            "Resuming from %s with %d existing image row(s)",
            output_path,
            len(existing_ids),
        )

    targets = {
        label: args.samples_per_group
        for label in groups
    }
    available = {label: len(group) for label, group in groups.items()}
    for label, group in groups.items():
        if len(group) < targets[label] and not args.allow_short:
            raise ValueError(
                f"Only found {len(group)} image(s) for {label}; "
                f"need {targets[label]}. Use --allow-short to sample all available."
            )
        if args.allow_short:
            targets[label] = min(targets[label], len(group))

    remaining = {
        label: max(0, targets[label] - existing_counts.get(label, 0))
        for label in targets
    }
    for label in targets:
        if remaining[label] == 0:
            logger.info(
                "%s already has quota met: %d/%d row(s)",
                label,
                existing_counts.get(label, 0),
                targets[label],
            )
        else:
            logger.info(
                "%s needs %d more row(s) to reach %d",
                label,
                remaining[label],
                targets[label],
            )

    candidates = {
        label: _shuffled_group_candidates(
            groups[label],
            label=label,
            existing_ids=existing_ids,
            rng=rng,
        )
        for label in targets
    }
    for label in targets:
        if len(candidates[label]) < remaining[label] and not args.allow_short:
            raise ValueError(
                f"Only {len(candidates[label])} unprocessed candidate(s) remain for {label}; "
                f"need {remaining[label]}. Use --allow-short or lower --samples-per-group."
            )

    cache_dir = Path(args.segmentation_cache_dir) if args.segmentation_cache_dir else None
    save_images_dir = Path(args.save_images_dir) if args.save_images_dir else None
    if save_images_dir:
        save_images_dir.mkdir(parents=True, exist_ok=True)

    fieldnames, class_ids, prop_fields, no_temp_fields = _make_output_fields(segmenter)
    _check_existing_header(output_path, fieldnames=fieldnames, overwrite=args.overwrite)
    write_header = (
        args.overwrite
        or not output_path.exists()
        or output_path.stat().st_size == 0
    )
    mode = "w" if args.overwrite else "a"
    written = 0

    with output_path.open(mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
            f.flush()

        with requests.Session() as session:
            total_needed = sum(remaining.values())
            with tqdm(total=total_needed, desc="segment", unit="image") as pbar:
                while any(count > 0 for count in remaining.values()):
                    active_labels = [
                        label
                        for label, count in remaining.items()
                        if count > 0 and candidates[label]
                    ]
                    if not active_labels:
                        break

                    label = str(rng.choice(active_labels))
                    meta = candidates[label].pop(0)
                    image_id = str(meta["id"])
                    if image_id in existing_ids:
                        continue

                    image = fetch_image(
                        image_id,
                        session,
                        args.access_token,
                        size=args.image_size,
                        request_delay=args.mapillary_request_delay,
                        max_retries=args.mapillary_max_retries,
                        retry_delay=args.mapillary_retry_delay,
                    )
                    if image is None:
                        logger.warning("Skipping %s: download failed", image_id)
                        existing_ids.add(image_id)
                        continue

                    image = image.convert("RGB")
                    image_for_analysis = crop_vertical_middle(
                        image,
                        keep_ratio=args.crop_keep,
                        top_bias=args.crop_top_bias,
                    )
                    if save_images_dir:
                        image_for_analysis.save(save_images_dir / f"{image_id}.png")

                    cache_path = None
                    if cache_dir:
                        suffix = f"w{int(segmentation_max_width)}" if segmentation_max_width else "full"
                        crop_suffix = (
                            f"_crop{args.crop_keep:.3f}_bias{args.crop_top_bias:.3f}"
                            if args.crop_keep < 1.0 or abs(args.crop_top_bias) > 1e-9
                            else ""
                        )
                        cache_path = cache_dir / f"{image_id}_{suffix}{crop_suffix}.npy"

                    mask = segment_cached(
                        segmenter,
                        image_for_analysis,
                        cache_path=cache_path,
                        max_width=segmentation_max_width,
                    )

                    props = segmenter.compute_class_proportions(mask)

                    if getattr(segmenter, "temporary_class_ids", []):
                        temp_pixels = np.isin(mask, segmenter.temporary_class_ids)
                    else:
                        temp_pixels = np.zeros(mask.shape, dtype=bool)
                    props_no_temp = segmenter.compute_class_proportions(mask, ~temp_pixels)
                    temporary_fraction = float(temp_pixels.mean()) if temp_pixels.size else 0.0

                    row: dict[str, object] = {
                        "id": image_id,
                        "captured_at": meta.get("captured_at"),
                        "year": int(meta["year"]),
                        "year_group": meta["year_group"],
                        "lat": float(meta["lat"]),
                        "lon": float(meta["lon"]),
                        "compass_angle": meta.get("compass_angle"),
                        "is_pano": bool(meta.get("is_pano")),
                        "image_type": args.image_type,
                        "source_width": image.width,
                        "source_height": image.height,
                        "analysis_width": image_for_analysis.width,
                        "analysis_height": image_for_analysis.height,
                        "seg_width": int(mask.shape[1]),
                        "seg_height": int(mask.shape[0]),
                        "seg_dataset": args.seg_dataset,
                        "crop_keep": float(args.crop_keep),
                        "crop_top_bias": float(args.crop_top_bias),
                        "class_props_json": json_props(props),
                        "class_props_named_json": named_props(props, id2label),
                        "class_props_no_temporary_json": json_props(props_no_temp),
                        "class_props_no_temporary_named_json": named_props(props_no_temp, id2label),
                        "temporary_fraction": temporary_fraction,
                    }
                    for class_id, field, no_temp_field in zip(class_ids, prop_fields, no_temp_fields):
                        row[field] = float(props.get(class_id, 0.0))
                        row[no_temp_field] = float(props_no_temp.get(class_id, 0.0))

                    writer.writerow(row)
                    f.flush()
                    os.fsync(f.fileno())

                    existing_ids.add(image_id)
                    remaining[label] -= 1
                    written += 1
                    pbar.update(1)

    if any(count > 0 for count in remaining.values()) and not args.allow_short:
        raise RuntimeError(
            "Could not fulfill requested quota. Remaining: "
            + ", ".join(f"{label}={count}" for label, count in remaining.items())
        )

    final = pd.read_csv(output_path, dtype=str)
    final_counts = final.groupby(["year_group", "year"]).size()
    logger.info("Wrote %d new segmented image row(s) to %s", written, output_path)
    logger.info("Final sample counts by group/year:\n%s", final_counts.to_string())


if __name__ == "__main__":
    raise SystemExit(
        "random_semantic_sampler_common.py is a shared helper and should not be "
        "executed directly. Run sample_random_flat_semantics.py or "
        "sample_random_panorama_semantics.py instead."
    )
