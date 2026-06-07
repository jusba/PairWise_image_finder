#!/usr/bin/env python
"""
Process image pairs using LightGlue feature matching + optional semantic segmentation.

Two input modes (mutually exclusive):

  CSV mode (default)
    --input-csv  Pair CSV with columns filename_left and filename_right
                 (plus optional id_left, id_right, date_left, date_right, index).
                 Images must exist locally in --images-dir.

  Area mode
    --area-wkt   WKT polygon; queries Mapillary for images in that area, pairs them by
                 GPS proximity, downloads them, and processes every matched pair.
    Requires a Mapillary token via .env/config or --access-token.
"""
import logging
import os
import csv
import argparse
import traceback
import sys
import tempfile
import random
from collections import defaultdict, deque
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.io_utils import (
    CSV_HEADER, CSV_HEADER_PANORAMA, CSV_HEADER_SCALE,
    CSV_HEADER_AREA, CSV_HEADER_PANORAMA_AREA, CSV_HEADER_SCALE_AREA,
    case_insensitive_join, ensure_dir, completed_pairs_set,
    with_optional_image_metrics,
)
from utils.fetcher import fetch_and_cache_image
from utils.config import add_config_argument, parse_args_with_config
from utils.run_manifest import write_run_manifest

logger = logging.getLogger(__name__)

TIMING_HEADER = ["id_left", "id_right", "stage", "seconds"]
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PAIR_CSV_FILENAME_LEFT = ("filename_left", "left_filename", "filename_1", "filename_a")
PAIR_CSV_FILENAME_RIGHT = ("filename_right", "right_filename", "filename_2", "filename_b")
PAIR_CSV_ID_LEFT = ("id_left", "left_id", "id_1", "id_a")
PAIR_CSV_ID_RIGHT = ("id_right", "right_id", "id_2", "id_b")
PAIR_CSV_DATE_LEFT = ("date_left", "left_date", "date_1", "date_a")
PAIR_CSV_DATE_RIGHT = ("date_right", "right_date", "date_2", "date_b")

RESULT_FILTERS = [
    (
        "filter_match_ratio_min",
        "lightglue_match_ratio",
        ">=",
        "LightGlue match ratio",
    ),
    (
        "filter_avg_distance_max",
        "lightglue_avg_distance",
        "<=",
        "LightGlue average normalized distance",
    ),
    (
        "filter_keypoint_coverage_min",
        "lightglue_keypoint_coverage_min",
        ">=",
        "LightGlue matched-keypoint non-sky coverage",
    ),
    (
        "filter_keypoint_hull_iou_min",
        "lightglue_keypoint_hull_iou",
        ">=",
        "LightGlue matched-keypoint hull IoU",
    ),
    (
        "filter_road_iou_min",
        "seg_overlap_road_iou",
        ">=",
        "semantic road IoU",
    ),
    (
        "filter_mean_iou_min",
        "seg_overlap_mean_iou",
        ">=",
        "semantic mean IoU",
    ),
]


# ---------- helpers ----------

def _default_config_path() -> Path:
    cwd_config = Path.cwd() / "config.toml"
    if cwd_config.exists():
        return cwd_config
    return PROJECT_ROOT / "config.toml"


def resolve_local(images_dir: str | None, filename: str | None, img_id: str) -> str | None:
    if not images_dir:
        return None
    if filename and str(filename).strip():
        p = os.path.join(images_dir, filename)
        if os.path.exists(p):
            return p
        found = case_insensitive_join(images_dir, filename)
        if found:
            return found
    for ext in (".png", ".jpg", ".jpeg"):
        p = os.path.join(images_dir, f"{img_id}{ext}")
        if os.path.exists(p):
            return p
        found = case_insensitive_join(images_dir, f"{img_id}{ext}")
        if found:
            return found
    return None


def _csv_value(row: dict, names: tuple[str, ...], default=None):
    for name in names:
        if name in row and not pd.isna(row[name]):
            value = str(row[name]).strip()
            if value:
                return value
    return default


def _id_from_filename(filename: str, fallback: str) -> str:
    stem = Path(str(filename)).stem
    return stem if stem else fallback


def _has_explicit_pair_columns(df: pd.DataFrame) -> bool:
    columns = set(df.columns)
    return bool(columns.intersection(PAIR_CSV_FILENAME_LEFT)) and bool(
        columns.intersection(PAIR_CSV_FILENAME_RIGHT)
    )


def _csv_output_header(args) -> list[str]:
    if args.panorama:
        header = CSV_HEADER_PANORAMA
    elif args.scale_search:
        header = CSV_HEADER_SCALE
    else:
        header = CSV_HEADER
    return with_optional_image_metrics(header, args.image_metrics)


def _count(value: int | None) -> str:
    if value is None:
        return "not applicable"
    return f"{int(value):,}"


def _yes_no(value: bool) -> str:
    return "yes" if value else "no"


def _will_save_images(args) -> bool:
    return bool(args.area_wkt and args.save_mapillary_images and not args.no_save_artifacts)


def _dry_run_time_filter_label(args) -> str:
    active = [f for f in (args.time_filter or []) if f != "any"]
    if len(active) > 1:
        return "after time filters"
    if args.year_group_left or active == ["year"]:
        return "after year filter"
    if active == ["month"]:
        return "after month filter"
    if active == ["season"]:
        return "after season filter"
    if active == ["same-season"]:
        return "after same-season filter"
    if active == ["time"]:
        return "after day/night filter"
    return "after time filters"


def _active_result_filters(args) -> list[dict[str, object]]:
    filters: list[dict[str, object]] = []
    for attr, column, operator, label in RESULT_FILTERS:
        value = getattr(args, attr, None)
        if value is None:
            continue
        filters.append(
            {
                "setting": attr,
                "column": column,
                "operator": operator,
                "threshold": float(value),
                "label": label,
            }
        )
    return filters


def _apply_numeric_filter(
    frame: pd.DataFrame,
    *,
    column: str,
    operator: str,
    threshold: float,
) -> pd.Series:
    values = pd.to_numeric(frame[column], errors="coerce")
    if operator == ">=":
        return values >= threshold
    if operator == "<=":
        return values <= threshold
    raise ValueError(f"Unsupported filter operator: {operator}")


def _write_filtered_results(args) -> dict[str, object]:
    filters = _active_result_filters(args)
    if not args.filtered_output:
        return {
            "enabled": False,
            "reason": "no filtered_output configured",
            "criteria": filters,
        }

    if not os.path.exists(args.output) or os.stat(args.output).st_size == 0:
        logger.warning("Filtered results requested, but output CSV is empty: %s", args.output)
        return {
            "enabled": True,
            "filtered_output": args.filtered_output,
            "source_rows": 0,
            "filtered_rows": 0,
            "criteria": filters,
        }

    frame = pd.read_csv(args.output, on_bad_lines="skip")
    mask = pd.Series(True, index=frame.index)
    missing_columns: list[str] = []

    for spec in filters:
        column = str(spec["column"])
        if column not in frame.columns:
            missing_columns.append(column)
            continue
        mask &= _apply_numeric_filter(
            frame,
            column=column,
            operator=str(spec["operator"]),
            threshold=float(spec["threshold"]),
        ).fillna(False)

    if missing_columns:
        raise ValueError(
            "Cannot write filtered results because these columns are missing from "
            f"{args.output}: {', '.join(missing_columns)}"
        )

    filtered = frame[mask].copy()
    filtered_output = Path(args.filtered_output)
    if filtered_output.parent != Path("."):
        filtered_output.parent.mkdir(parents=True, exist_ok=True)
    filtered.to_csv(filtered_output, index=False)

    criteria_text = ", ".join(
        f"{spec['column']} {spec['operator']} {spec['threshold']}"
        for spec in filters
    ) or "no criteria; copied all rows"
    logger.info(
        "Filtered results: kept %s/%s rows in %s (%s)",
        _count(len(filtered)),
        _count(len(frame)),
        filtered_output,
        criteria_text,
    )
    return {
        "enabled": True,
        "filtered_output": str(filtered_output),
        "source_rows": int(len(frame)),
        "filtered_rows": int(len(filtered)),
        "criteria": filters,
    }


def _process_csv_pair(
    *,
    args,
    pair: dict,
    writer,
    csvfile,
    device,
    extractor,
    matcher,
    segmenter,
    seg_output_root,
    timing_writer=None,
    timing_file=None,
):
    id_left = str(pair["id_left"])
    id_right = str(pair["id_right"])
    left_path = resolve_local(args.images_dir, pair.get("filename_left", ""), id_left)
    right_path = resolve_local(args.images_dir, pair.get("filename_right", ""), id_right)

    if (not left_path or not right_path) and args.download:
        if not args.access_token:
            raise ValueError(
                "A Mapillary token is required with --download; "
                "set it in .env/config or pass --access-token."
            )
        logger.warning("Download not yet implemented; skipping %s vs %s", id_left, id_right)

    if not left_path or not right_path:
        logger.warning("Skipping %s vs %s: missing local files", id_left, id_right)
        return device, extractor, matcher, segmenter, False

    pair_kwargs = dict(
        id_left=id_left,
        date_left=str(pair.get("date_left", "N/A")),
        id_right=id_right,
        date_right=str(pair.get("date_right", "N/A")),
        left_path=left_path,
        right_path=right_path,
        writer=writer,
        csvfile=csvfile,
        panorama=args.panorama,
        cache=args.cache,
        yaw_step=args.yaw_step,
        crop_keep=args.crop_keep,
        crop_top_bias=args.crop_top_bias,
        scale_search=args.scale_search,
        scale_reproject=args.scale_reproject,
        include_image_metrics=args.image_metrics,
        panorama_perspective_preview=args.panorama_perspective_preview,
        perspective_output_dir=args.perspective_output_dir,
        perspective_yaws=args.perspective_yaws,
        perspective_pitch=args.perspective_pitch,
        perspective_fov=args.perspective_fov,
        perspective_size=args.perspective_size,
        perspective_align_horizon=args.perspective_align_horizon,
        perspective_keypoint_align=args.perspective_keypoint_align,
        panorama_semantic_rerank=args.panorama_semantic_rerank,
        semantic_rerank_radius=args.semantic_rerank_radius,
        semantic_rerank_step=args.semantic_rerank_step,
        ignore_sky_keypoints=args.ignore_sky_keypoints,
        sky_keypoint_source=args.sky_keypoint_source,
        sky_keypoint_boundary_px=args.sky_keypoint_boundary_px,
        panorama_fast_yaw=args.panorama_fast_yaw,
        extra_tail=[pair.get("index", "N/A")],
        seg_output_root=seg_output_root,
        seg_crop_top_frac=args.seg_crop_top_frac,
        seg_crop_bottom_frac=args.seg_crop_bottom_frac,
        profile_timing=args.profile_timing,
        save_debug_images=args.save_debug_images,
        segmentation_cache_dir=args.segmentation_cache_dir,
        segmentation_max_width=args.segmentation_max_width,
    )

    device, extractor, matcher, segmenter = _run_pair(
        pair_kwargs,
        device,
        extractor,
        matcher,
        segmenter,
        csvfile,
        timing_writer=timing_writer,
        timing_file=timing_file,
    )
    return device, extractor, matcher, segmenter, True


def _load_explicit_csv_pairs(df: pd.DataFrame, args) -> list[dict]:
    if args.indices and "index" not in df.columns:
        raise ValueError("--indices requires an 'index' column in pair CSV mode")
    if args.indices:
        wanted = set(map(str, args.indices))
        df = df[df["index"].astype(str).isin(wanted)].copy()

    pairs: list[dict] = []
    for row_number, row in enumerate(df.to_dict("records"), start=1):
        left_filename = _csv_value(row, PAIR_CSV_FILENAME_LEFT)
        right_filename = _csv_value(row, PAIR_CSV_FILENAME_RIGHT)
        if not left_filename or not right_filename:
            raise ValueError(
                "Pair CSV rows must include filename_left and filename_right "
                "(or left_filename/right_filename aliases)."
            )
        row_index = _csv_value(row, ("index",), str(row_number))
        pairs.append(
            {
                "id_left": _csv_value(row, PAIR_CSV_ID_LEFT, _id_from_filename(left_filename, f"row{row_number}_left")),
                "id_right": _csv_value(row, PAIR_CSV_ID_RIGHT, _id_from_filename(right_filename, f"row{row_number}_right")),
                "date_left": _csv_value(row, PAIR_CSV_DATE_LEFT, "N/A"),
                "date_right": _csv_value(row, PAIR_CSV_DATE_RIGHT, "N/A"),
                "filename_left": left_filename,
                "filename_right": right_filename,
                "index": row_index,
            }
        )
    return pairs


def _load_grouped_csv_pairs(df: pd.DataFrame, args) -> list[dict]:
    needed = {"index", "id", "filename"}
    if not needed.issubset(df.columns):
        raise ValueError(
            "CSV mode expects either explicit pair columns "
            "filename_left/filename_right, or legacy grouped columns "
            f"{needed}; got {list(df.columns)}"
        )

    logger.warning(
        "Using legacy grouped CSV mode. Prefer one-row-per-pair CSVs with "
        "filename_left and filename_right columns."
    )
    df["index"] = df["index"].astype(str)
    df["id"] = df["id"].astype(str)
    if "date" not in df.columns:
        df["date"] = "N/A"
    if args.indices:
        wanted = set(map(str, args.indices))
        df = df[df["index"].isin(wanted)].copy()

    pairs: list[dict] = []
    for idx_value, group in df.groupby("index"):
        rows = group.to_dict("records")
        if len(rows) < 2:
            continue
        pairs_done = 0
        for i, j in combinations(range(len(rows)), 2):
            L, R = rows[i], rows[j]
            pairs.append(
                {
                    "id_left": str(L["id"]),
                    "id_right": str(R["id"]),
                    "date_left": str(L.get("date", "N/A")),
                    "date_right": str(R.get("date", "N/A")),
                    "filename_left": L.get("filename", ""),
                    "filename_right": R.get("filename", ""),
                    "index": idx_value,
                }
            )
            pairs_done += 1
            if args.max_pairs_per_index and pairs_done >= args.max_pairs_per_index:
                break
    return pairs


def _load_csv_pairs(args) -> list[dict]:
    df = pd.read_csv(args.input_csv, on_bad_lines="skip")
    if _has_explicit_pair_columns(df):
        return _load_explicit_csv_pairs(df, args)
    return _load_grouped_csv_pairs(df, args)


def _init_pipeline(args):
    """Initialise device, feature models, and optional segmenter."""
    from utils.models import pick_device, init_models, init_segmenter

    device = pick_device(args.device)
    logger.info("Using device: %s", device)
    extractor, matcher = init_models(device)

    segmenter = None
    seg_output_root = None
    if args.segmentation:
        segmenter = init_segmenter(device, model_type=args.seg_dataset)
        if segmenter is not None:
            segmenter.ignore_temporary = args.ignore_temporary
        if not args.no_save_artifacts:
            seg_output_root = (
                args.seg_output_dir
                if args.seg_output_dir
                else os.path.join(os.path.dirname(args.output) or ".", "segmentations")
            )
            os.makedirs(seg_output_root, exist_ok=True)
        logger.info(
            "Segmentation enabled (dataset=%s, output=%s)",
            args.seg_dataset,
            seg_output_root if seg_output_root is not None else "disabled by --no-save-artifacts",
        )

    return device, extractor, matcher, segmenter, seg_output_root


def _write_timing_rows(timing_writer, timing_file, id_left: str, id_right: str, timing_rows):
    if not timing_writer or not timing_rows:
        return
    for stage, seconds in timing_rows:
        timing_writer.writerow([id_left, id_right, stage, round(float(seconds), 6)])
    timing_file.flush()
    os.fsync(timing_file.fileno())


def _run_pair(
    pair_kwargs: dict,
    device,
    extractor,
    matcher,
    segmenter,
    csvfile,
    timing_writer=None,
    timing_file=None,
):
    """
    Call process_one_pair; if CUDA OOM occurs, move all models to CPU and retry.
    Returns (device, extractor, matcher, segmenter) — may differ from input on OOM.
    """
    id_left = pair_kwargs["id_left"]
    id_right = pair_kwargs["id_right"]
    from utils.processing import process_one_pair

    try:
        result = process_one_pair(
            device=device, extractor=extractor, matcher=matcher,
            segmenter=segmenter, **pair_kwargs,
        )
        _write_timing_rows(
            timing_writer,
            timing_file,
            id_left,
            id_right,
            result.get("timing") if isinstance(result, dict) else None,
        )
        csvfile.flush()
        os.fsync(csvfile.fileno())
    except RuntimeError as e:
        if "CUDA out of memory" in str(e) and device.type == "cuda":
            logger.warning("CUDA OOM for %s vs %s — falling back to CPU", id_left, id_right)
            import torch
            from utils.models import init_models

            try:
                del extractor, matcher
                torch.cuda.empty_cache()
            except Exception:
                pass
            device = torch.device("cpu")
            extractor, matcher = init_models(device)
            if segmenter is not None:
                segmenter.model = segmenter.model.to(device)
                segmenter.device = device
            try:
                result = process_one_pair(
                    device=device, extractor=extractor, matcher=matcher,
                    segmenter=segmenter, **pair_kwargs,
                )
                _write_timing_rows(
                    timing_writer,
                    timing_file,
                    id_left,
                    id_right,
                    result.get("timing") if isinstance(result, dict) else None,
                )
                csvfile.flush()
                os.fsync(csvfile.fileno())
            except Exception as e2:
                tb = traceback.extract_tb(sys.exc_info()[2])[-1]
                logger.error(
                    "CPU retry failed for %s vs %s: %s (line %d in %s)",
                    id_left, id_right, e2, tb.lineno, tb.filename,
                )
        else:
            tb = traceback.extract_tb(sys.exc_info()[2])[-1]
            logger.error(
                "Skipping %s vs %s: %s (line %d in %s)",
                id_left, id_right, e, tb.lineno, tb.filename,
            )

    return device, extractor, matcher, segmenter


# ---------- CSV mode ----------

def _run_csv_mode(
    args,
    writer,
    csvfile,
    device,
    extractor,
    matcher,
    segmenter,
    seg_output_root,
    timing_writer=None,
    timing_file=None,
):
    pairs = _load_csv_pairs(args)
    if not os.path.exists(args.output) or os.stat(args.output).st_size == 0:
        writer.writerow(_csv_output_header(args))

    for pair in tqdm(pairs, total=len(pairs), desc="pairs"):
        device, extractor, matcher, segmenter, _ = _process_csv_pair(
            args=args,
            pair=pair,
            writer=writer,
            csvfile=csvfile,
            device=device,
            extractor=extractor,
            matcher=matcher,
            segmenter=segmenter,
            seg_output_root=seg_output_root,
            timing_writer=timing_writer,
            timing_file=timing_file,
        )

    ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    pd.DataFrame([], columns=_csv_output_header(args)).to_csv(
        os.path.join(args.backup_dir, f"backup_csv_{ts}.csv"),
        index=False,
    )

    return device, extractor, matcher, segmenter


def _run_csv_dry_run(args) -> None:
    """Summarize CSV-mode work without loading models, writing files, or downloading images."""
    pairs = _load_csv_pairs(args)
    local_pairs = 0
    missing_pairs = 0
    local_images: set[str] = set()
    all_images: set[str] = set()
    for pair in pairs:
        for side in ("left", "right"):
            filename = str(pair.get(f"filename_{side}", "")).strip()
            image_id = str(pair.get(f"id_{side}", "")).strip()
            if filename or image_id:
                all_images.add(filename or image_id)
        if args.images_dir:
            left_path = resolve_local(args.images_dir, pair.get("filename_left", ""), str(pair["id_left"]))
            right_path = resolve_local(args.images_dir, pair.get("filename_right", ""), str(pair["id_right"]))
            if left_path and right_path:
                local_pairs += 1
                local_images.update({left_path, right_path})
            else:
                if left_path:
                    local_images.add(left_path)
                if right_path:
                    local_images.add(right_path)
                missing_pairs += 1

    logger.info("images found: %s", _count(len(local_images) if args.images_dir else len(all_images)))
    logger.info("pairs found: %s", _count(len(pairs)))
    logger.info("after year filter: not applicable")
    logger.info("estimated output: %s", args.output or "not set")
    logger.info("will save images: %s", _yes_no(False))
    logger.info("will run segmentation: %s", _yes_no(bool(args.segmentation)))
    logger.info("Dry run: CSV pair count: %d", len(pairs))
    if args.images_dir:
        logger.info("Dry run: local image pairs found: %d", local_pairs)
        logger.info("Dry run: pairs missing one or both local images: %d", missing_pairs)
    else:
        logger.info("Dry run: --images-dir not set, so local image availability was not checked.")
    logger.info("Dry run: no models loaded, images downloaded, output CSVs written, or artifacts saved.")


# ---------- Area mode ----------

def _format_captured_range(df: pd.DataFrame) -> str:
    if "captured_at" not in df.columns or df.empty:
        return "unknown"
    captured = pd.to_numeric(df["captured_at"], errors="coerce").dropna()
    if captured.empty:
        return "unknown"
    dt = pd.to_datetime(captured.astype("int64"), unit="ms", utc=True)
    return f"{dt.min().date()} to {dt.max().date()}"


def _describe_time_filters(filters: list[str]) -> str:
    active = [f for f in (filters or []) if f != "any"]
    if not active:
        return "any capture time (no time-difference filter)"

    descriptions = {
        "year": "different years",
        "month": "different calendar months",
        "season": "different seasons",
        "same-season": "different years in the same season",
        "time": "one daytime and one nighttime image",
    }
    return " and ".join(descriptions.get(f, f) for f in active)


def _describe_year_groups(args) -> str:
    """Return a human-readable area-mode year-group filter description."""
    if not args.year_group_left or not args.year_group_right:
        return "not set"
    return (
        f"{sorted(args.year_group_left)} vs {sorted(args.year_group_right)} "
        "(one image from each group)"
    )


def _filter_df_to_year_groups(df: pd.DataFrame, args) -> pd.DataFrame:
    """Keep only metadata rows whose capture year is in either requested group."""
    if not args.year_group_left or not args.year_group_right:
        return df

    wanted_years = set(args.year_group_left) | set(args.year_group_right)
    captured = pd.to_numeric(df["captured_at"], errors="coerce")
    years = pd.to_datetime(captured, unit="ms", utc=True, errors="coerce").dt.year
    filtered = df[years.isin(wanted_years)].copy()
    counts = years[years.isin(wanted_years)].value_counts().sort_index()
    logger.info(
        "Filtered metadata to requested years: %d/%d images kept; counts=%s",
        len(filtered),
        len(df),
        counts.to_dict(),
    )
    return filtered


def _captured_year(meta: dict) -> int | None:
    """Return UTC capture year from Mapillary Unix-ms metadata, or None."""
    try:
        value = meta.get("captured_at")
        if value is None:
            return None
        return datetime.fromtimestamp(int(value) / 1000, tz=timezone.utc).year
    except (TypeError, ValueError, OSError):
        return None


def _year_pair_key(left_meta: dict, right_meta: dict) -> tuple[int, int] | None:
    """Return normalized (older_year, newer_year) key for a pair, if available."""
    left_year = _captured_year(left_meta)
    right_year = _captured_year(right_meta)
    if left_year is None or right_year is None:
        return None
    return tuple(sorted((left_year, right_year)))


def _image_tile_key(meta: dict) -> tuple[int, int, int] | None:
    """Return the source Mapillary tile key for an image, if metadata has it."""
    try:
        return (int(meta["tile_z"]), int(meta["tile_x"]), int(meta["tile_y"]))
    except (KeyError, TypeError, ValueError):
        return None


def _tile_pair_key(left_meta: dict, right_meta: dict) -> tuple | None:
    """Return a normalized tile-combination key for a pair, if available."""
    left_tile = _image_tile_key(left_meta)
    right_tile = _image_tile_key(right_meta)
    if left_tile is None or right_tile is None:
        return None
    return tuple(sorted((left_tile, right_tile)))


def _order_area_pairs(pairs: list[tuple[dict, dict]], args) -> list[tuple[dict, dict]]:
    """
    Randomize area-mode processing order.

    Process pairs round-robin by source tile-combination. With --time-filter
    year, include the year-combination in the bucket key too, so a long
    interrupted run samples broadly across both space and time.
    """
    rng = random.Random(args.pair_random_seed)
    balance_years = "year" in (args.time_filter or [])
    buckets: dict[tuple, deque] = defaultdict(deque)
    for pair in pairs:
        year_key = _year_pair_key(pair[0], pair[1]) if balance_years else None
        tile_key = _tile_pair_key(pair[0], pair[1])
        buckets[(year_key, tile_key)].append(pair)

    for bucket in buckets.values():
        bucket_items = list(bucket)
        rng.shuffle(bucket_items)
        bucket.clear()
        bucket.extend(bucket_items)

    known_year_buckets = len({key[0] for key in buckets if key[0] is not None})
    known_tile_buckets = len({key[1] for key in buckets if key[1] is not None})
    unknown_tile_count = sum(len(bucket) for key, bucket in buckets.items() if key[1] is None)
    keys = list(buckets.keys())
    rng.shuffle(keys)
    ordered: list[tuple[dict, dict]] = []
    while keys:
        next_keys = []
        for key in keys:
            bucket = buckets[key]
            if bucket:
                ordered.append(bucket.popleft())
            if bucket:
                next_keys.append(key)
        rng.shuffle(next_keys)
        keys = next_keys

    logger.info(
        "Balanced randomized %d area pairs across %d tile combinations%s%s.",
        len(ordered),
        known_tile_buckets,
        f" and {known_year_buckets} year combinations" if balance_years else "",
        f" plus {unknown_tile_count} unknown-tile pairs" if unknown_tile_count else "",
    )
    return ordered


def _delete_temp_pair_images(paths: list[str | None], temp_root: str | None) -> None:
    """Delete downloaded pair images when using temporary Mapillary storage."""
    if not temp_root:
        return
    temp_root_abs = os.path.abspath(temp_root)
    for path in set(p for p in paths if p):
        try:
            path_abs = os.path.abspath(path)
            if os.path.commonpath([temp_root_abs, path_abs]) != temp_root_abs:
                logger.warning("Refusing to delete path outside temp image store: %s", path)
                continue
            if os.path.isfile(path_abs):
                os.remove(path_abs)
        except Exception as e:
            logger.warning("Failed to delete temporary image %s: %s", path, e)


def _log_area_image_summary(df: pd.DataFrame, requested_image_type: str, time_filters: list[str]):
    pano_count = int(df["is_pano"].fillna(False).astype(bool).sum()) if "is_pano" in df else 0
    flat_count = int(len(df) - pano_count)
    logger.info("Requested image type: %s", requested_image_type)
    logger.info(
        "Found image types: %d panorama, %d flat, %d total",
        pano_count,
        flat_count,
        len(df),
    )
    logger.info("Found capture date range: %s", _format_captured_range(df))
    logger.info("Pair time filter: %s", _describe_time_filters(time_filters))


def _run_area_mode(
    args,
    writer,
    csvfile,
    device,
    extractor,
    matcher,
    segmenter,
    seg_output_root,
    timing_writer=None,
    timing_file=None,
):
    from utils.area_pairing import fetch_images_in_area, pair_by_proximity, compute_pair_metadata

    # 1. Query Mapillary metadata
    logger.info(
        "Looking for image_type=%s with time_filter=%s",
        args.image_type,
        _describe_time_filters(args.time_filter),
    )
    df = fetch_images_in_area(
        args.area_wkt,
        args.access_token,
        image_type=args.image_type,
        request_delay=args.mapillary_request_delay,
        max_retries=args.mapillary_max_retries,
        retry_delay=args.mapillary_retry_delay,
    )
    if df.empty:
        logger.warning("No images found in the specified area — nothing to process.")
        return device, extractor, matcher, segmenter
    _log_area_image_summary(df, args.image_type, args.time_filter)
    logger.info("Pair year-group filter: %s", _describe_year_groups(args))
    df = _filter_df_to_year_groups(df, args)
    if df.empty:
        logger.warning("No images remain after applying year-group metadata filter.")
        return device, extractor, matcher, segmenter

    # 2. Pair by proximity + optional angle/time filters
    pairs = pair_by_proximity(
        df,
        max_distance_m=args.max_distance,
        max_angle_diff=args.max_angle_diff,
        time_filters=args.time_filter,
        year_group_left=args.year_group_left,
        year_group_right=args.year_group_right,
    )
    if not pairs:
        logger.warning("Year-group filter used for pairing: %s", _describe_year_groups(args))
        logger.warning(
            "No pairs found within %.1f m%s.",
            args.max_distance,
            f" and {args.max_angle_diff}° angle diff" if args.max_angle_diff else "",
        )
        return device, extractor, matcher, segmenter
    pairs = _order_area_pairs(pairs, args)

    # 3. Write CSV header
    if not os.path.exists(args.output) or os.stat(args.output).st_size == 0:
        if args.panorama:
            header = CSV_HEADER_PANORAMA_AREA
        elif args.scale_search:
            header = CSV_HEADER_SCALE_AREA
        else:
            header = CSV_HEADER_AREA
        writer.writerow(with_optional_image_metrics(header, args.image_metrics))

    # 4. Load already-completed pairs so we can resume interrupted runs
    done = completed_pairs_set(args.output)
    if done:
        logger.info("Resuming: %d pairs already in output, skipping them.", len(done))

    # 5. Download images and process each pair
    temp_root = os.path.abspath(args.temp_dir)
    if not args.save_mapillary_images:
        ensure_dir(temp_root)
    image_store_ctx = (
        tempfile.TemporaryDirectory(
            prefix="pairwise_mapillary_images_",
            dir=temp_root,
        )
        if not args.save_mapillary_images
        else None
    )
    image_store_dir = args.images_dir
    if image_store_ctx is not None:
        image_store_dir = image_store_ctx.name
        logger.info(
            "Downloaded Mapillary images will use temporary storage: %s",
            image_store_dir,
        )
    else:
        ensure_dir(args.images_dir)

    with requests.Session() as session:
        try:
            for pair_idx, (left_meta, right_meta) in enumerate(
                tqdm(pairs, desc="pairs")
            ):
                id_left = left_meta["id"]
                id_right = right_meta["id"]
                left_path = None
                right_path = None

                if (id_left, id_right) in done or (id_right, id_left) in done:
                    continue

                try:
                    left_path = fetch_and_cache_image(
                        id_left, args.access_token, image_store_dir, session,
                        size=args.image_size,
                        request_delay=args.mapillary_request_delay,
                        max_retries=args.mapillary_max_retries,
                        retry_delay=args.mapillary_retry_delay,
                    )
                    right_path = fetch_and_cache_image(
                        id_right, args.access_token, image_store_dir, session,
                        size=args.image_size,
                        request_delay=args.mapillary_request_delay,
                        max_retries=args.mapillary_max_retries,
                        retry_delay=args.mapillary_retry_delay,
                    )

                    if not left_path or not right_path:
                        logger.warning(
                            "Skipping pair %d (%s vs %s): download failed",
                            pair_idx, id_left, id_right,
                        )
                        continue

                    meta = compute_pair_metadata(left_meta, right_meta)

                    # Spatial metadata is appended after the standard "index" column
                    extra_tail = [
                        pair_idx,
                        meta["dist_m"],
                        meta["angle_diff_deg"] if meta["angle_diff_deg"] is not None else "NA",
                        left_meta.get("is_pano", False),
                        right_meta.get("is_pano", False),
                        round(left_meta["lat"], 7),
                        round(left_meta["lon"], 7),
                        round(right_meta["lat"], 7),
                        round(right_meta["lon"], 7),
                    ]

                    pair_kwargs = dict(
                        id_left=id_left,
                        date_left=str(left_meta.get("captured_at", "N/A")),
                        id_right=id_right,
                        date_right=str(right_meta.get("captured_at", "N/A")),
                        left_path=left_path,
                        right_path=right_path,
                        writer=writer,
                        csvfile=csvfile,
                        panorama=args.panorama,
                        cache=args.cache,
                        yaw_step=args.yaw_step,
                        crop_keep=args.crop_keep,
                        crop_top_bias=args.crop_top_bias,
                        scale_search=args.scale_search,
                        scale_reproject=args.scale_reproject,
                        include_image_metrics=args.image_metrics,
                        panorama_perspective_preview=args.panorama_perspective_preview,
                        perspective_output_dir=args.perspective_output_dir,
                        perspective_yaws=args.perspective_yaws,
                        perspective_pitch=args.perspective_pitch,
                        perspective_fov=args.perspective_fov,
                        perspective_size=args.perspective_size,
                        perspective_align_horizon=args.perspective_align_horizon,
                        perspective_keypoint_align=args.perspective_keypoint_align,
                        panorama_semantic_rerank=args.panorama_semantic_rerank,
                        semantic_rerank_radius=args.semantic_rerank_radius,
                        semantic_rerank_step=args.semantic_rerank_step,
                        ignore_sky_keypoints=args.ignore_sky_keypoints,
                        sky_keypoint_source=args.sky_keypoint_source,
                        sky_keypoint_boundary_px=args.sky_keypoint_boundary_px,
                        panorama_fast_yaw=args.panorama_fast_yaw,
                        extra_tail=extra_tail,
                        seg_output_root=seg_output_root,
                        seg_crop_top_frac=args.seg_crop_top_frac,
                        seg_crop_bottom_frac=args.seg_crop_bottom_frac,
                        profile_timing=args.profile_timing,
                        save_debug_images=args.save_debug_images,
                        segmentation_cache_dir=args.segmentation_cache_dir,
                        segmentation_max_width=args.segmentation_max_width,
                    )

                    device, extractor, matcher, segmenter = _run_pair(
                        pair_kwargs,
                        device,
                        extractor,
                        matcher,
                        segmenter,
                        csvfile,
                        timing_writer=timing_writer,
                        timing_file=timing_file,
                    )
                finally:
                    if image_store_ctx is not None:
                        _delete_temp_pair_images([left_path, right_path], image_store_dir)
        finally:
            if image_store_ctx is not None:
                image_store_ctx.cleanup()

    return device, extractor, matcher, segmenter


def _run_area_dry_run(args) -> None:
    """Summarize area-mode work without downloading images, loading models, or writing output."""
    from utils.area_pairing import fetch_images_in_area, pair_by_proximity

    logger.info(
        "Dry run: querying image_type=%s with time_filter=%s",
        args.image_type,
        _describe_time_filters(args.time_filter),
    )
    df = fetch_images_in_area(
        args.area_wkt,
        args.access_token,
        image_type=args.image_type,
        request_delay=args.mapillary_request_delay,
        max_retries=args.mapillary_max_retries,
        retry_delay=args.mapillary_retry_delay,
    )
    if df.empty:
        logger.warning("Dry run: no images found in the specified area.")
        return

    _log_area_image_summary(df, args.image_type, args.time_filter)
    logger.info("Dry run: pair year-group filter: %s", _describe_year_groups(args))
    filtered = _filter_df_to_year_groups(df, args)
    if filtered.empty:
        logger.warning("Dry run: no images remain after applying year-group metadata filter.")
        return

    base_pairs = pair_by_proximity(
        filtered,
        max_distance_m=args.max_distance,
        max_angle_diff=args.max_angle_diff,
        time_filters=["any"],
    )
    pairs = pair_by_proximity(
        filtered,
        max_distance_m=args.max_distance,
        max_angle_diff=args.max_angle_diff,
        time_filters=args.time_filter,
        year_group_left=args.year_group_left,
        year_group_right=args.year_group_right,
    )
    done = completed_pairs_set(args.output) if args.output else set()
    pending = [
        pair
        for pair in pairs
        if (str(pair[0]["id"]), str(pair[1]["id"])) not in done
        and (str(pair[1]["id"]), str(pair[0]["id"])) not in done
    ]

    logger.info("images found: %s", _count(len(df)))
    logger.info("pairs found: %s", _count(len(base_pairs)))
    logger.info("%s: %s", _dry_run_time_filter_label(args), _count(len(pairs)))
    logger.info("estimated output: %s", args.output or "not set")
    logger.info("will save images: %s", _yes_no(_will_save_images(args)))
    logger.info("will run segmentation: %s", _yes_no(bool(args.segmentation)))
    logger.info("Dry run: metadata rows after filters: %d", len(filtered))
    logger.info("Dry run: candidate pair count: %d", len(pairs))
    if args.output:
        logger.info("Dry run: pairs already present in output CSV: %d", len(done))
        logger.info("Dry run: pairs remaining to process: %d", len(pending))
    logger.info("Dry run: no images downloaded, models loaded, output CSVs written, or artifacts saved.")


# ---------- CLI ----------

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_config_argument(ap)

    # ---- Input mode (mutually exclusive) ----
    mode = ap.add_argument_group("Input mode (choose one)")
    mode.add_argument(
        "--input-csv",
        help=(
            "Pair CSV with filename_left and filename_right columns "
            "(id/date/index optional). [CSV mode]"
        ),
    )
    mode.add_argument(
        "--area-wkt",
        help='WKT polygon of the area to query on Mapillary, e.g. "POLYGON ((lon lat, ...))" [Area mode]',
    )

    # ---- Area-mode options ----
    area = ap.add_argument_group("Area mode options")
    area.add_argument(
        "--max-distance", type=float, default=2.5,
        help="Max distance in metres between paired images (default: 2.5).",
    )
    area.add_argument(
        "--max-angle-diff", type=float, default=None,
        help=(
            "Max compass-angle difference in degrees between paired images "
            "(root config.toml sets 30.0 by default; omit in custom configs for no filter)."
        ),
    )
    area.add_argument(
        "--image-type", choices=["all", "panorama", "flat"], default="all",
        help="Which image types to include: all, panorama-only, or flat-only (default: all).",
    )
    area.add_argument(
        "--image-size", choices=["256", "1024", "2048", "original"], default="1024",
        help="Mapillary thumbnail size to download (default: 1024).",
    )
    area.add_argument(
        "--mapillary-request-delay",
        type=float,
        default=0.1,
        help=(
            "Seconds to wait before each Mapillary metadata/image request "
            "(default: 0.1)."
        ),
    )
    area.add_argument(
        "--mapillary-max-retries",
        type=int,
        default=5,
        help="Number of retries after Mapillary returns 429 Too Many Requests (default: 5).",
    )
    area.add_argument(
        "--mapillary-retry-delay",
        type=float,
        default=60.0,
        help="Seconds to wait before each retry after 429 Too Many Requests (default: 60).",
    )
    area.add_argument(
        "--time-filter",
        nargs="+",
        choices=["any", "year", "month", "season", "same-season", "time"],
        default=["any"],
        metavar="FILTER",
        help=(
            "Require a time difference between paired images. "
            "Choices: any (no constraint), year (different years), "
            "month (different months), season (different seasons: spring=Mar-May, "
            "summer=Jun-Aug, autumn=Sep-Nov, winter=Dec-Feb), "
            "same-season (different years, same season), "
            "time (one day / one night, requires: pip install \".[area]\"). "
            "Multiple values are combined with AND logic, e.g. --time-filter season year."
        ),
    )
    area.add_argument(
        "--year-group-left",
        nargs="+",
        type=int,
        default=None,
        metavar="YEAR",
        help=(
            "Area mode: only keep pairs where one image is from this year set "
            "and the other is from --year-group-right."
        ),
    )
    area.add_argument(
        "--year-group-right",
        nargs="+",
        type=int,
        default=None,
        metavar="YEAR",
        help=(
            "Area mode: comparison year set for --year-group-left, e.g. "
            "--year-group-left 2016 2017 2018 --year-group-right 2024 2025 2026."
        ),
    )
    area.add_argument(
        "--pair-random-seed",
        type=int,
        default=None,
        help=(
            "Random seed for area-mode pair order. By default the order changes "
            "between runs; set this for reproducible randomized processing."
        ),
    )

    # ---- Common ----
    ap.add_argument("--images-dir",
                    help="Local image directory (source for CSV mode; download cache for area mode).")
    ap.add_argument("--output", help="Output CSV for metrics.")
    ap.add_argument(
        "--manifest-path",
        default="outputs/run_manifest.json",
        help=(
            "JSON run manifest path written after successful processing "
            "(default: outputs/run_manifest.json)."
        ),
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Validate inputs and report metadata/pair counts without loading models, "
            "downloading images, or writing outputs."
        ),
    )
    ap.add_argument("--backup-dir", default="backup",
                    help="Directory for periodic heartbeat backups (default: backup).")
    ap.add_argument(
        "--temp-dir",
        default=".",
        help=(
            "Directory where temporary run folders are created when Mapillary "
            "images are not kept persistently (default: current working directory)."
        ),
    )
    ap.add_argument("--device", choices=["cuda", "mps", "cpu"],
                    help="Force compute device (default: auto-detect best available).")
    ap.add_argument("--access-token",
                    help=(
                        "Mapillary API token. Prefer access_token_env in config.toml "
                        "with the real token stored in local .env."
                    ))
    ap.add_argument("--download", action="store_true",
                    help="(CSV mode) Fetch missing images from Mapillary.")
    ap.add_argument("--cache", action="store_true", default=False)
    ap.add_argument("--indices", nargs="*",
                    help="(CSV mode) Only process these index values.")
    ap.add_argument("--max-pairs-per-index", type=int,
                    help="(CSV mode) Test throttle: max pairs per index group.")

    # ---- Filtered result export ----
    filtering = ap.add_argument_group("Filtered result export")
    filtering.add_argument(
        "--filtered-output",
        default=None,
        help=(
            "Optional CSV path for rows that pass the selected result filters. "
            "The full unfiltered output CSV is always kept."
        ),
    )
    filtering.add_argument(
        "--filter-match-ratio-min",
        type=float,
        default=None,
        help="Keep rows with lightglue_match_ratio >= this value.",
    )
    filtering.add_argument(
        "--filter-avg-distance-max",
        type=float,
        default=None,
        help="Keep rows with lightglue_avg_distance <= this value.",
    )
    filtering.add_argument(
        "--filter-keypoint-coverage-min",
        type=float,
        default=None,
        help="Keep rows with lightglue_keypoint_coverage_min >= this value.",
    )
    filtering.add_argument(
        "--filter-keypoint-hull-iou-min",
        type=float,
        default=None,
        help="Keep rows with lightglue_keypoint_hull_iou >= this value.",
    )
    filtering.add_argument(
        "--filter-road-iou-min",
        type=float,
        default=None,
        help="Keep rows with seg_overlap_road_iou >= this value.",
    )
    filtering.add_argument(
        "--filter-mean-iou-min",
        type=float,
        default=None,
        help="Keep rows with seg_overlap_mean_iou >= this value.",
    )
    ap.add_argument(
        "--profile-timing",
        action="store_true",
        help=(
            "Measure per-pair processing stage durations and write them to a "
            "*_timing.csv file next to --output."
        ),
    )
    ap.add_argument(
        "--save-debug-images",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Save match/debug PNGs such as pano_debug and match_viz outputs "
            "(default: enabled). Use --no-save-debug-images for faster runs."
        ),
    )
    ap.add_argument(
        "--no-save-artifacts",
        action="store_true",
        help=(
            "Do not save optional image artifacts that are not needed for processing, "
            "including debug visualizations, panorama/scale debug images, perspective "
            "preview images, segmentation mask PNGs, and persistent downloaded "
            "Mapillary image files. CSV outputs are still written."
        ),
    )
    ap.add_argument(
        "--save-mapillary-images",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Keep downloaded Mapillary images in --images-dir for later reuse "
            "(default: enabled). Use --no-save-mapillary-images to download them "
            "to a temporary run directory instead."
        ),
    )

    # ---- Panorama ----
    pano = ap.add_argument_group("Panorama")
    pano.add_argument("--panorama", action="store_true",
                      help="Assume equirectangular panoramas; search for best horizontal yaw alignment.")
    pano.add_argument("--yaw-step", type=int, default=10,
                      help="Coarse yaw-search step in degrees (default: 10).")
    pano.add_argument("--crop-keep", type=float, default=0.55,
                      help=(
                          "Vertical keep ratio for panorama yaw search and aligned "
                          "analysis metrics (default: 0.55)."
                      ))
    pano.add_argument("--crop-top-bias", type=float, default=0.0,
                      help="Vertical crop bias for panoramas, -1..1 (default: 0).")
    pano.add_argument(
        "--panorama-perspective-preview",
        action="store_true",
        help=(
            "Export rectilinear perspective preview images for panorama pairs before "
            "matching. This does not change matching yet."
        ),
    )
    pano.add_argument(
        "--perspective-output-dir",
        default=None,
        help=(
            "Directory for --panorama-perspective-preview outputs "
            "(default: panorama_perspective_views next to the output CSV)."
        ),
    )
    pano.add_argument(
        "--perspective-yaws",
        nargs="*",
        type=float,
        default=[0.0, 60.0, 120.0, 180.0, 240.0, 300.0],
        metavar="DEG",
        help="Yaw directions to export in degrees (default: 0 60 120 180 240 300).",
    )
    pano.add_argument(
        "--perspective-pitch",
        type=float,
        default=0.0,
        help="Pitch direction for exported perspective views in degrees (default: 0).",
    )
    pano.add_argument(
        "--perspective-fov",
        type=float,
        default=90.0,
        help="Horizontal FOV for exported perspective views in degrees (default: 90).",
    )
    pano.add_argument(
        "--perspective-size",
        type=int,
        default=768,
        help="Square output size in pixels for perspective previews (default: 768).",
    )
    pano.add_argument(
        "--perspective-align-horizon",
        action="store_true",
        help=(
            "For preview exports, try to rotate/translate each perspective view so "
            "the detected horizon is level and vertically centred."
        ),
    )
    pano.add_argument(
        "--perspective-keypoint-align",
        action="store_true",
        help=(
            "For preview exports, match each left/right perspective view with "
            "LightGlue and save keypoint-aligned right views for debugging."
        ),
    )
    pano.add_argument(
        "--panorama-semantic-rerank",
        action="store_true",
        help=(
            "After keypoint yaw alignment, rerank nearby yaw candidates using "
            "semantic segmentation masks. Requires --panorama and --segmentation."
        ),
    )
    pano.add_argument(
        "--panorama-fast-yaw",
        action="store_true",
        help=(
            "Fast panorama alignment: estimate yaw from one keypoint matching pass "
            "and shift once, skipping coarse/fine yaw search."
        ),
    )
    pano.add_argument(
        "--semantic-rerank-radius",
        type=int,
        default=8,
        help="Yaw radius in degrees for --panorama-semantic-rerank (default: 8).",
    )
    pano.add_argument(
        "--semantic-rerank-step",
        type=int,
        default=2,
        help="Yaw step in degrees for --panorama-semantic-rerank (default: 2).",
    )

    # ---- Scale / FOV search ----
    scale = ap.add_argument_group("FOV alignment (mutually exclusive with --panorama)")
    scale.add_argument(
        "--scale-search", action="store_true",
        help=(
            "Align same-spot images with different fields of view by cropping both "
            "images to the robust area covered by matched keypoints, ignoring isolated "
            "outlier matches. Outputs fov_crop_fraction = the smaller retained-area "
            "fraction of the two crops, plus per-image retained/cropped fractions and "
            "full-image orig_* metrics."
        ),
    )
    scale.add_argument(
        "--scale-reproject", action="store_true",
        help=(
            "With --scale-search, estimate robust homographies from LightGlue matches, "
            "warp the image whose homography is less geometrically distorted, then "
            "crop both images to the shared inlier area."
        ),
    )

    # ---- Image quality metrics ----
    metrics = ap.add_argument_group("Image metrics")
    metrics.add_argument(
        "--image-metrics",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Include per-image brightness, contrast, sharpness, noise, dark-fraction, "
            "and horizon-angle metrics in the output CSV (default: enabled). "
            "Use --no-image-metrics to skip these columns."
        ),
    )

    # ---- Segmentation ----
    seg = ap.add_argument_group("Segmentation")
    seg.add_argument("--segmentation", action="store_true",
                     help="Run OneFormer semantic segmentation and log per-class IoU.")
    seg.add_argument("--seg-dataset", choices=["cityscapes", "ade20k"], default="cityscapes",
                     help="Segmentation model/dataset (default: cityscapes).")
    seg.add_argument("--seg-output-dir", default=None,
                     help="Directory for saved segmentation mask images.")
    seg.add_argument(
        "--segmentation-cache-dir",
        default=None,
        help=(
            "Directory for cached semantic label masks as .npy files "
            "(default: segmentation_cache next to the output CSV when segmentation is enabled)."
        ),
    )
    seg.add_argument(
        "--segmentation-max-width",
        type=int,
        default=1024,
        help=(
            "Resize images to this max width before semantic segmentation "
            "(default: 1024; use 0 to segment at full resolution)."
        ),
    )
    seg.add_argument("--ignore-temporary", action="store_true",
                     help="Exclude dynamic classes (vehicles, people) from IoU computation.")
    seg.add_argument(
        "--ignore-sky-keypoints",
        action="store_true",
        help=(
            "Exclude LightGlue keypoints that are deep inside ignored regions. "
            "With segmentation masks, this ignores sky plus temporary classes "
            "such as vehicles and people. Keypoints are kept if they are within "
            "--sky-keypoint-boundary-px pixels of a non-ignored class."
        ),
    )
    seg.add_argument(
        "--sky-keypoint-source",
        choices=["segmentation", "heuristic"],
        default="segmentation",
        help=(
            "Sky mask source for --ignore-sky-keypoints: segmentation is more "
            "accurate but slow; heuristic is fast and does not require --segmentation "
            "(default: segmentation)."
        ),
    )
    seg.add_argument(
        "--sky-keypoint-boundary-px",
        type=int,
        default=10,
        help=(
            "Boundary distance in pixels for --ignore-sky-keypoints (default: 10). "
            "Ignored-region keypoints within this distance of non-ignored classes "
            "are kept."
        ),
    )
    seg.add_argument("--seg-crop-top-frac", type=float, default=0.0,
                     help="Fraction of image height to crop from top before mIoU (default: 0).")
    seg.add_argument("--seg-crop-bottom-frac", type=float, default=0.0,
                     help="Fraction of image height to crop from bottom before mIoU (default: 0).")

    args = parse_args_with_config(
        ap,
        default_config_path=_default_config_path(),
        script_sections=("process", "process_pairs"),
    )
    started_at = datetime.now(timezone.utc)

    # Validate mode selection
    if not args.input_csv and not args.area_wkt:
        ap.error("One of --input-csv or --area-wkt is required.")
    if args.input_csv and args.area_wkt:
        ap.error("--input-csv and --area-wkt are mutually exclusive.")
    if args.area_wkt and not args.access_token:
        ap.error("A Mapillary token is required with --area-wkt; set it in .env/config or pass --access-token.")
    if not args.dry_run:
        if not args.images_dir:
            ap.error("--images-dir is required unless --dry-run is used.")
        if not args.output:
            ap.error("--output is required unless --dry-run is used.")
    if args.panorama and args.scale_search:
        ap.error("--panorama and --scale-search are mutually exclusive.")
    if args.scale_reproject and not args.scale_search:
        ap.error("--scale-reproject requires --scale-search.")
    if args.panorama_fast_yaw and not args.panorama:
        ap.error("--panorama-fast-yaw requires --panorama.")
    if args.crop_keep <= 0:
        ap.error("--crop-keep must be greater than 0.")
    if args.panorama_perspective_preview and not args.panorama:
        ap.error("--panorama-perspective-preview requires --panorama.")
    if args.perspective_keypoint_align and not args.panorama_perspective_preview:
        ap.error("--perspective-keypoint-align requires --panorama-perspective-preview.")
    if args.panorama_semantic_rerank and not args.panorama:
        ap.error("--panorama-semantic-rerank requires --panorama.")
    if args.panorama_semantic_rerank and not args.segmentation:
        ap.error("--panorama-semantic-rerank requires --segmentation.")
    if (
        args.ignore_sky_keypoints
        and args.sky_keypoint_source == "segmentation"
        and not args.segmentation
    ):
        ap.error("--ignore-sky-keypoints with --sky-keypoint-source segmentation requires --segmentation.")
    if args.sky_keypoint_boundary_px < 0:
        ap.error("--sky-keypoint-boundary-px must be >= 0.")
    if args.perspective_size <= 0:
        ap.error("--perspective-size must be greater than 0.")
    if not (0 < args.perspective_fov < 180):
        ap.error("--perspective-fov must be between 0 and 180 degrees.")
    if not args.perspective_yaws:
        ap.error("--perspective-yaws must include at least one yaw direction.")
    if args.semantic_rerank_radius < 0:
        ap.error("--semantic-rerank-radius must be >= 0.")
    if args.semantic_rerank_step <= 0:
        ap.error("--semantic-rerank-step must be greater than 0.")
    if args.mapillary_request_delay < 0:
        ap.error("--mapillary-request-delay must be >= 0.")
    if args.mapillary_max_retries < 0:
        ap.error("--mapillary-max-retries must be >= 0.")
    if args.mapillary_retry_delay < 0:
        ap.error("--mapillary-retry-delay must be >= 0.")
    if args.filter_match_ratio_min is not None and not (0.0 <= args.filter_match_ratio_min <= 1.0):
        ap.error("--filter-match-ratio-min must be between 0 and 1.")
    if args.filter_road_iou_min is not None and not (0.0 <= args.filter_road_iou_min <= 1.0):
        ap.error("--filter-road-iou-min must be between 0 and 1.")
    if args.filter_mean_iou_min is not None and not (0.0 <= args.filter_mean_iou_min <= 1.0):
        ap.error("--filter-mean-iou-min must be between 0 and 1.")
    if args.filter_keypoint_coverage_min is not None and not (
        0.0 <= args.filter_keypoint_coverage_min <= 1.0
    ):
        ap.error("--filter-keypoint-coverage-min must be between 0 and 1.")
    if args.filter_keypoint_hull_iou_min is not None and not (
        0.0 <= args.filter_keypoint_hull_iou_min <= 1.0
    ):
        ap.error("--filter-keypoint-hull-iou-min must be between 0 and 1.")
    if args.filter_avg_distance_max is not None and args.filter_avg_distance_max < 0:
        ap.error("--filter-avg-distance-max must be >= 0.")
    if _active_result_filters(args) and not args.filtered_output and not args.dry_run:
        ap.error("--filtered-output is required when result filter thresholds are set.")
    if bool(args.year_group_left) != bool(args.year_group_right):
        ap.error("--year-group-left and --year-group-right must be used together.")
    if args.year_group_left and args.year_group_right:
        args.time_filter = list(dict.fromkeys([*(args.time_filter or []), "year"]))
    if args.no_save_artifacts:
        args.save_debug_images = False
        args.panorama_perspective_preview = False
        args.save_mapillary_images = False
        args.segmentation_cache_dir = None
    if (
        args.segmentation
        and args.segmentation_cache_dir is None
        and not args.no_save_artifacts
        and args.output
    ):
        args.segmentation_cache_dir = os.path.join(
            os.path.dirname(args.output) or ".",
            "segmentation_cache",
        )
    if args.segmentation_max_width < 0:
        ap.error("--segmentation-max-width must be >= 0.")
    if args.segmentation_max_width == 0:
        args.segmentation_max_width = None

    if args.dry_run:
        if args.area_wkt:
            _run_area_dry_run(args)
        else:
            _run_csv_dry_run(args)
        return

    ensure_dir(args.images_dir)
    ensure_dir(args.backup_dir)
    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    logger.info("Writing results to: %s", os.path.abspath(args.output))

    device, extractor, matcher, segmenter, seg_output_root = _init_pipeline(args)

    timing_file = None
    timing_writer = None
    timing_path = None
    if args.profile_timing:
        output_root, output_ext = os.path.splitext(args.output)
        timing_path = f"{output_root}_timing{output_ext or '.csv'}"
        timing_file = open(timing_path, "a", newline="")
        timing_writer = csv.writer(timing_file)
        if not os.path.exists(timing_path) or os.stat(timing_path).st_size == 0:
            timing_writer.writerow(TIMING_HEADER)
        logger.info("Writing timing profile to: %s", os.path.abspath(timing_path))

    try:
        with open(args.output, "a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            if args.area_wkt:
                _run_area_mode(
                    args, writer, csvfile,
                    device, extractor, matcher, segmenter, seg_output_root,
                    timing_writer=timing_writer,
                    timing_file=timing_file,
                )
            else:
                _run_csv_mode(
                    args, writer, csvfile,
                    device, extractor, matcher, segmenter, seg_output_root,
                    timing_writer=timing_writer,
                    timing_file=timing_file,
                )
        filter_info = _write_filtered_results(args)
        manifest_path = write_run_manifest(
            path=args.manifest_path,
            project_root=PROJECT_ROOT,
            script_name="process_pairs.py",
            args=args,
            started_at=started_at,
            status="completed",
            extra={
                "input_mode": "area" if args.area_wkt else "csv",
                "output": args.output,
                "filtered_results": filter_info,
                "timing_path": timing_path,
                "segmentation_output_dir": seg_output_root,
            },
        )
        logger.info("Wrote run manifest to: %s", manifest_path)
    finally:
        if timing_file is not None:
            timing_file.close()


if __name__ == "__main__":
    main()
