from __future__ import annotations

import sys
import argparse
import json
import os
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from random_semantic_sampler_common import (  # noqa: E402
    _add_time_columns,
    _build_sample_groups,
    crop_vertical_middle,
    parse_captured_year,
    safe_label,
)
from process_pairs import _load_explicit_csv_pairs, _write_filtered_results  # noqa: E402
from utils.area_pairing import compute_pair_metadata, pair_by_proximity  # noqa: E402
from utils.config import add_config_argument, parse_args_with_config  # noqa: E402
from utils.io_utils import CSV_HEADER, IMAGE_METRIC_COLUMNS, with_optional_image_metrics  # noqa: E402
from utils.metrics import compute_image_quality_metrics  # noqa: E402
from utils.processing import _matched_keypoint_coverage, _matched_keypoint_hull_iou  # noqa: E402
from utils.run_manifest import write_run_manifest  # noqa: E402


class CoreHelperTests(unittest.TestCase):
    def test_parse_captured_year_from_mapillary_timestamp(self) -> None:
        self.assertEqual(parse_captured_year(1_704_067_200_000), 2024)
        self.assertIsNone(parse_captured_year(None))
        self.assertIsNone(parse_captured_year("not-a-timestamp"))

    def test_crop_vertical_middle_keeps_requested_band(self) -> None:
        image = Image.fromarray(np.arange(5 * 4, dtype=np.uint8).reshape(5, 4))
        cropped = crop_vertical_middle(image, keep_ratio=0.6)
        self.assertEqual(cropped.size, (4, 3))

    def test_safe_label_normalizes_for_column_names(self) -> None:
        self.assertEqual(safe_label("Road / Sidewalk"), "road_sidewalk")
        self.assertEqual(safe_label(""), "class")

    def test_optional_image_metrics_can_be_removed_from_header(self) -> None:
        reduced = with_optional_image_metrics(CSV_HEADER, include_image_metrics=False)
        for column in IMAGE_METRIC_COLUMNS:
            self.assertNotIn(column, reduced)
        self.assertIn("lightglue_match_ratio", reduced)

    def test_image_quality_metrics_are_numeric(self) -> None:
        image = np.full((8, 8), 128, dtype=np.uint8)
        metrics = compute_image_quality_metrics(image)
        self.assertEqual(len(metrics), 6)
        self.assertTrue(all(isinstance(value, float) for value in metrics))

    def test_keypoint_coverage_uses_hull_and_ignores_sky_and_outlier(self) -> None:
        coverable = np.zeros((100, 100), dtype=bool)
        coverable[50:, :] = True
        clustered_points = np.array(
            [
                [10, 50],
                [50, 50],
                [10, 90],
                [20, 65],
                [25, 75],
                [35, 60],
            ],
            dtype=np.float32,
        )
        with_outlier = np.vstack([clustered_points, np.array([[99, 75]], dtype=np.float32)])

        clustered_coverage = _matched_keypoint_coverage(
            clustered_points,
            (100, 100),
            coverable,
        )
        outlier_coverage = _matched_keypoint_coverage(
            with_outlier,
            (100, 100),
            coverable,
        )

        self.assertGreater(clustered_coverage, 0.15)
        self.assertLess(clustered_coverage, 0.25)
        self.assertAlmostEqual(outlier_coverage, clustered_coverage, places=2)

    def test_keypoint_hull_iou_compares_same_size_hull_masks(self) -> None:
        left_points = np.array(
            [
                [10, 10],
                [50, 10],
                [10, 50],
                [30, 20],
            ],
            dtype=np.float32,
        )
        right_points_same = left_points.copy()
        right_points_shifted = left_points + np.array([40, 40], dtype=np.float32)

        self.assertAlmostEqual(
            _matched_keypoint_hull_iou(left_points, right_points_same, (100, 100)),
            1.0,
            places=4,
        )
        self.assertLess(
            _matched_keypoint_hull_iou(left_points, right_points_shifted, (100, 100)),
            0.05,
        )

    def test_pair_metadata_and_pairing_for_nearby_points(self) -> None:
        left = {"lat": 60.0, "lon": 24.0, "compass_angle": 5.0}
        right = {"lat": 60.0, "lon": 24.00001, "compass_angle": 355.0}
        meta = compute_pair_metadata(left, right)
        self.assertGreater(meta["dist_m"], 0)
        self.assertEqual(meta["angle_diff_deg"], 10.0)

        df = pd.DataFrame(
            [
                {"id": "a", **left, "captured_at": 1_704_067_200_000},
                {"id": "b", **right, "captured_at": 1_735_689_600_000},
            ]
        )
        pairs = pair_by_proximity(df, max_distance_m=2.0, time_filters=["year"])
        self.assertEqual(len(pairs), 1)

    def test_config_file_defaults_can_be_overridden(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "config.toml"
            config_path.write_text(
                """
                [process_pairs]
                area_wkt = "POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))"
                output = "outputs/from_config.csv"

                [outputs]
                images_dir = "data/from_config"

                [sampler]
                samples_per_group = 25
                """,
                encoding="utf-8",
            )
            parser = argparse.ArgumentParser()
            add_config_argument(parser)
            parser.add_argument("--area-wkt")
            parser.add_argument("--images-dir")
            parser.add_argument("--output")
            args = parse_args_with_config(
                parser,
                argv=[
                    "--config",
                    str(config_path),
                    "--output",
                    "outputs/from_cli.csv",
                ],
                default_config_path=config_path,
                script_sections=("process_pairs",),
            )
            self.assertEqual(args.area_wkt, "POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))")
            self.assertEqual(args.images_dir, "data/from_config")
            self.assertEqual(args.output, "outputs/from_cli.csv")

    def test_dotenv_supplies_access_token_env(self) -> None:
        old_token = os.environ.pop("PAIRWISE_TEST_MAPILLARY_TOKEN", None)
        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                config_path = Path(tmp_dir) / "config.toml"
                config_path.write_text(
                    """
                    [mapillary]
                    access_token_env = "PAIRWISE_TEST_MAPILLARY_TOKEN"
                    """,
                    encoding="utf-8",
                )
                (Path(tmp_dir) / ".env").write_text(
                    'PAIRWISE_TEST_MAPILLARY_TOKEN="from-dotenv"\n',
                    encoding="utf-8",
                )
                parser = argparse.ArgumentParser()
                add_config_argument(parser)
                parser.add_argument("--access-token")
                args = parse_args_with_config(
                    parser,
                    argv=["--config", str(config_path)],
                    default_config_path=config_path,
                    script_sections=("process_pairs",),
                )
                self.assertEqual(args.access_token, "from-dotenv")
        finally:
            if old_token is not None:
                os.environ["PAIRWISE_TEST_MAPILLARY_TOKEN"] = old_token
            else:
                os.environ.pop("PAIRWISE_TEST_MAPILLARY_TOKEN", None)

    def test_real_environment_overrides_dotenv(self) -> None:
        old_token = os.environ.get("PAIRWISE_TEST_MAPILLARY_TOKEN")
        os.environ["PAIRWISE_TEST_MAPILLARY_TOKEN"] = "from-environment"
        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                config_path = Path(tmp_dir) / "config.toml"
                config_path.write_text(
                    """
                    [mapillary]
                    access_token_env = "PAIRWISE_TEST_MAPILLARY_TOKEN"
                    """,
                    encoding="utf-8",
                )
                (Path(tmp_dir) / ".env").write_text(
                    "PAIRWISE_TEST_MAPILLARY_TOKEN=from-dotenv\n",
                    encoding="utf-8",
                )
                parser = argparse.ArgumentParser()
                add_config_argument(parser)
                parser.add_argument("--access-token")
                args = parse_args_with_config(
                    parser,
                    argv=["--config", str(config_path)],
                    default_config_path=config_path,
                    script_sections=("process_pairs",),
                )
                self.assertEqual(args.access_token, "from-environment")
        finally:
            if old_token is not None:
                os.environ["PAIRWISE_TEST_MAPILLARY_TOKEN"] = old_token
            else:
                os.environ.pop("PAIRWISE_TEST_MAPILLARY_TOKEN", None)

    def test_explicit_pair_csv_rows_are_loaded(self) -> None:
        args = argparse.Namespace(indices=None)
        df = pd.DataFrame(
            [
                {
                    "filename_left": "left_image.jpg",
                    "filename_right": "right_image.jpg",
                    "date_left": "2020",
                    "date_right": "2024",
                    "index": "pair-a",
                }
            ]
        )
        pairs = _load_explicit_csv_pairs(df, args)
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0]["id_left"], "left_image")
        self.assertEqual(pairs[0]["id_right"], "right_image")
        self.assertEqual(pairs[0]["index"], "pair-a")

    def test_sampler_can_build_month_groups(self) -> None:
        args = argparse.Namespace(
            time_filter=["month"],
            old_label="old",
            new_label="new",
            year_group_old=[2020],
            year_group_new=[2024],
        )
        images = pd.DataFrame(
            [
                {"id": "a", "captured_at": 1_704_067_200_000, "lat": 60.0, "lon": 24.0},
                {"id": "b", "captured_at": 1_707_350_400_000, "lat": 60.0, "lon": 24.0},
            ]
        )
        images = _add_time_columns(images, "month")
        groups = _build_sample_groups(images, args)
        self.assertEqual(len(groups["month_01"]), 1)
        self.assertEqual(len(groups["month_02"]), 1)
        self.assertEqual(len(groups["month_03"]), 0)

    def test_filtered_results_keep_rows_matching_all_criteria(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            source = Path(tmp_dir) / "results.csv"
            filtered = Path(tmp_dir) / "filtered.csv"
            pd.DataFrame(
                [
                    {
                        "id_left": "a",
                        "id_right": "b",
                        "lightglue_match_ratio": 0.30,
                        "lightglue_avg_distance": 0.04,
                        "lightglue_keypoint_coverage_min": 0.70,
                        "lightglue_keypoint_hull_iou": 0.40,
                        "seg_overlap_road_iou": 0.60,
                        "seg_overlap_mean_iou": 0.40,
                    },
                    {
                        "id_left": "c",
                        "id_right": "d",
                        "lightglue_match_ratio": 0.10,
                        "lightglue_avg_distance": 0.03,
                        "lightglue_keypoint_coverage_min": 0.90,
                        "lightglue_keypoint_hull_iou": 0.20,
                        "seg_overlap_road_iou": 0.80,
                        "seg_overlap_mean_iou": 0.50,
                    },
                ]
            ).to_csv(source, index=False)
            args = argparse.Namespace(
                output=str(source),
                filtered_output=str(filtered),
                filter_match_ratio_min=0.20,
                filter_avg_distance_max=0.08,
                filter_keypoint_coverage_min=0.50,
                filter_keypoint_hull_iou_min=0.30,
                filter_road_iou_min=0.50,
                filter_mean_iou_min=None,
            )
            info = _write_filtered_results(args)
            rows = pd.read_csv(filtered)
            self.assertEqual(info["filtered_rows"], 1)
            self.assertEqual(rows.loc[0, "id_left"], "a")

    def test_run_manifest_redacts_access_token(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            manifest_path = Path(tmp_dir) / "outputs" / "run_manifest.json"
            args = argparse.Namespace(access_token="secret-token", output="outputs/results.csv")
            original_argv = sys.argv
            sys.argv = [
                "process_pairs.py",
                "--access-token",
                "secret-token",
                "--output",
                "outputs/results.csv",
            ]
            try:
                written = write_run_manifest(
                    path=manifest_path,
                    project_root=ROOT,
                    script_name="process_pairs.py",
                    args=args,
                    started_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
                    extra={"input_mode": "csv"},
                )
            finally:
                sys.argv = original_argv
            manifest = json.loads(written.read_text(encoding="utf-8"))
            self.assertEqual(manifest["settings"]["access_token"], "***redacted***")
            self.assertEqual(manifest["command"][2], "***redacted***")
            self.assertEqual(manifest["extra"]["input_mode"], "csv")


if __name__ == "__main__":
    unittest.main()
