# Settings Reference

This repository can be run with normal command-line flags or with the root
`config.toml` file. The scripts automatically read `config.toml` from the
current working directory when it exists, which is normally the repository
root. Values passed on the command line override config values.

For installation and runnable examples, see the
[README](../README.md#quick-start). For algorithm details, see
[Methods](methods.md).

## Config Structure

The config loader flattens TOML sections into normal CLI-style option names.
For example:

```toml
[pairing]
max_distance = 2.5
time_filter = ["year"]
```

is equivalent to:

```bash
--max-distance 2.5 --time-filter year
```

The sections are for readability. Shared sections such as `[mapillary]`,
`[pairing]`, `[processing]`, and `[outputs]` can be used by multiple scripts.
The `[sampler]` section is only used by the random semantic sampling scripts.


## Pair Discovery

CSV-mode settings:

- `input_csv`: one-row-per-pair CSV with two image filename columns.
- `images_dir`: local directory containing the images named in the CSV.
- `indices`: optional subset of index values.
- `max_pairs_per_index`: optional throttle for legacy grouped CSV input.
- `filename_left` and `filename_right`: required pair columns.
- `id_left`, `id_right`, `date_left`, `date_right`, and `index`: optional
  metadata columns.

Area-mode settings:

- `area_wkt`: WKT polygon or multipolygon in longitude/latitude coordinates.
- `image_type`: `all`, `panorama`, or `flat`.
- `max_distance`: maximum GPS distance in metres between paired images.
- `max_angle_diff`: maximum compass-angle difference in degrees.
- `time_filter`: `any`, `year`, `month`, `season`, `same-season`, or `time`.
- `year_group_left` and `year_group_right`: optional year groups; one image
  must come from each group.
- `pair_random_seed`: optional seed for reproducible pair order.

Shared processing settings:

- `device`: `cuda`, `mps`, or `cpu`.
- `image_metrics`: include brightness, contrast, sharpness, noise,
  dark-fraction, and horizon metrics.

## Panorama Alignment

- `panorama`: enable equirectangular panorama alignment.
- `yaw_step`: coarse yaw-search step in degrees.
- `crop_keep`: vertical fraction retained for alignment and analysis.
- `crop_top_bias`: vertical crop bias from `-1.0` to `1.0`.
- `panorama_fast_yaw`: estimate yaw from one keypoint matching pass.
- `panorama_semantic_rerank`: rerank nearby yaw candidates with segmentation.
- `semantic_rerank_radius`: yaw search radius for semantic reranking.
- `semantic_rerank_step`: yaw step for semantic reranking.

## FOV Alignment

- `scale_search`: enable robust field-of-view crop alignment.
- `scale_reproject`: estimate homographies and warp the less distorted image
  before cropping to shared overlap.

`scale_search` and `panorama` are mutually exclusive. `scale_reproject`
requires `scale_search`.

## Semantic Segmentation

- `segmentation`: enable semantic segmentation.
- `seg_dataset`: `cityscapes` or `ade20k`.
- `segmentation_max_width`: resize width before segmentation; `0` means full
  resolution.
- `segmentation_cache_dir`: optional `.npy` mask cache.
- `ignore_temporary`: ignore dynamic classes such as people and vehicles in IoU.
- `seg_crop_top_frac` and `seg_crop_bottom_frac`: crop vertical mask regions.
- `ignore_sky_keypoints`: exclude keypoints deep inside sky or temporary classes.
- `sky_keypoint_source`: `segmentation` or `heuristic`.
- `sky_keypoint_boundary_px`: retain keypoints close to non-ignored boundaries.

## Random Sampling

Sampler settings:

- `samples_per_group`: target sample count for each year group.
- `time_filter`: balance mode. Possible values are:
  `any`, `year`, `month`, `season`, `same-season`, `time`.
- `year_group_old` and `year_group_new`: year lists used by `year` and
  `same-season`.
- `old_label` and `new_label`: labels written into the output CSV for
  year-based modes.
- `seed`: random seed for reproducible sampling.
- `allow_short`: allow fewer samples if a group has too few images.
- `save_images_dir`: optional directory for sampled image crops.

Sampler balance modes:

- `any`: one random sample pool from all images in the area.
- `year`: equal samples from `year_group_old` and `year_group_new`.
- `month`: equal samples from each calendar month, January through December.
- `season`: equal samples from spring, summer, autumn, and winter.
- `same-season`: equal samples from old/new year groups inside each season.
- `time`: equal samples from daytime and nighttime images.

`sample_random_panorama_semantics.py` defaults to panorama images and
`crop_keep = 0.6`. `sample_random_flat_semantics.py` defaults to flat images.

## Output And Artifacts

Generated data should stay outside the public repository. The `.gitignore`
already excludes result CSVs, downloaded image caches, segmentation caches,
debug images, and local config variants.

Useful output-related settings:

- `output`: result CSV path.
- `manifest_path`: JSON run manifest path. Default: `outputs/run_manifest.json`.
- `backup_dir`: heartbeat backup directory for CSV mode.
- `save_debug_images`: write match and panorama/FOV debug images.
- `save_mapillary_images`: keep or discard downloaded Mapillary images.
- `no_save_artifacts`: disable optional image artifacts.
- `profile_timing`: write per-stage timing CSV.
- `filtered_output`: optional CSV path for filtered result rows.

## Filtered Result Outputs

Set `filtered_output` to write a second CSV while preserving the complete
result at `output`. Enabled thresholds use AND logic. Without thresholds, the
filtered output contains all result rows.

Available filter settings:

- `filter_match_ratio_min`: keep rows where `lightglue_match_ratio` is greater
  than or equal to the threshold.
- `filter_avg_distance_max`: keep rows where `lightglue_avg_distance` is less
  than or equal to the threshold.
- `filter_keypoint_coverage_min`: keep rows where
  `lightglue_keypoint_coverage_min` is greater than or equal to the threshold.
  This requires both images to have matched keypoints spread across enough of
  their non-sky coverable area.
- `filter_keypoint_hull_iou_min`: keep rows where
  `lightglue_keypoint_hull_iou` is greater than or equal to the threshold. This
  requires the left/right matched-keypoint hulls to occupy similar image
  regions after resizing to the same dimensions.
- `filter_road_iou_min`: keep rows where `seg_overlap_road_iou` is greater than
  or equal to the threshold. This is most useful when segmentation is enabled.
- `filter_mean_iou_min`: keep rows where `seg_overlap_mean_iou` is greater than
  or equal to the threshold. This is most useful when segmentation is enabled.

## Result CSV Columns

Important pairwise result columns:

| Column | Produced in | Meaning | Useful direction |
| --- | --- | --- | --- |
| `id_left`, `id_right` | all modes | Image IDs or filename-derived IDs for the two compared images. | identifier |
| `date_left`, `date_right` | all modes | Capture date metadata when available. Area mode stores Mapillary timestamps in milliseconds. | metadata |
| `lightglue_match_ratio` | all modes | Fraction of detected keypoints that LightGlue matched between the two analysis images. | higher is usually better |
| `lightglue_avg_distance` | all modes | Average matched-keypoint distance normalized by the image diagonal. | lower is usually better |
| `lightglue_keypoint_coverage_left`, `lightglue_keypoint_coverage_right`, `lightglue_keypoint_coverage_min` | all modes | Non-sky image fraction covered by the robust convex hull of matched keypoints; in FOV mode this is measured on the original full images. | higher is usually better |
| `lightglue_keypoint_hull_iou` | all modes | IoU between same-size left/right robust matched-keypoint convex hull masks. | higher means hulls occupy similar image regions |
| `lightglue_homography_inliers` | all modes | Number of LightGlue matches kept as homography inliers by OpenCV RANSAC. | higher is usually better |
| `lightglue_homography_total` | all modes | Number of matches considered for homography estimation. | context |
| `lightglue_homography_ratio` | all modes | Homography inlier share, calculated from inliers divided by total matches. | higher is usually better |
| `seg_overlap_mean_iou` | segmentation | Mean intersection-over-union across comparable semantic classes. | higher means stronger semantic overlap |
| `seg_overlap_road_iou` | segmentation | Semantic IoU for road-like classes when the selected model exposes a road class. | higher means stronger road alignment |
| `seg_overlap_per_class_json` | segmentation | JSON object containing semantic IoU per class ID. | inspect per class |
| `seg_class_props_left_before_json`, `seg_class_props_right_before_json` | segmentation | Class proportions for each image before temporary-object masking. | inspect composition |
| `seg_class_props_left_temp_masked_json`, `seg_class_props_right_temp_masked_json` | segmentation | Class proportions after masking temporary/dynamic classes. | inspect stable scene composition |
| `seg_temp_union_fraction` | segmentation | Fraction of pixels occupied by temporary/dynamic classes in either image. | lower means fewer temporary objects |
| `orig_lightglue_match_ratio`, `orig_lightglue_avg_distance`, `orig_lightglue_keypoint_coverage_min`, `orig_lightglue_keypoint_hull_iou` | panorama/FOV modes | Full-image or unaligned baseline feature metrics before panorama yaw or FOV crop processing. | compare with aligned metrics |
| `best_yaw_deg` | panorama mode | Horizontal yaw shift, in degrees, selected for panorama alignment. | diagnostic |
| `panorama_alignment_method` | panorama mode | Alignment path used for the panorama pair, such as normal yaw search or fast yaw. | diagnostic |
| `fov_crop_fraction` | FOV mode | Smaller retained-area fraction after robust overlap cropping. | higher means less cropping |
| `fov_left_retained_fraction`, `fov_right_retained_fraction` | FOV mode | Per-image retained fractions after FOV alignment. | higher means less cropping |
| `dist_m` | area mode | Haversine GPS distance between the two image positions in metres. | lower means closer capture points |
| `angle_diff_deg` | area mode | Smallest compass-angle difference between the two captures. | lower means more similar viewing direction |
| `is_pano_left`, `is_pano_right` | area mode | Whether each Mapillary image is marked as panorama metadata. | metadata |
| `lat_left`, `lon_left`, `lat_right`, `lon_right` | area mode | Mapillary image positions used for pairing. | metadata |

## Run Manifests

Area and CSV processing runs write a JSON manifest after successful processing.
By default it is saved to `outputs/run_manifest.json`; change this with
`manifest_path` or `--manifest-path`.

