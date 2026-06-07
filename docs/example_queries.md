### Example Area Commands

Remember to add your Mapillary TOKEN.

Define the Helsinki example polygon once as a shell variable. WKT coordinates
are in longitude/latitude order:

```bash
export AREA_WKT='POLYGON ((24.956946 60.166365, 24.969606 60.162629, 24.982438 60.167603, 24.975314 60.170827, 24.96922 60.169055, 24.963942 60.170293, 24.959865 60.169909, 24.958749 60.169098, 24.956088 60.166813, 24.956946 60.166365))'
```

Preview flat-image pair counts without downloading images or loading models:

```bash
python scripts/process_pairs.py \
  --area-wkt "$AREA_WKT" \
  --image-type flat \
  --max-distance 2.5 \
  --max-angle-diff 30 \
  --time-filter year \
  --year-group-left 2016 2017 2018 \
  --year-group-right 2024 2025 2026 \
  --dry-run
```

Process flat images with segmentation but without saving downloaded images,
segmentation caches, debug images, or other image artifacts:

```bash
python scripts/process_pairs.py \
  --area-wkt "$AREA_WKT" \
  --image-type flat \
  --images-dir ./images_cache \
  --output outputs/results_flat.csv \
  --manifest-path outputs/results_flat_manifest.json \
  --max-distance 2.5 \
  --max-angle-diff 30 \
  --device cuda \
  --time-filter year \
  --segmentation \
  --ignore-temporary \
  --ignore-sky-keypoints \
  --sky-keypoint-source segmentation \
  --sky-keypoint-boundary-px 50 \
  --segmentation-max-width 1024 \
  --year-group-left 2016 2017 2018 \
  --year-group-right 2024 2025 2026 \
  --no-save-mapillary-images \
  --no-save-debug-images \
  --no-save-artifacts
```

`--images-dir` is required for a normal run, although area-mode downloads use
temporary storage when `--no-save-mapillary-images` or `--no-save-artifacts`
is active. This command writes the result CSV and JSON run manifest, but no
image artifacts or segmentation cache.

For flat images with differing fields of view, enable overlap cropping and
homography reprojection:

```bash
python scripts/process_pairs.py \
  --area-wkt "$AREA_WKT" \
  --image-type flat \
  --images-dir ./images_cache \
  --output outputs/results_flat_reprojected.csv \
  --manifest-path outputs/results_flat_reprojected_manifest.json \
  --max-distance 2.5 \
  --max-angle-diff 30 \
  --device cuda \
  --time-filter year \
  --scale-search \
  --scale-reproject \
  --year-group-left 2016 2017 2018 \
  --year-group-right 2024 2025 2026 \
  --no-save-artifacts
```

Process panorama pairs with horizontal yaw alignment:

```bash
python scripts/process_pairs.py \
  --area-wkt "$AREA_WKT" \
  --image-type panorama \
  --images-dir ./images_cache \
  --output outputs/results_panorama.csv \
  --manifest-path outputs/results_panorama_manifest.json \
  --max-distance 2.5 \
  --max-angle-diff 30 \
  --device cuda \
  --time-filter year \
  --panorama \
  --year-group-left 2016 2017 2018 \
  --year-group-right 2024 2025 2026 \
  --no-save-artifacts
```

Saved visualizationds and outputs

```bash
python scripts/process_pairs.py \
--area-wkt "$AREA_WKT" \
--image-type flat \
--images-dir ./images_cache \
--save-mapillary-images \
--output outputs/results_flat_visualized.csv \
--manifest-path outputs/results_flat_visualized_manifest.json \
--max-distance 2.5 \
--max-angle-diff 30 \
--device cuda \
--time-filter year \
--segmentation \
--ignore-temporary \
--ignore-sky-keypoints \
--sky-keypoint-source segmentation \
--sky-keypoint-boundary-px 50 \
--segmentation-cache-dir segmentation_cache \
--segmentation-max-width 1024 \
--scale-search \
--scale-reproject \
--year-group-left 2016 2017 2018 \
--year-group-right 2024 2025 2026 \
--save-debug-images
```

