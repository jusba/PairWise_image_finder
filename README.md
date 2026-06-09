# PairWise Image Finder

[![arXiv paper](https://img.shields.io/badge/📄-arXiv_Paper-9cf)](https://doi.org/10.48550/arXiv.2606.08795)

A tool for finding visually matching Mapillary (or user-defined) street-level images. 

The tool pairs images by metadata and geographic proximity, aligns flat
or panoramic views, computes feature matching metrics, and optionally adds
semantic segmentation overlap statistics.

## Main Capabilities

Three modes:
- Area mode: query Mapillary images inside a WKT polygon and pair nearby images.
- CSV mode: compare explicit image pairs listed in a local CSV file.
- Random sampling mode: sample images from an area for semantic-class proportions.

Compute visual alignment metrics
- Feature match ratio
- Matched feature average distance
- Matched feature convex hull coverage
- Matched feature convex hull overlap (mIoU)
- Semantic mIoU overlap

### Other features Capabilities

- Align panoramas by horizontal yaw.
- WIP Align perspective images with different fields of view by crop or reprojection.
- WIP Compute feature-match, image-quality, horizon, and semantic-overlap metrics.
- WIP temporal masking
- Export optional debug visualizations for inspection.

## Usage

Longitudinal/temporal change studies
- The tool can be used to find visually matching image pairs from different time periods for more accurate longitudinal/temporal assesments
  - Avoids problems due to irregular image capture methods (different camera angles/perspectives, road lanes, coverage)
  - Can find pairs from different temporal periods (years, months, seasons, same-season-different-year, diurnal)

Street-level perception studies
- Can help to find street-level images for perception studies attempting to compare locations at different times
  - Reduces need for costly manual inspection of images.


## Repository Layout

```text
scripts/        Command-line entry points for processing and sampling
utils/          Shared image, Mapillary, matching, geometry, and segmentation code
docs/           Settings, methods, and reproducibility reference
tests/          Lightweight tests for pure helper behavior
requirements.txt
                Environment file for local installs
```

## Installation

Use Python 3.10 or newer. You also need `git`, because LightGlue is installed
from its official GitHub repository rather than from PyPI.

To create a fresh virtual environment and install the dependencies from
`requirements.txt` on Linux/macOS:

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

On Windows:

```powershell
py -3 -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Optional package entry points can also be installed from the repository root:

```bash
pip install .
```

After that, `pairwise-process`, `pairwise-sample-panorama`, and
`pairwise-sample-flat` are available as CLI commands. The original
`python scripts/...` commands remain supported.

The installed commands are thin wrappers around the scripts in `scripts/`.
They use the same flags and read `config.toml` from the directory where you run
the command:

```bash
pairwise-process --dry-run
pairwise-process --config config.toml
pairwise-sample-panorama --dry-run
pairwise-sample-flat --dry-run
```

If an optional dependency is missing, install the relevant group:

```bash
pip install ".[area]"          # astral, shapely, mercantile, mapbox-vector-tile
pip install ".[segmentation]"  # transformers, scipy
```

If you see `No matching distribution found for lightglue`, pull the latest repo version and install again. The
requirement should read:

```text
lightglue @ git+https://github.com/cvg/LightGlue.git
```

### Compute Device

The default `config.toml` sets `device = "cuda"`, so normal runs require a
working CUDA-enabled PyTorch environment. If you do not have CUDA, change this
setting to `cpu` or `mps`, or override it on the command line:

```bash
python scripts/process_pairs.py --device cpu
```

If CUDA is requested but unavailable, the script exits with an error instead of
silently using CPU. 

For Mapillary area mode, create a Mapillary developer account and copy its
access token. Store the token in a local `.env` file, which is ignored by Git:

```bash
cp .env.example .env
```

Then edit `.env` and replace the placeholder token:

```bash
MAPILLARY_ACCESS_TOKEN=your-token
```

The public `config.toml` does not store the token. It only points to the
environment variable name:

```toml
[mapillary]
access_token_env = "MAPILLARY_ACCESS_TOKEN"
```

When a script starts, it loads `.env` automatically and then reads the token
through `access_token_env`.

## Quick Start

The scripts automatically read `config.toml` from the directory where you run
them, which is usually the repository root. You can fill in `area_wkt` there
and keep your Mapillary token in `.env`.

To preview an area run without downloading images or running models:

```bash
python scripts/process_pairs.py --dry-run
```

A dry run reports the shape of the planned run, for example:

```text
images found: 1,240
pairs found: 312
after year filter: 104
estimated output: outputs/area_results.csv
will save images: no
will run segmentation: yes
```

CLI flags work and override values from `config.toml`.

#### Area Mode

Area mode queries Mapillary image metadata inside a WKT polygon, pairs nearby
images, downloads thumbnails, and processes the pairs.

```bash
python scripts/process_pairs.py \
  --area-wkt "POLYGON ((24.93 60.16, 24.94 60.16, 24.94 60.17, 24.93 60.17, 24.93 60.16))" \
  --images-dir data/mapillary_cache \
  --output outputs/area_results.csv \
  --image-type panorama \
  --panorama \
  --segmentation
```

#### CSV Mode

CSV mode expects one explicit image pair per row. Images are read from
`--images-dir`.

```csv
filename_left,filename_right,id_left,id_right,date_left,date_right,index
old_street.jpg,new_street.jpg,old_street,new_street,2018-06-14,2024-06-20,pair_001
```

```bash
python scripts/process_pairs.py \
  --input-csv data/input_pairs.csv \
  --images-dir data/images \
  --output outputs/results.csv \
  --no-save-debug-images
```

#### Random area sampling mode
For sampling within an area:

```bash
python scripts/sample_random_panorama_semantics.py \
  --area-wkt "POLYGON ((...))" \
  --samples-per-group 50 \
  --output outputs/random_panorama_semantic_samples.csv \
  --seed 42
```

The sampler also supports dry runs:

```bash
python scripts/sample_random_panorama_semantics.py --dry-run
```

#### Result filtering

Filtering is an optional export step, not a separate input mode. It is enabled
by setting `filtered_output` or passing `--filtered-output`. The full result
CSV is always preserved, and the filtered CSV is written after processing
finishes. This second file is useful when you want to keep all raw pair metrics
for auditing, but only inspect or share pairs that meet quality thresholds.

Filter criteria are combined with AND logic. For example, the command below
keeps only rows where `lightglue_match_ratio >= 0.20`,
`lightglue_avg_distance <= 0.08`,
`lightglue_keypoint_coverage_min >= 0.50`,
`lightglue_keypoint_hull_iou >= 0.30`, and
`seg_overlap_road_iou >= 0.50`:

```bash
python scripts/process_pairs.py \
  --filter-match-ratio-min 0.20 \
  --filter-avg-distance-max 0.08 \
  --filter-keypoint-coverage-min 0.50 \
  --filter-keypoint-hull-iou-min 0.30 \
  --filter-road-iou-min 0.50 \
  --filtered-output outputs/filtered_results.csv
```

The same settings can be placed in `config.toml`:

```toml
[outputs]
filtered_output = "outputs/filtered_results.csv"

[filtering]
filter_match_ratio_min = 0.20
filter_avg_distance_max = 0.08
filter_keypoint_coverage_min = 0.50
filter_keypoint_hull_iou_min = 0.30
filter_road_iou_min = 0.50
filter_mean_iou_min = 0.35
```

Available filters are `filter_match_ratio_min`,
`filter_avg_distance_max`, `filter_keypoint_coverage_min`,
`filter_keypoint_hull_iou_min`, `filter_road_iou_min`, and
`filter_mean_iou_min`. Semantic IoU filters only produce meaningful filtered
outputs when `segmentation = true` or `--segmentation` was used.

Successful area and CSV processing runs write `outputs/run_manifest.json` by
default. The manifest records the command, settings, Git state, Python version,
selected dependency versions, output paths, and filter summary.


See [docs/settings_reference.md](docs/settings_reference.md) for a fuller reference.
See [docs/methods.md](docs/methods.md) for a methods-oriented description of
the pairing, alignment, matching, segmentation, filtering, and reproducibility
workflow.
See [docs/example_queries.md](docs/example_queries.md) for example queries.

## Key Result Columns

| Column | Meaning |
| --- | --- |
| `lightglue_match_ratio` | Fraction of detected keypoints that LightGlue matched; higher usually means stronger visual correspondence. |
| `lightglue_avg_distance` | Average matched-keypoint distance normalized by image diagonal; lower means matched points are spatially closer. |
| `lightglue_keypoint_coverage_min` | Lower of the left/right non-sky image fractions covered by the convex hull of matched keypoints. |
| `lightglue_keypoint_hull_iou` | mIoU between left/right matched-keypoint convex hull masks; higher means matches occupy similar image regions. |
| `seg_overlap_road_iou` | Semantic mIoU for road-like classes when available; useful for finding images on the same street lane. |

`seg_class_props_left_before_json` and `seg_class_props_right_before_json` can used to access the semantic segmentation results.

## Citation

If you find the tool useful, citing the reference paper is appreciated :):

```text
@online{torkkoPairWiseImageFinder2026,
  title = {PairWise Image Finder: An Open-source Tool for Finding Visually Aligned Street-Level Image Pairs for Urban Perception Studies},
  shorttitle = {PairWise Image Finder},
  author = {Torkko, Jussi},
  date = {2026},
  doi = {10.48550/ARXIV.2606.08795},
  urldate = {2026-06-09},
  pubstate = {prepublished},
  version = {1},
  keywords = {Computer Vision and Pattern Recognition (cs.CV),FOS: Computer and information sciences}
}
```

## Acknowledgements

This project builds on several external datasets, APIs, models, and libraries:

- Mapillary provides the street-level image metadata and image access used in
  area mode.
- LightGlue provides the feature-matching model used for visual correspondence.
- SuperPoint provides the local feature keypoints and descriptors used with
  LightGlue.
- OneFormer provides semantic segmentation models used for optional class
  overlap metrics.

When publishing results, cite the relevant upstream projects and model papers
according to their own citation guidance.

## License

MIT. See [LICENSE](LICENSE).
