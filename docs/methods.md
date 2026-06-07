# Methods

This document summarizes the processing logic used by Pairwise Image Finder.
For configuration details, see [settings_reference.md](settings_reference.md).

## Overview

Pairwise Image Finder identifies and evaluates repeated street-level images of
the same or nearby locations. It supports two main processing inputs:

- Area mode queries Mapillary metadata inside a WKT polygon, pairs images using
  spatial and temporal constraints, downloads thumbnails when needed, and then
  evaluates each pair.
- CSV mode reads explicit image pairs from a local CSV and evaluates the
  corresponding local image files.

The output is a row-wise CSV of pair metadata, visual matching metrics,
optional image-quality metrics, optional semantic-overlap metrics, and optional
debug artifacts.

## Pair Selection

In area mode, candidate images are filtered and paired from Mapillary metadata.
The main pairing constraints are:

- geographic distance between capture locations;
- compass-angle difference;
- image type, such as panorama or flat perspective;
- optional temporal grouping by year, month, season, same season, or day/night.

The geographic distance is written as `dist_m`, and the compass difference is
written as `angle_diff_deg`. Randomized pair ordering can be made reproducible
with the configured random seed.

CSV mode skips geographic discovery and uses one explicit image pair per input
row. This mode is useful for reproducible experiments, non-Mapillary imagery,
and small validation examples.

## Image Preparation

Images are loaded from a local image directory or from a Mapillary download
cache. The processing path depends on the selected image geometry:

- Panorama mode treats images as equirectangular panoramas and can search for a
  horizontal yaw offset that best aligns the right image to the left image.
- Flat-image mode compares the input views directly.
- FOV/scale-search mode estimates overlap between perspective images with
  different fields of view, optionally using homography reprojection before
  cropping to the shared view.

For panorama processing, a configurable vertical crop can remove parts of the
image that are less useful for street-level alignment, such as excess sky or
vehicle hood regions.

## Feature Matching

The tool uses SuperPoint features with LightGlue matching. For each analysis
pair:

1. SuperPoint extracts keypoints and descriptors from both images.
2. LightGlue matches keypoints between the left and right images.
3. Optional keypoint masks remove keypoints in ignored regions, such as sky or
   temporary semantic classes.
4. The remaining matches are used to compute feature-matching metrics.

The main feature metrics are:

- `lightglue_match_ratio`: valid LightGlue matches divided by the smaller
  allowed keypoint count across the two images.
- `lightglue_avg_distance`: mean Euclidean distance between matched keypoints,
  normalized by the diagonal length of the first image. Lower values indicate
  that matched points are spatially closer in image coordinates.
- `lightglue_keypoint_coverage_min`: the fraction of each image's coverable
  non-sky area inside a robust convex hull around valid matched keypoints.
  Isolated spatial outliers are removed before measuring the hull, and the
  minimum value is useful when both images must be well covered.
  In FOV/scale-search mode, the primary coverage columns are measured on the
  original full images rather than the cropped/reprojected analysis views.
- `lightglue_keypoint_hull_iou`: intersection-over-union between same-size
  robust matched-keypoint convex hull masks from the left and right images.
  This measures whether matches occupy similar image regions, without
  estimating a homography between the views.

When no valid matches are available, the average distance is recorded as
infinite internally and may appear as an infinite or missing value depending on
the CSV consumer.

## Panorama Alignment

For panoramas, the right image can be circularly shifted along the horizontal
axis to compensate for yaw differences. The alignment score is based on
LightGlue matches after applying candidate yaw shifts. The selected shift is
written as `best_yaw_deg`.

The fast yaw path estimates yaw from one keypoint matching pass. The slower
search path evaluates candidate yaw angles at a configured step size. Optional
semantic reranking can refine candidate yaw choices using semantic-overlap
scores near the feature-based candidate.

## FOV and Scale Alignment

For perspective images with different fields of view, the scale-search path
estimates a shared overlap region. The retained overlap is summarized by
`fov_crop_fraction`; lower values indicate that a stronger crop was needed to
compare the views. When reprojection is enabled, a robust homography is
estimated from feature matches and used before selecting the overlap crop.

## Semantic Segmentation

When segmentation is enabled, the tool runs a OneFormer semantic segmentation
model. The configured dataset controls the label space, commonly Cityscapes or
ADE20K.

Semantic masks can be used in three ways:

- compute per-class intersection-over-union between the aligned pair;
- compute summary overlap metrics such as `seg_overlap_mean_iou` and
  `seg_overlap_road_iou`;
- optionally ignore temporary or weakly comparable regions when computing
  overlap or filtering keypoints.

Temporary classes such as people and vehicles can be excluded from semantic
IoU, depending on configuration. This is useful for longitudinal street-view
comparisons where transient objects should not dominate the pair score.

## Filtering

Filtering is an export step. The full result CSV is preserved, and an optional
filtered CSV can be written using thresholds such as:

- minimum `lightglue_match_ratio`;
- maximum `lightglue_avg_distance`;
- minimum `lightglue_keypoint_coverage_min`;
- minimum `lightglue_keypoint_hull_iou`;
- minimum `seg_overlap_road_iou`;
- minimum `seg_overlap_mean_iou`.

Filters are combined with AND logic. This keeps raw metrics available for audit
while producing a smaller review set.

## Limitations

The metrics are intended to support candidate discovery and quality filtering,
not to prove that two images depict an identical physical state. GPS metadata,
compass angles, panorama seams, motion blur, occlusions, seasonal changes,
lighting, and model errors can all affect scores. Human inspection or a
task-specific validation set is recommended when thresholds are used for
research decisions.
