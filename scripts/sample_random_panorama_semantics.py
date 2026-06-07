#!/usr/bin/env python
"""Randomly sample panorama Mapillary images and write semantic class proportions."""

from __future__ import annotations

try:
    from .random_semantic_sampler_common import run
except ImportError:  # pragma: no cover - direct script execution
    from random_semantic_sampler_common import run


def main() -> None:
    run(
        default_image_type="panorama",
        default_output="random_panorama_semantic_samples.csv",
        default_crop_keep=0.6,
    )


if __name__ == "__main__":
    main()
