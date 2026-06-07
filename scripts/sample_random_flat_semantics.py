#!/usr/bin/env python
"""Randomly sample flat Mapillary images and write semantic class proportions."""

from __future__ import annotations

try:
    from .random_semantic_sampler_common import run
except ImportError:  # pragma: no cover - direct script execution
    from random_semantic_sampler_common import run


def main() -> None:
    run(
        default_image_type="flat",
        default_output="random_flat_semantic_samples.csv",
    )


if __name__ == "__main__":
    main()
