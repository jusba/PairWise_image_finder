from typing import Optional

import torch

try:
    from lightglue import LightGlue, SuperPoint
except ImportError as exc:
    raise ImportError(
        "LightGlue is required for feature matching. It is not available on "
        "PyPI as lightglue; install project dependencies with "
        "`pip install -r requirements.txt`, or install it directly with "
        "`pip install \"lightglue @ git+https://github.com/cvg/LightGlue.git\"`."
    ) from exc

try:
    from .segmentation import Segmenter
except ImportError as exc:
    Segmenter = None
    _SEGMENTATION_IMPORT_ERROR = exc
else:
    _SEGMENTATION_IMPORT_ERROR = None

def pick_device(prefer: str | None = None) -> torch.device:
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

def init_models(device: torch.device):
    extractor = SuperPoint(max_num_keypoints=2048).eval().to(device)
    matcher = LightGlue(features='superpoint').eval().to(device)
    return extractor, matcher


def init_segmenter(
    device: torch.device,
    model_type: Optional[str] = None,
):
    """
    Initialize the semantic segmenter.

    model_type: "cityscapes" or "ade20k" or None (disabled)
    Returns CityscapesSegmenter or None.
    """
    if model_type is None:
        return None
    if Segmenter is None:
        raise ImportError(
            "utils.segmentation.Segmenter is unavailable. "
            "Install transformers and make sure utils/segmentation.py is on PYTHONPATH."
        ) from _SEGMENTATION_IMPORT_ERROR
    return Segmenter(model_type=model_type, device=device)
