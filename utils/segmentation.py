# utils/segmentation.py

import json
import logging
from typing import Dict, Tuple, List, Optional

logger = logging.getLogger(__name__)

import torch
import numpy as np
from PIL import Image

try:
    from transformers import OneFormerProcessor, OneFormerForUniversalSegmentation
except ImportError as e:
    OneFormerProcessor = None
    OneFormerForUniversalSegmentation = None
    _IMPORT_ERROR = e
else:
    _IMPORT_ERROR = None

# Cityscapes color palette
CITYSCAPES_COLORS: Dict[int, Tuple[int, int, int]] = {
    0: (128, 64, 128),
    1: (244, 35, 232),
    2: (70, 70, 70),
    3: (102, 102, 156),
    4: (190, 153, 153),
    5: (153, 153, 153),
    6: (250, 170, 30),
    7: (220, 220, 0),
    8: (107, 142, 35),
    9: (152, 251, 152),
    10: (70, 130, 180),
    11: (220, 20, 60),
    12: (255, 0, 0),
    13: (0, 0, 142),
    14: (0, 0, 70),
    15: (0, 60, 100),
    16: (0, 80, 100),
    17: (0, 0, 230),
    18: (119, 11, 32),
    19: (0, 0, 0),
}

# ADE20K color palette
ADE20K_COLORS: Dict[int, Tuple[int, int, int]] = {
    0: (120, 120, 120),
    1: (180, 120, 120),
    2: (6, 230, 230),
    3: (80, 50, 50),
    4: (4, 200, 3),
    5: (120, 120, 80),
    6: (140, 140, 140),
    7: (204, 5, 255),
    8: (230, 230, 230),
    9: (4, 250, 7),
    10: (224, 5, 255),
    11: (235, 255, 7),
    12: (150, 5, 61),
    13: (120, 120, 70),
    14: (8, 255, 51),
    15: (255, 6, 82),
    16: (143, 255, 140),
    17: (204, 255, 4),
    18: (255, 51, 7),
    19: (204, 70, 3),
    20: (0, 102, 200),
    21: (61, 230, 250),
    22: (255, 6, 51),
    23: (11, 102, 255),
    24: (255, 7, 71),
    25: (255, 9, 224),
    26: (9, 7, 230),
    27: (220, 220, 220),
    28: (255, 9, 92),
    29: (112, 9, 255),
    30: (8, 255, 214),
    31: (7, 255, 224),
    32: (255, 184, 6),
    33: (10, 255, 71),
    34: (255, 41, 10),
    35: (7, 255, 255),
    36: (224, 255, 8),
    37: (102, 8, 255),
    38: (255, 61, 6),
    39: (255, 194, 7),
    40: (255, 122, 8),
    41: (0, 255, 20),
    42: (255, 8, 41),
    43: (255, 5, 153),
    44: (6, 51, 255),
    45: (235, 12, 255),
    46: (160, 150, 20),
    47: (0, 163, 255),
    48: (140, 140, 140),
    49: (250, 10, 15),
    50: (20, 255, 0),
    51: (31, 255, 0),
    52: (255, 31, 0),
    53: (255, 224, 0),
    54: (153, 255, 0),
    55: (0, 0, 255),
    56: (255, 71, 0),
    57: (0, 235, 255),
    58: (0, 173, 255),
    59: (31, 0, 255),
    60: (11, 200, 200),
    61: (255, 82, 0),
    62: (0, 255, 245),
    63: (0, 61, 255),
    64: (0, 255, 112),
    65: (0, 255, 133),
    66: (255, 0, 0),
    67: (255, 163, 0),
    68: (255, 102, 0),
    69: (194, 255, 0),
    70: (0, 143, 255),
    71: (51, 255, 0),
    72: (0, 82, 255),
    73: (0, 255, 41),
    74: (0, 255, 173),
    75: (10, 0, 255),
    76: (173, 255, 0),
    77: (0, 255, 153),
    78: (255, 92, 0),
    79: (255, 0, 255),
    80: (255, 0, 245),
    81: (255, 0, 102),
    82: (255, 173, 0),
    83: (255, 0, 20),
    84: (255, 184, 184),
    85: (0, 31, 255),
    86: (0, 255, 61),
    87: (0, 71, 255),
    88: (255, 0, 204),
    89: (0, 255, 194),
    90: (0, 255, 82),
    91: (0, 10, 255),
    92: (0, 112, 255),
    93: (51, 0, 255),
    94: (0, 194, 255),
    95: (0, 122, 255),
    96: (0, 255, 163),
    97: (255, 153, 0),
    98: (0, 255, 10),
    99: (255, 112, 0),
    100: (143, 255, 0),
    101: (82, 0, 255),
    102: (163, 255, 0),
    103: (255, 235, 0),
    104: (8, 184, 170),
    105: (133, 0, 255),
    106: (0, 255, 92),
    107: (184, 0, 255),
    108: (255, 0, 31),
    109: (0, 184, 255),
    110: (0, 214, 255),
    111: (255, 0, 112),
    112: (92, 255, 0),
    113: (0, 224, 255),
    114: (112, 224, 255),
    115: (70, 184, 160),
    116: (163, 0, 255),
    117: (153, 0, 255),
    118: (71, 255, 0),
    119: (255, 0, 163),
    120: (255, 204, 0),
    121: (255, 0, 143),
    122: (0, 255, 235),
    123: (133, 255, 0),
    124: (255, 0, 235),
    125: (245, 0, 255),
    126: (255, 0, 122),
    127: (255, 245, 0),
    128: (10, 190, 212),
    129: (214, 255, 0),
    130: (0, 204, 255),
    131: (20, 0, 255),
    132: (255, 255, 0),
    133: (0, 153, 255),
    134: (0, 41, 255),
    135: (0, 255, 204),
    136: (41, 0, 255),
    137: (41, 255, 0),
    138: (173, 0, 255),
    139: (0, 245, 255),
    140: (71, 0, 255),
    141: (122, 0, 255),
    142: (0, 255, 184),
    143: (0, 92, 255),
    144: (184, 255, 0),
    145: (0, 133, 255),
    146: (255, 214, 0),
    147: (25, 194, 194),
    148: (102, 255, 0),
    149: (92, 0, 255),
}


class Segmenter:
    """
    Wrapper around OneFormer that supports both Cityscapes and ADE20K.

    - segment_image(image) -> (H, W) uint8 class index mask
    - colorize_mask(mask)  -> (H, W, 3) uint8 RGB mask
    - road_class_ids       -> list of class indices whose label contains 'road'
    """

    def __init__(
        self,
        model_type: str = "cityscapes",
        model_name: Optional[str] = None,
        device: Optional[torch.device] = None,
    ):
        if _IMPORT_ERROR is not None:
            raise ImportError(
                "transformers and OneFormer are required for segmentation: "
                'pip install ".[segmentation]"'
            ) from _IMPORT_ERROR

        self.model_type = model_type.lower().strip()
        if self.model_type == "ade20k":
            model_name = model_name or "shi-labs/oneformer_ade20k_dinat_large"
            self.colors = ADE20K_COLORS
        elif self.model_type == "cityscapes":
            model_name = model_name or "shi-labs/oneformer_cityscapes_swin_large"
            self.colors = CITYSCAPES_COLORS
        else:
            raise ValueError(f"Unsupported model_type: {model_type}")

        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.processor = OneFormerProcessor.from_pretrained(model_name)
        self.model = OneFormerForUniversalSegmentation.from_pretrained(model_name).to(
            self.device
        )
        self.model.eval()

               # id2label mapping to discover road classes at runtime
        cfg = getattr(self.model, "config", None)
        raw_id2label = getattr(cfg, "id2label", {}) or {}
        self.id2label: Dict[int, str] = {
            int(k): v for k, v in raw_id2label.items()
        }

        # Precompute set of "road-like" classes (label containing 'road')
        self.road_class_ids: List[int] = [
            i for i, name in self.id2label.items() if "road" in name.lower()
        ]

        # Precompute set of sky classes for optional keypoint filtering.
        self.sky_class_ids: List[int] = [
            i for i, name in self.id2label.items() if "sky" in name.lower()
        ]

        # Precompute "temporary / dynamic" classes we want to ignore (people, vehicles...)
        TEMPORAL_KEYWORDS = [
            "person",
            "people",
            "pedestrian",
            "rider",
            "car",
            "van",
            "taxi",
            "trailer",
            "caravan",
            "truck",
            "bus",
            "train",
            "tram",
            "motorcycle",
            "bicycle",
            "bike",
            "vehicle",
        ]
        self.temporary_class_ids: List[int] = [
            i
            for i, name in self.id2label.items()
            if any(kw in name.lower() for kw in TEMPORAL_KEYWORDS)
        ]
         # default: do NOT ignore temporary classes unless flag is passed
        self.ignore_temporary: bool = False

        logger.info("Segmentation model initialized: %s on %s", self.model_type, self.device)
        if self.road_class_ids:
            logger.info("Road classes: %s", self.road_class_ids)
        else:
            logger.warning("No road classes found in id2label; road IoU will be NA.")
        if self.sky_class_ids:
            logger.info("Sky classes: %s", self.sky_class_ids)
        else:
            logger.warning("No sky classes found in id2label; sky keypoint filtering will be disabled.")
        if self.temporary_class_ids:
            logger.info("Temporary classes available for ignore: %s", self.temporary_class_ids)


    # -------- core API --------

    def segment_image(self, image: Image.Image) -> np.ndarray:
        """Segment a PIL image -> (H, W) array of class indices (uint8)."""
        image = image.convert("RGB")
        inputs = self.processor(
            images=image, task_inputs=["semantic"], return_tensors="pt"
        ).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            semantic_map = self.processor.post_process_semantic_segmentation(
                outputs, target_sizes=[image.size[::-1]]
            )[0]
        return semantic_map.cpu().numpy().astype(np.uint8)

    def colorize_mask(self, mask: np.ndarray) -> np.ndarray:
        """Convert (H, W) mask of class indices to an RGB mask using the palette."""
        h, w = mask.shape
        color_mask = np.zeros((h, w, 3), dtype=np.uint8)
        for class_id, color in self.colors.items():
            color_mask[mask == class_id] = color
        return color_mask

    # -------- helpers for overlaps --------

    @staticmethod
    def compute_per_class_iou(
            left_mask: np.ndarray,
            right_mask: np.ndarray,
        ) -> Dict[int, float]:
            """
            Compute IoU per class between two segmentation masks (same H, W).
            Returns {class_id: IoU}.
            """
            assert left_mask.shape == right_mask.shape, "Masks must have same shape"
            classes = np.union1d(left_mask.flatten(), right_mask.flatten()).astype(np.int64)
            iou: Dict[int, float] = {}

            for cls in classes:
                m_l = left_mask == cls
                m_r = right_mask == cls
                inter = np.logical_and(m_l, m_r).sum()
                union = np.logical_or(m_l, m_r).sum()
                if union == 0:
                    continue
                iou[int(cls)] = float(inter) / float(union)

            return iou

    @staticmethod
    def compute_class_proportions(
        mask: np.ndarray,
        valid_pixels: Optional[np.ndarray] = None,
    ) -> Dict[int, float]:
        """
        Compute per-class pixel proportions within valid pixels.

        Returns {class_id: fraction}. Fractions sum to 1.0 unless there are no
        valid pixels, in which case an empty dict is returned.
        """
        if valid_pixels is not None:
            values = mask[valid_pixels]
        else:
            values = mask.reshape(-1)
        if values.size == 0:
            return {}
        classes, counts = np.unique(values.astype(np.int64), return_counts=True)
        total = float(values.size)
        return {int(cls): float(count) / total for cls, count in zip(classes, counts)}

    def summarize_class_proportions(
        self,
        left_mask: np.ndarray,
        right_mask: np.ndarray,
    ) -> Tuple[str, str, str, str, float]:
        """
        Return class proportions before and after cross-image temporary masking.

        The temporary mask is the union of temporary classes in either image.
        Sky is not part of temporary_class_ids, so sky remains unless covered by
        a temporary object in the opposite image at the same pixel.
        """
        assert left_mask.shape == right_mask.shape, "Masks must have same shape"
        left_before = self.compute_class_proportions(left_mask)
        right_before = self.compute_class_proportions(right_mask)

        if self.temporary_class_ids:
            temp_union = (
                np.isin(left_mask, self.temporary_class_ids)
                | np.isin(right_mask, self.temporary_class_ids)
            )
        else:
            temp_union = np.zeros(left_mask.shape, dtype=bool)
        valid_after = ~temp_union
        left_after = self.compute_class_proportions(left_mask, valid_after)
        right_after = self.compute_class_proportions(right_mask, valid_after)
        temp_union_fraction = float(temp_union.mean()) if temp_union.size else 0.0

        return (
            json.dumps(left_before, sort_keys=True),
            json.dumps(right_before, sort_keys=True),
            json.dumps(left_after, sort_keys=True),
            json.dumps(right_after, sort_keys=True),
            temp_union_fraction,
        )

    def summarize_iou(
        self,
        left_mask: np.ndarray,
        right_mask: np.ndarray,
    ) -> Tuple[Optional[float], Optional[float], str]:
        """
        Compute IoUs, optionally ignoring temporary classes (cars, people, vehicles).
        Returns: (mean_iou_all, mean_iou_road, per_class_json)
        """

        # If flag is OFF → behave normally
        if not getattr(self, "ignore_temporary", False):
            iou_dict = self.compute_per_class_iou(left_mask, right_mask)
        else:
            # Pixels where either image has temporary class
            ignore_pixels = (
                np.isin(left_mask, self.temporary_class_ids)
                | np.isin(right_mask, self.temporary_class_ids)
            )

            if ignore_pixels.any():
                sentinel = 255  # safe "void" label
                left_proc = left_mask.copy()
                right_proc = right_mask.copy()

                left_proc[ignore_pixels] = sentinel
                right_proc[ignore_pixels] = sentinel

                iou_dict = self.compute_per_class_iou(left_proc, right_proc)
                iou_dict.pop(sentinel, None)  # remove sentinel
            else:
                iou_dict = self.compute_per_class_iou(left_mask, right_mask)

        # Compute mean IoUs
        mean_iou_all = float(np.mean(list(iou_dict.values()))) if iou_dict else None

        mean_iou_road = None
        if self.road_class_ids:
            road_values = [iou_dict[c] for c in self.road_class_ids if c in iou_dict]
            if road_values:
                mean_iou_road = float(np.mean(road_values))

        per_class_json = json.dumps(
            {int(k): float(v) for k, v in iou_dict.items()},
            sort_keys=True,
        )

        return mean_iou_all, mean_iou_road, per_class_json
