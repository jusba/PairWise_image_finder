"""Shared utilities with lazy exports for optional ML dependencies."""

_EXPORTS = {
    "build_pairs": ("fetcher", "build_pairs"),
    "fetch_image": ("fetcher", "fetch_image"),
    "download_pair_to_disk": ("fetcher", "download_pair_to_disk"),
    "stream_downloaded_pairs": ("fetcher", "stream_downloaded_pairs"),
    "process_one_pair": ("processing", "process_one_pair"),
    "compute_lightglue_score": ("processing", "compute_lightglue_score"),
    "pick_device": ("models", "pick_device"),
    "init_models": ("models", "init_models"),
    "init_segmenter": ("models", "init_segmenter"),
    "estimate_horizon_angle": ("metrics", "estimate_horizon_angle"),
    "compute_image_quality_metrics": ("metrics", "compute_image_quality_metrics"),
    "resize_to_smallest": ("metrics", "resize_to_smallest"),
    "CSV_HEADER": ("io_utils", "CSV_HEADER"),
    "CSV_HEADER_PANORAMA": ("io_utils", "CSV_HEADER_PANORAMA"),
    "CSV_HEADER_SCALE": ("io_utils", "CSV_HEADER_SCALE"),
    "CSV_HEADER_AREA": ("io_utils", "CSV_HEADER_AREA"),
    "CSV_HEADER_PANORAMA_AREA": ("io_utils", "CSV_HEADER_PANORAMA_AREA"),
    "CSV_HEADER_SCALE_AREA": ("io_utils", "CSV_HEADER_SCALE_AREA"),
    "ensure_manifest": ("io_utils", "ensure_manifest"),
    "load_completed_pairs_from_manifest": ("io_utils", "load_completed_pairs_from_manifest"),
    "completed_pairs_set": ("io_utils", "completed_pairs_set"),
    "iter_manifest_rows": ("io_utils", "iter_manifest_rows"),
    "iter_folder_rows": ("io_utils", "iter_folder_rows"),
    "load_image": ("tensor_utils", "load_image"),
    "fetch_images_in_area": ("area_pairing", "fetch_images_in_area"),
    "pair_by_proximity": ("area_pairing", "pair_by_proximity"),
    "compute_pair_metadata": ("area_pairing", "compute_pair_metadata"),
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _EXPORTS[name]
    from importlib import import_module

    value = getattr(import_module(f".{module_name}", __name__), attr_name)
    globals()[name] = value
    return value
