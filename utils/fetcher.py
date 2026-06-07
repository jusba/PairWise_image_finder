import logging
import os
import csv
import random
from io import BytesIO
from typing import Dict, Iterator, Optional
import requests
from PIL import Image
import pandas as pd

from .http_utils import get_with_backoff
from .io_utils import ensure_manifest, load_completed_pairs_from_manifest

logger = logging.getLogger(__name__)

IMAGE_SIZES = ("256", "1024", "2048", "original")
_SIZE_FIELD = {s: f"thumb_{s}_url" for s in IMAGE_SIZES}


def fetch_image_url(
    image_id: str,
    access_token: str,
    size: str = "1024",
    session: Optional[requests.Session] = None,
    request_delay: float = 0.1,
    max_retries: int = 5,
    retry_delay: float = 60.0,
) -> Optional[str]:
    """Return the downloadable URL for a Mapillary image. size: 256|1024|2048|original."""
    if size not in _SIZE_FIELD:
        raise ValueError(f"size must be one of {IMAGE_SIZES}, got {size!r}")
    try:
        sess = session or requests.Session()
        field = _SIZE_FIELD[size]
        url = f"https://graph.mapillary.com/{image_id}?fields={field}&access_token={access_token}"
        r = get_with_backoff(
            sess,
            url,
            timeout=10,
            request_delay=request_delay,
            max_retries=max_retries,
            retry_delay=retry_delay,
        )
        r.raise_for_status()
        return r.json().get(field)
    except Exception as e:
        logger.warning("url failed %s: %s", image_id, e)
        return None


def fetch_and_cache_image(
    image_id: str,
    access_token: str,
    images_dir: str,
    session: Optional[requests.Session] = None,
    size: str = "1024",
    request_delay: float = 0.1,
    max_retries: int = 5,
    retry_delay: float = 60.0,
) -> Optional[str]:
    """
    Fetch a Mapillary image by ID and cache it to disk.
    If already downloaded, returns the existing path without a network call.

    Args:
        image_id:    Mapillary image ID
        access_token: API token
        images_dir:  Directory to store images
        session:     Optional requests Session for connection reuse
        size:        Image size: "256", "1024", "2048", or "original" (default: "1024")

    Returns:
        Local file path, or None on failure.
    """
    try:
        os.makedirs(images_dir, exist_ok=True)
        out_path = os.path.join(images_dir, f"{image_id}.png")
        if os.path.exists(out_path):
            return out_path
        url = fetch_image_url(
            image_id,
            access_token,
            size=size,
            session=session,
            request_delay=request_delay,
            max_retries=max_retries,
            retry_delay=retry_delay,
        )
        if not url:
            return None
        sess = session or requests.Session()
        resp = get_with_backoff(
            sess,
            url,
            timeout=30,
            request_delay=request_delay,
            max_retries=max_retries,
            retry_delay=retry_delay,
        )
        resp.raise_for_status()
        img = Image.open(BytesIO(resp.content)).convert("RGB")
        img.save(out_path)
        return out_path
    except Exception as e:
        logger.warning("fetch&cache failed %s: %s", image_id, e)
        return None


def fetch_image(
    image_id: str,
    session: requests.Session,
    access_token: str,
    size: str = "1024",
    request_delay: float = 0.1,
    max_retries: int = 5,
    retry_delay: float = 60.0,
) -> Optional[Image.Image]:
    """Fetch and return a PIL Image from Mapillary. size: 256|1024|2048|original."""
    try:
        field = _SIZE_FIELD.get(size, "thumb_1024_url")
        url = f"https://graph.mapillary.com/{image_id}?fields={field}&access_token={access_token}"
        r = get_with_backoff(
            session,
            url,
            timeout=5,
            request_delay=request_delay,
            max_retries=max_retries,
            retry_delay=retry_delay,
        )
        r.raise_for_status()
        image_url = r.json().get(field)
        if not image_url:
            return None
        img_resp = get_with_backoff(
            session,
            image_url,
            timeout=10,
            request_delay=request_delay,
            max_retries=max_retries,
            retry_delay=retry_delay,
        )
        img_resp.raise_for_status()
        return Image.open(BytesIO(img_resp.content))
    except Exception as e:
        logger.warning("fetch_image failed %s: %s", image_id, e)
        return None

def build_pairs(data_dict: Dict) -> list[tuple[str, str, Dict, Dict]]:
    pairs = []
    for _, entry in data_dict.items():
        left = entry['point_info']
        for neighbor in entry['conflicting_neighbors']:
            id_left = str(left['id_left'])
            id_right = str(neighbor['id'])
            pairs.append((id_left, id_right, left, neighbor))
    random.shuffle(pairs)
    return pairs

def download_pair_to_disk(
    session,
    access_token,
    id_left,
    id_right,
    left_info,
    right_info,
    base_dir,
    request_delay: float = 0.1,
    max_retries: int = 5,
    retry_delay: float = 60.0,
):
    subdir = id_left[:2]
    folder = os.path.join(base_dir, subdir, f"{id_left}_{id_right}")
    os.makedirs(folder, exist_ok=True)
    left_path = os.path.join(folder, f"{id_left}.png")
    right_path = os.path.join(folder, f"{id_right}.png")

    # If already on disk, skip network
    if os.path.exists(left_path) and os.path.exists(right_path):
        return {
            "id_left": id_left,
            "date_left": left_info.get("captured_at_left", "N/A"),
            "id_right": id_right,
            "date_right": right_info.get("captured_at", "N/A"),
            "left_path": left_path,
            "right_path": right_path,
        }

    left_img = fetch_image(
        id_left,
        session,
        access_token,
        request_delay=request_delay,
        max_retries=max_retries,
        retry_delay=retry_delay,
    )
    right_img = fetch_image(
        id_right,
        session,
        access_token,
        request_delay=request_delay,
        max_retries=max_retries,
        retry_delay=retry_delay,
    )
    if left_img is None or right_img is None:
        return None

    left_img.convert("RGB").save(left_path)
    right_img.convert("RGB").save(right_path)

    return {
        "id_left": id_left,
        "date_left": left_info.get("captured_at_left", "N/A"),
        "id_right": id_right,
        "date_right": right_info.get("captured_at", "N/A"),
        "left_path": left_path,
        "right_path": right_path,
    }


def stream_downloaded_pairs(
    data_dict,
    access_token,
    base_dir,
    manifest_csv: Optional[str] = None,
    skip_completed_manifest: bool = True,
    request_delay: float = 0.1,
    max_retries: int = 5,
    retry_delay: float = 60.0,
) -> Iterator[Dict]:
    if manifest_csv:
        ensure_manifest(manifest_csv)
        completed = load_completed_pairs_from_manifest(manifest_csv) if skip_completed_manifest else set()
        manifest_f = open(manifest_csv, 'a', newline='')
        manifest_writer = csv.writer(manifest_f)
    else:
        completed = set()
        manifest_f = None
        manifest_writer = None

    try:
        with requests.Session() as session:
            for (id_left, id_right, left_info, right_info) in build_pairs(data_dict):
                if (id_left, id_right) in completed:
                    continue
                row = download_pair_to_disk(
                    session,
                    access_token,
                    id_left,
                    id_right,
                    left_info,
                    right_info,
                    base_dir,
                    request_delay=request_delay,
                    max_retries=max_retries,
                    retry_delay=retry_delay,
                )
                if row is None:
                    continue
                if manifest_writer:
                    manifest_writer.writerow([
                        row["id_left"], row["date_left"], row["id_right"], row["date_right"],
                        row["left_path"], row["right_path"]
                    ])
                    manifest_f.flush()
                    os.fsync(manifest_f.fileno())
                yield row
    finally:
        if manifest_f:
            manifest_f.close()
