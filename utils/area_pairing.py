"""
Mapillary area querying via the vector tile API, and image pairing by proximity.

Used when --area-wkt is specified. Requires: shapely, mercantile, mapbox-vector-tile.
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

from .http_utils import get_with_backoff

logger = logging.getLogger(__name__)

_TILE_URL = "https://tiles.mapillary.com/maps/vtp/mly1_public/2/{z}/{x}/{y}"
_TILE_ZOOM = 14       # Mapillary image data is stored at zoom 14
_TILE_EXTENT = 4096   # standard MVT tile coordinate extent


# ---------- Geometry helpers ----------

def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in metres between two GPS points."""
    R = 6_371_000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def _angle_diff(a1: float, a2: float) -> float:
    """Smallest absolute difference between two compass angles in [0, 360)."""
    diff = abs(a1 - a2) % 360
    return min(diff, 360 - diff)


def _tile_px_to_wgs84(px: float, py: float, tile_bounds) -> Tuple[float, float]:
    """
    Convert tile-local pixel coordinates (0..EXTENT) to WGS84 (lon, lat).
    y=0 is the north edge of the tile.
    """
    lon = tile_bounds.west + (px / _TILE_EXTENT) * (tile_bounds.east - tile_bounds.west)
    lat = tile_bounds.north - (py / _TILE_EXTENT) * (tile_bounds.north - tile_bounds.south)
    return lon, lat


# ---------- Mapillary tile fetch ----------

def fetch_images_in_area(
    wkt_str: str,
    access_token: str,
    image_type: str = "all",
    request_delay: float = 0.1,
    max_retries: int = 5,
    retry_delay: float = 60.0,
) -> pd.DataFrame:
    """
    Fetch Mapillary image metadata for all images inside a WKT polygon.

    Uses the Mapillary vector tile API (zoom 14) instead of the Graph API bbox
    endpoint, which requires elevated API access. Tiles are publicly available
    with any valid access token.

    Args:
        wkt_str:      WKT polygon, e.g. "POLYGON ((lon lat, ...))"
        access_token: Mapillary API token
        image_type:   "all" | "panorama" | "flat"

    Returns:
        DataFrame with columns: id, lat, lon, compass_angle, is_pano, captured_at
    """
    try:
        from shapely import wkt as shapely_wkt
        from shapely.geometry import Point
    except ImportError:
        raise ImportError(
            'shapely is required for area mode. Install area dependencies with: pip install ".[area]"'
        )

    try:
        import mercantile
    except ImportError:
        raise ImportError(
            'mercantile is required for area mode. Install area dependencies with: pip install ".[area]"'
        )

    try:
        import mapbox_vector_tile
    except ImportError:
        raise ImportError(
            'mapbox-vector-tile is required for Mapillary vector tiles. '
            'Install area dependencies with: pip install ".[area]"'
        )

    polygon = shapely_wkt.loads(wkt_str)
    min_lon, min_lat, max_lon, max_lat = polygon.bounds

    tiles = list(mercantile.tiles(min_lon, min_lat, max_lon, max_lat, zooms=_TILE_ZOOM))
    logger.info(
        "Fetching %d tile(s) at zoom %d for bbox (%.4f,%.4f,%.4f,%.4f)",
        len(tiles), _TILE_ZOOM, min_lon, min_lat, max_lon, max_lat,
    )

    rows: List[dict] = []
    seen_ids: set = set()

    for tile in tqdm(tiles, desc="tiles", unit="tile"):
        url = _TILE_URL.format(z=tile.z, x=tile.x, y=tile.y)
        resp = get_with_backoff(
            None,
            url,
            params={"access_token": access_token},
            timeout=30,
            request_delay=request_delay,
            max_retries=max_retries,
            retry_delay=retry_delay,
        )
        if not resp.ok:
            logger.warning(
                "Tile %d/%d/%d failed: %d %s",
                tile.z, tile.x, tile.y, resp.status_code, resp.text[:200],
            )
            continue

        try:
            tile_data = mapbox_vector_tile.decode(resp.content)
        except Exception as e:
            logger.warning("Failed to decode tile %d/%d/%d: %s", tile.z, tile.x, tile.y, e)
            continue

        image_layer = tile_data.get("image", {})
        features = image_layer.get("features", [])
        bounds = mercantile.bounds(tile)

        for feature in features:
            props = feature.get("properties", {})
            img_id = str(props.get("id", ""))
            if not img_id or img_id in seen_ids:
                continue

            geom = feature.get("geometry", {})
            if geom.get("type") != "Point":
                continue
            coords = geom.get("coordinates", [])
            if len(coords) < 2:
                continue

            lon, lat = _tile_px_to_wgs84(coords[0], coords[1], bounds)

            if not Point(lon, lat).within(polygon):
                continue

            is_pano = bool(props.get("is_pano", False))
            if image_type == "panorama" and not is_pano:
                continue
            if image_type == "flat" and is_pano:
                continue

            seen_ids.add(img_id)
            rows.append({
                "id": img_id,
                "lat": lat,
                "lon": lon,
                "compass_angle": props.get("compass_angle"),
                "is_pano": is_pano,
                "captured_at": props.get("captured_at"),
                "tile_z": tile.z,
                "tile_x": tile.x,
                "tile_y": tile.y,
            })

    df = (
        pd.DataFrame(rows)
        if rows
        else pd.DataFrame(
            columns=[
                "id",
                "lat",
                "lon",
                "compass_angle",
                "is_pano",
                "captured_at",
                "tile_z",
                "tile_x",
                "tile_y",
            ]
        )
    )
    logger.info("Found %d images inside polygon (image_type=%s)", len(df), image_type)
    return df


# ---------- Time-difference filtering ----------

VALID_TIME_FILTERS = {"any", "year", "month", "season", "same-season", "time"}


def _parse_captured_at(val) -> Optional[datetime]:
    """Convert Mapillary captured_at (Unix ms integer) to UTC datetime, or None."""
    if val is None:
        return None
    try:
        return datetime.fromtimestamp(int(val) / 1000, tz=timezone.utc)
    except (ValueError, TypeError):
        return None


def _get_season(dt: datetime) -> str:
    m = dt.month
    if m in (3, 4, 5):
        return "spring"
    if m in (6, 7, 8):
        return "summer"
    if m in (9, 10, 11):
        return "autumn"
    return "winter"  # 12, 1, 2


def _is_daytime(dt: datetime, lat: float, lon: float) -> bool:
    """Return True if dt falls between sunrise and sunset at (lat, lon)."""
    try:
        from astral import LocationInfo
        from astral.sun import sun
    except ImportError:
        raise ImportError(
            'astral is required for --time-filter time. '
            'Install area dependencies with: pip install ".[area]"'
        )
    loc = LocationInfo(latitude=lat, longitude=lon)
    s = sun(loc.observer, date=dt.date(), tzinfo=timezone.utc)
    return s["sunrise"] <= dt <= s["sunset"]


def _passes_time_filter(filters: List[str], left_meta: dict, right_meta: dict) -> bool:
    """
    Return True if the pair satisfies all active time filters (AND logic).
    'any' is a no-op. If either timestamp is missing, the pair is included.
    """
    active = [f for f in filters if f != "any"]
    if not active:
        return True

    dt_left = _parse_captured_at(left_meta.get("captured_at"))
    dt_right = _parse_captured_at(right_meta.get("captured_at"))
    if dt_left is None or dt_right is None:
        return True

    for f in active:
        if f == "year":
            if dt_left.year == dt_right.year:
                return False
        elif f == "month":
            if (dt_left.year, dt_left.month) == (dt_right.year, dt_right.month):
                return False
        elif f == "season":
            if _get_season(dt_left) == _get_season(dt_right):
                return False
        elif f == "same-season":
            if dt_left.year == dt_right.year:
                return False
            if _get_season(dt_left) != _get_season(dt_right):
                return False
        elif f == "time":
            lat = (left_meta["lat"] + right_meta["lat"]) / 2
            lon = (left_meta["lon"] + right_meta["lon"]) / 2
            if _is_daytime(dt_left, lat, lon) == _is_daytime(dt_right, lat, lon):
                return False
    return True


def _passes_year_groups(
    left_meta: dict,
    right_meta: dict,
    year_group_left: Optional[set[int]],
    year_group_right: Optional[set[int]],
) -> bool:
    """Return True when pair years are split across the two configured groups."""
    if not year_group_left and not year_group_right:
        return True
    if not year_group_left or not year_group_right:
        return True

    dt_left = _parse_captured_at(left_meta.get("captured_at"))
    dt_right = _parse_captured_at(right_meta.get("captured_at"))
    if dt_left is None or dt_right is None:
        return False

    left_year = dt_left.year
    right_year = dt_right.year
    return (
        left_year in year_group_left and right_year in year_group_right
    ) or (
        left_year in year_group_right and right_year in year_group_left
    )


# ---------- Pairing ----------

def pair_by_proximity(
    df: pd.DataFrame,
    max_distance_m: float = 2.5,
    max_angle_diff: Optional[float] = None,
    time_filters: Optional[List[str]] = None,
    year_group_left: Optional[List[int]] = None,
    year_group_right: Optional[List[int]] = None,
) -> List[Tuple[dict, dict]]:
    """
    Return all unordered image pairs within max_distance_m of each other,
    optionally filtered by compass-angle difference and time-difference rules.

    Uses an equirectangular projection + vectorised numpy.
    Handles datasets up to ~10 000 images efficiently.

    Args:
        df:              DataFrame from fetch_images_in_area
        max_distance_m:  Max separation in metres (default 2.5)
        max_angle_diff:  Max compass-angle difference in degrees (None = no filter)
        time_filters:    List of time-difference rules to apply (AND logic).
                         Valid values: "any" (no-op), "year", "month", "season",
                         "same-season", "time".
        year_group_left/year_group_right:
                         Optional year sets. When both are provided, only pairs
                         with one image in each set are returned.

    Returns:
        List of (left_row_dict, right_row_dict) pairs.
    """
    if len(df) < 2:
        return []

    lats = df["lat"].values
    lons = df["lon"].values
    rows = df.to_dict("records")
    n = len(rows)

    lat_c = math.radians(float(lats.mean()))
    R = 6_371_000
    x = lons * (math.pi / 180.0) * R * math.cos(lat_c)
    y = lats * (math.pi / 180.0) * R

    active_time = [f for f in (time_filters or []) if f != "any"]
    left_years = set(year_group_left or [])
    right_years = set(year_group_right or [])
    result: List[Tuple[dict, dict]] = []

    for i in tqdm(range(n - 1), desc="pair candidates", unit="image"):
        dx = x[i + 1:] - x[i]
        dy = y[i + 1:] - y[i]
        dists = np.sqrt(dx * dx + dy * dy)
        close_indices = np.where(dists <= max_distance_m)[0]

        for rel in close_indices:
            j = i + 1 + int(rel)

            if max_angle_diff is not None:
                a1 = rows[i].get("compass_angle")
                a2 = rows[j].get("compass_angle")
                if a1 is not None and a2 is not None:
                    if _angle_diff(float(a1), float(a2)) > max_angle_diff:
                        continue

            if active_time and not _passes_time_filter(active_time, rows[i], rows[j]):
                continue
            if not _passes_year_groups(rows[i], rows[j], left_years, right_years):
                continue

            result.append((rows[i], rows[j]))

    time_desc = f", time_filter={active_time}" if active_time else ""
    year_group_desc = (
        f", year_groups={sorted(left_years)} vs {sorted(right_years)}"
        if left_years and right_years
        else ""
    )
    logger.info(
        "Paired %d image pairs (max_dist=%.1f m%s%s%s)",
        len(result),
        max_distance_m,
        f", max_angle_diff={max_angle_diff}°" if max_angle_diff is not None else "",
        time_desc,
        year_group_desc,
    )
    return result


def compute_pair_metadata(left_meta: dict, right_meta: dict) -> dict:
    """Return distance (metres) and compass angle difference for a row pair."""
    dist = _haversine_m(
        left_meta["lat"], left_meta["lon"],
        right_meta["lat"], right_meta["lon"],
    )
    a1 = left_meta.get("compass_angle")
    a2 = right_meta.get("compass_angle")
    angle = _angle_diff(float(a1), float(a2)) if (a1 is not None and a2 is not None) else None
    return {
        "dist_m": round(dist, 3),
        "angle_diff_deg": round(angle, 2) if angle is not None else None,
    }
