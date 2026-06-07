import logging
import time
from typing import Optional

import requests

logger = logging.getLogger(__name__)


def get_with_backoff(
    session: Optional[requests.Session],
    url: str,
    *,
    request_delay: float = 0.1,
    max_retries: int = 5,
    retry_delay: float = 60.0,
    timeout: Optional[float] = None,
    **kwargs,
) -> requests.Response:
    """GET with a small throttle and long waits for Mapillary rate limits."""
    sess = session or requests
    retries = max(0, int(max_retries))
    delay = max(0.0, float(request_delay))
    wait = max(0.0, float(retry_delay))

    for attempt in range(retries + 1):
        if delay > 0:
            time.sleep(delay)

        response = sess.get(url, timeout=timeout, **kwargs)
        if response.status_code != 429:
            return response

        if attempt >= retries:
            return response

        logger.warning(
            "Mapillary rate limit hit (429); waiting %.1fs before retry %d/%d",
            wait,
            attempt + 1,
            retries,
        )
        time.sleep(wait)

    return response
