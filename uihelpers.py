"""
Small helpers for controls whose bounds depend on live data.

A slider's minimum must be below its maximum, but list lengths change: the
commodities list is three entries, the crypto list is ninety-odd. Working the
bounds out here keeps that logic testable instead of hidden in page code, where
an empty or very short list becomes a crash.
"""

from typing import Optional, Tuple


def scan_count(available: int, default: int = 10, minimum: int = 3
               ) -> Tuple[Optional[int], Optional[int], int]:
    """Bounds for a "how many to scan" slider.

    Returns (slider_min, slider_max, value). Both bounds are None when a
    slider makes no sense — nothing to scan, or so few that the only sensible
    answer is "all of them" — and the caller shows a caption instead.
    """
    if available <= 0:
        return None, None, 0
    if available <= minimum:
        return None, None, available
    return minimum, available, max(minimum, min(default, available))
