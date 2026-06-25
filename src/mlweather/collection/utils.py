from datetime import datetime, timezone
from zoneinfo import ZoneInfo


# DATE TREAMENT UTILITIES ######################################################
def to_utc_safe(dt: datetime) -> datetime:
    """
    Return a timezone-aware datetime normalized to UTC.

    Behavior:
    - If `dt` is naive (`tzinfo is None`), it is interpreted as UTC.
    - If `dt` is timezone-aware, it is converted to UTC.
    - If `dt.tzinfo` is present but has no UTC offset (`utcoffset(dt) is None`),
      a `ValueError` is raised.

    Args:
        dt (datetime): Datetime to normalize.

    Returns:
        datetime: A timezone-aware datetime in UTC.

    Raises:
        ValueError: If `dt.tzinfo` is invalid and returns no UTC offset.
    """
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=ZoneInfo("UTC"))

    if dt.tzinfo is not None and dt.tzinfo.utcoffset(dt) is None:
        raise ValueError("Invalid tzinfo with no utcoffset")

    return dt.astimezone(timezone.utc)
