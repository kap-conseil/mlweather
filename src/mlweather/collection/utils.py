from datetime import datetime, timezone
from zoneinfo import ZoneInfo


# DATE TREAMENT UTILITIES ######################################################
def to_utc_safe(dt: datetime, assume_tz=ZoneInfo("UTC")) -> datetime:
    """
    Convert a naive datetime to a UTC-aware datetime by assuming a timezone (as default).
    If the datetime is already timezone-aware, converrt to UTC in accordance to original TZ.
    Args:
        dt (datetime): The datetime object to convert.
        assume_tz (ZoneInfo): The timezone to assume for naive datetime objects.
    Returns:
        datetime: A timezone aware datetime object.
    """
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=assume_tz)

        return dt.astimezone(timezone.utc)
    elif dt.tzinfo is not None and dt.tzinfo.utcoffset(dt) is None:
        raise ValueError("Invalid tzinfo with no utcoffset")
    else:
        return dt.astimezone(timezone.utc)
