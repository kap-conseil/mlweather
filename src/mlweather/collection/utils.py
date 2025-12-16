from datetime import datetime, timezone
from zoneinfo import ZoneInfo


# DATE TREAMENT UTILITIES ######################################################
def to_utc_safe(dt: datetime, assume_tz=ZoneInfo("UTC")) -> datetime:
    if dt.tzinfo is None or dt.tzinfo.utcoffset(dt) is None:
        dt = dt.replace(tzinfo=assume_tz)

    return dt.astimezone(timezone.utc)
