from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from mlweather.collection.utils import to_utc_safe


class TestToUtcSafe:
    """Tests for the to_utc_safe function."""

    def test_naive_datetime(self):
        """Test conversion of naive datetime to UTC."""
        naive_dt = datetime(2024, 1, 1, 12, 0, 0)  # Naive datetime
        utc_dt = to_utc_safe(naive_dt)
        assert utc_dt.tzinfo is not None
        # Validate the tz offset
        assert utc_dt.tzinfo.utcoffset(utc_dt) == timezone.utc.utcoffset(utc_dt)
        assert utc_dt.hour == 12  # Assuming input was in UTC

    def test_aware_datetime_non_utc(self):
        """Test conversion of aware datetime in non-UTC timezone to UTC."""
        cet = ZoneInfo("Europe/Paris")
        aware_dt = datetime(2024, 1, 1, 13, 0, 0, tzinfo=cet)  # CET is UTC+1
        utc_dt = to_utc_safe(aware_dt)
        assert utc_dt.tzinfo is not None
        assert utc_dt.tzinfo.utcoffset(utc_dt) == timezone.utc.utcoffset(utc_dt)
        assert utc_dt.hour == 12  # Converted to UTC

    def test_aware_datetime_utc(self):
        """Test that an already UTC aware datetime remains unchanged."""
        utc_aware_dt = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        utc_dt = to_utc_safe(utc_aware_dt)
        assert utc_dt == utc_aware_dt  # Should be unchanged
