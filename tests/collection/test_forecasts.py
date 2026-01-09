import pytest
from polars import DataFrame, Float64, arange
from dotenv import load_dotenv
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from mlweather.collection.forecasts import Forecasts

load_dotenv()

# TESTS FOR THE FORECAST CLASS #################################################


class TestForecasts:
    def test_forecasts_init_fields(self):
        lat_lon = (50.0, 8.0)
        elevation = 100.0
        units = {"temperature_2m": "°C", "precipitation": "mm"}
        record_table = DataFrame(
            {
                "dt_target": ["2023-10-01T00:00", "2023-10-01T01:00"],
                "temperature_2m": [15.0, 14.5],
                "precipitation": [0.0, 1.2],
            }
        )

        fore = Forecasts(lat_lon, elevation, units, 2, record_table)
        assert fore.lat_lon == lat_lon
        assert fore.elevation == elevation
        assert fore.units == units
        assert fore.record_table.equals(record_table)

    def test_collect(self):
        # Collect observations for a given location and time range
        fore = Forecasts.collect(
            lat_lon=(52.52, 13.41),
            variables_names=["temperature_2m", "precipitation"],
            forecast_horizon_days_max=2,
            start=datetime(2024, 2, 1, 0, 0, 0, tzinfo=timezone.utc),
            end=datetime(2024, 3, 1, 1, 0, 0, tzinfo=timezone.utc),
        )

        # Valid the type
        assert isinstance(fore, Forecasts)
        # Check columns presence
        expected_columns = [
            "valid_datetime",
            "init_datetime",
            # "days_forecast_horizon",
            "temperature_2m",
            "precipitation",
        ]
        for col_name in expected_columns:
            assert col_name in fore.record_table.columns
        # Control that datetimes are UTC or None for observations
        assert fore.record_table["valid_datetime"][0].tzinfo == ZoneInfo("UTC")
        assert fore.record_table["init_datetime"][0].tzinfo == ZoneInfo("UTC")
        # Check that variables are floats
        assert isinstance(fore.record_table["temperature_2m"].dtype, Float64)
        assert isinstance(fore.record_table["precipitation"].dtype, Float64)
        # Check that 3 records per valid_datetime are present for day 0,1,2
        count_per_valid_dt = (
            (fore.record_table.group_by("valid_datetime").len())["len"]
            .unique()
            .to_list()
        )
        assert count_per_valid_dt == [3]
        # Check if regular time grid given the valid_datetime (do not raise an error)
        Forecasts.is_regular_time(fore.record_table)

        # Check that error is raised if time grid is not regular
        with pytest.raises(ValueError):
            Forecasts.is_regular_time(
                fore.record_table.filter(arange(0, fore.record_table.height) != 2)
            )
