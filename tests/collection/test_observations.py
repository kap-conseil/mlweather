import pytest
from polars import DataFrame, Float64, arange
from dotenv import load_dotenv
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from mlweather.collection.observations import Observations

load_dotenv()

# TESTS FOR THE OBSERVATION CLASS ##############################################


class TestObservations:
    def test_observations_init_fields(self):
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

        obs = Observations(lat_lon, elevation, units, record_table)

        assert obs.lat_lon == lat_lon
        assert obs.elevation == elevation
        assert obs.units == units
        assert obs.record_table.equals(record_table)

    def test_collect(self):
        # Collect observations for a given location and time range
        obs = Observations.collect(
            lat_lon=(52.52, 13.41),
            variables_names=["temperature_2m", "precipitation"],
            start=datetime(2021, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
            end=datetime(2021, 1, 1, 1, 0, 0, tzinfo=timezone.utc),
        )

        # Validate the type
        assert isinstance(obs, Observations)
        # Control that datetimes are UTC or None for observations
        assert obs.record_table["valid_datetime"][0].tzinfo == ZoneInfo("UTC")
        assert obs.record_table["init_datetime"][0] is None
        # Check that variables are floats
        assert isinstance(obs.record_table["temperature_2m"].dtype, Float64)
        assert isinstance(obs.record_table["precipitation"].dtype, Float64)
        # Check if regular time grid given the valid_datetime (do not raise an error)
        Observations.is_regular_time(obs.record_table)
        # Check that error is raised if time grid is not regular (second row is dropped for irregularity)
        with pytest.raises(ValueError):
            Observations.is_regular_time(
                obs.record_table.filter(arange(0, obs.record_table.height) != 2)
            )
