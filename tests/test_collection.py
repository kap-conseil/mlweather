from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from polars import DataFrame, col

from mlweather.collection import (
    Observations,
    Measure,
    ADMISSIBLE_MEASURES,
    to_utc_safe,
    prepare_hourly_raw_observations,
)


def test_Measure():
    measure = Measure("precipitation", "mm", float, "Preceding hour sum")
    assert measure.name == "precipitation"
    assert measure.unit == "mm"
    assert measure.value_type is float
    assert measure.valid_time == "Preceding hour sum"


def test_admissible_measures():
    assert ADMISSIBLE_MEASURES[0].name == "precipitation"


def test_to_utc_safe():
    dt = datetime(2023, 10, 1, 12, 0, 0)
    assert to_utc_safe(dt) == datetime(2023, 10, 1, 12, 0, tzinfo=timezone.utc)

    dt_with_tz = dt.replace(tzinfo=ZoneInfo("Europe/Paris"))
    assert to_utc_safe(dt_with_tz) == datetime(2023, 10, 1, 10, 0, tzinfo=timezone.utc)


def test_prepare_hourly_raw_observations():
    hourly_dict = {
        "time": ["2023-10-01T00:00", "2023-10-01T01:00"],
        "precipitation": [0.0, 1.2],
        "temperature_2m": [15.0, 14.5],
    }

    df = prepare_hourly_raw_observations(hourly_dict)

    assert isinstance(df, DataFrame)
    assert df.columns == ["dt_target", "precipitation", "temperature_2m"]
    assert (
        df["dt_target"].dtype
        == DataFrame(hourly_dict)
        .with_columns(col("time").str.to_datetime(format="%Y-%m-%dT%H:%M"))["time"]
        .dtype
    )


def test_Observations():
    # Check if the Observations class can evaluate the regularity
    assert Observations.is_regular_time(
        prepare_hourly_raw_observations(
            DataFrame(
                {
                    "time": ["2023-10-01T00:00", "2023-10-01T01:00"],
                    "precipitation": [0.0, 1.2],
                    "temperature_2m": [15.0, 14.5],
                }
            )
        )
    )
    # without caching
    obs = Observations.get_obs(
        (0.0, 0.0),
        ["precipitation", "temperature_2m"],
        datetime(2023, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
        datetime(2023, 12, 31, 0, 0, 0, tzinfo=timezone.utc),
        verbose=True,
        caching=False,
    )

    # with caching
    obs = Observations.get_obs(
        (0.0, 0.0),
        ["precipitation", "temperature_2m"],
        datetime(2023, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
        datetime(2023, 12, 31, 0, 0, 0, tzinfo=timezone.utc),
        verbose=True,
        caching=True,
    )

    assert isinstance(obs, Observations)
    assert obs.values is not None
    assert "dt_target" in obs.values.columns
    assert "precipitation" in obs.values.columns
    assert "temperature_2m" in obs.values.columns
    assert obs.values.shape[0] == (365 * 24)  # At least one year of hourly data
