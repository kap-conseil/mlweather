import pytest
from datetime import datetime, timedelta
import polars as pl
from mlweather.collection.forecasts import Forecasts


class TestValidatePastDayRange:
    def test_valid_range(self):
        assert Forecasts.validate_past_day_range((0, 7)) == (0, 7)
        assert Forecasts.validate_past_day_range((1, 5)) == (1, 5)
        assert Forecasts.validate_past_day_range((0, 0)) == (0, 0)

    def test_invalid_not_tuple(self):
        with pytest.raises(ValueError, match="past_days_range must be a tuple"):
            Forecasts.validate_past_day_range([0, 7])

    def test_invalid_wrong_length(self):
        with pytest.raises(ValueError, match="past_days_range must be a tuple"):
            Forecasts.validate_past_day_range((0, 7, 10))

    def test_invalid_not_integers(self):
        with pytest.raises(ValueError, match="past_days_range must be a tuple"):
            Forecasts.validate_past_day_range((0.5, 7))

    def test_invalid_negative_start(self):
        with pytest.raises(ValueError, match="past_days_range must be a tuple"):
            Forecasts.validate_past_day_range((-1, 7))

    def test_invalid_end_less_than_start(self):
        with pytest.raises(ValueError, match="past_days_range must be a tuple"):
            Forecasts.validate_past_day_range((7, 3))


class TestRenameDayOColumns:
    def test_rename_basic(self):
        df = pl.DataFrame(
            {
                "valid_datetime": [datetime(2024, 1, 1)],
                "temperature": [20.0],
                "humidity": [60.0],
                "temperature_previous_day1": [18.0],
                "humidity_previous_day1": [65.0],
            }
        )
        queried = ["temperature_previous_day1", "humidity_previous_day1"]

        result = Forecasts.rename_day0_columns(df, queried)

        assert "temperature_previous_day0" in result.columns
        assert "humidity_previous_day0" in result.columns
        assert "temperature" not in result.columns
        assert "humidity" not in result.columns

    def test_rename_preserves_valid_datetime(self):
        df = pl.DataFrame(
            {"valid_datetime": [datetime(2024, 1, 1)], "temperature": [20.0]}
        )

        result = Forecasts.rename_day0_columns(df, [])

        assert "valid_datetime" in result.columns


class TestMeltByWeatherVariable:
    def test_melt_single_variable(self):
        df = pl.DataFrame(
            {
                "valid_datetime": [datetime(2024, 1, 1), datetime(2024, 1, 1, 1)],
                "temperature_previous_day0": [20.0, 21.0],
                "temperature_previous_day1": [18.0, 19.0],
            }
        )

        result = Forecasts.melt_by_weather_variable(df)

        assert len(result) == 1
        assert "temperature_previous" in result[0].columns
        assert "past_day" in result[0].columns
        assert result[0].height == 4

    def test_melt_multiple_variables(self):
        df = pl.DataFrame(
            {
                "valid_datetime": [datetime(2024, 1, 1)],
                "temperature_previous_day0": [20.0],
                "temperature_previous_day1": [18.0],
                "humidity_previous_day0": [60.0],
                "humidity_previous_day1": [65.0],
            }
        )

        result = Forecasts.melt_by_weather_variable(df)

        assert len(result) == 2


class TestAddInitialDatetime:
    def test_add_initial_datetime_basic(self):
        df = pl.DataFrame(
            {
                "valid_datetime": [
                    datetime(2024, 1, 5, 12, 0),
                    datetime(2024, 1, 5, 13, 0),
                ],
                "past_day": [0, 1],
                "temperature": [20.0, 18.0],
            }
        )

        result = Forecasts.add_initial_datetime(df)

        assert "init_datetime" in result.columns
        assert "past_day" not in result.columns
        assert result["init_datetime"][0] == datetime(2024, 1, 5, 0, 0)
        assert result["init_datetime"][1] == datetime(2024, 1, 4, 0, 0)

    def test_add_initial_datetime_multiple_days(self):
        df = pl.DataFrame(
            {
                "valid_datetime": [datetime(2024, 1, 10, 15, 0)],
                "past_day": [3],
                "temperature": [15.0],
            }
        )

        result = Forecasts.add_initial_datetime(df)

        assert result["init_datetime"][0] == datetime(2024, 1, 7, 0, 0)


class TestForecastsInit:
    def test_init_valid(self):
        df = pl.DataFrame(
            {"valid_datetime": [datetime(2024, 1, 1)], "temperature": [20.0]}
        )

        forecasts = Forecasts(
            lat_lon=(45.0, -75.0),
            elevation=100.0,
            units={"temperature": "°C"},
            past_days_range=(0, 7),
            record_table=df,
        )

        assert forecasts.lat_lon == (45.0, -75.0)
        assert forecasts.elevation == 100.0
        assert forecasts.past_days_range == (0, 7)

    def test_init_invalid_range(self):
        df = pl.DataFrame(
            {"valid_datetime": [datetime(2024, 1, 1)], "temperature": [20.0]}
        )

        with pytest.raises(ValueError):
            Forecasts(
                lat_lon=(45.0, -75.0),
                elevation=100.0,
                units={"temperature": "°C"},
                past_days_range=(7, 0),
                record_table=df,
            )
