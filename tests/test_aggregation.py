from datetime import timezone
import pytest
from polars import col, Expr
from requests_cache import datetime, timedelta

from mlweather.collection.observations import Observations
from mlweather.collection.forecasts import Forecasts
from mlweather.aggregation import Aggregation, FeatureGenerator


class TestAggregation:
    def test_aggregation_init(self):
        # Definition of the operation for aggregation case
        col_name = "precipitation"
        base_exp = col(col_name)
        base_exp = base_exp.sum()
        # Declaration of the periods (observations and forecasts)
        obs_period = timedelta(hours=3)  # 3 hours
        fc_period = timedelta(hours=6)  # 6 hours

        Aggregation(
            expression=base_exp,
            observation_period=obs_period,
            forecast_period=fc_period,
        )

    def test_get_steps_in_agg_periods(self):
        # Check the number of steps in aggregation periods
        assert Aggregation.get_steps_in_agg_periods(timedelta(hours=24)) == 24
        # Error if negative or zero period
        with pytest.raises(ValueError) as excinfo:
            Aggregation.get_steps_in_agg_periods(timedelta(hours=0))
        assert "must be greater than zero" in str(excinfo.value)


class TestFeatureGenerator:
    def test_feature_generator_init(self):
        # Definition of aggregation operations
        agg1 = Aggregation(
            expression=col("temperature_2m").mean(),
            observation_period=timedelta(hours=6),
            forecast_period=None,
        )
        agg2 = Aggregation(
            expression=col("precipitation").sum(),
            observation_period=None,
            forecast_period=timedelta(hours=12),
        )

        fg = FeatureGenerator(aggregations=[agg1, agg2])

        assert len(fg.aggregations) == 2
        assert isinstance(fg.aggregations[0], Aggregation)
        assert isinstance(fg.aggregations[1], Aggregation)

    def test_feature_generator_invalid_init(self):
        # Should raise if aggregation is an empty list
        with pytest.raises(ValueError) as excinfo:
            FeatureGenerator(aggregations=[])
        assert "aggregations cannot be an empty list" in str(excinfo.value)

    def test__prepare_all_records(self):
        # Definition of aggregation operations
        agg1 = Aggregation(
            expression=col("temperature_2m").mean(),
            observation_period=timedelta(hours=6),
            forecast_period=None,
        )
        fg = FeatureGenerator(aggregations=[agg1])

        # Prepare dummy observations and forecasts
        start_dt = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        end_dt = start_dt + timedelta(days=30)
        obs = Observations.collect(
            lat_lon=(48.0, -2.0),
            variables_names=["temperature_2m"],
            start=start_dt,
            end=end_dt,
        )

        fore = Forecasts.collect(
            lat_lon=(48.0, -2.0),
            variables_names=["temperature_2m"],
            start=start_dt,
            end=end_dt,
            forecast_horizon_days_range=(0, 7),
        )
