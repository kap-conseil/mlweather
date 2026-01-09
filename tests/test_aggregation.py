from datetime import timezone, datetime, timedelta
import os
import pytest
from polars import arange, col, Expr, lit, when

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
            Aggregation.get_steps_in_agg_periods(timedelta(hours=-1))
        assert "must be equal or greater than zero" in str(excinfo.value)


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

    def test_generate_features(self):
        # Collect observations and forecasts for testing
        start_dt = datetime(2024, 4, 1, 0, 0, 0, tzinfo=timezone.utc)
        end_dt = start_dt + timedelta(days=50)
        loc = (48.0, -2.0)
        api_key = os.getenv("OPENMETEO_API_KEY")
        vars = [
            "precipitation",
            "temperature_2m",
        ]

        # Collect data
        obs = Observations.collect(
            loc,
            vars,
            start_dt,
            end_dt,
            api_key=api_key,
            verbose=False,
        )
        fore = Forecasts.collect(
            loc,
            vars,
            start_dt,
            end_dt,
            7,
            api_key=api_key,
        )

        # Check that they have the same valid_datetimes
        assert (obs.record_table["valid_datetime"].unique().sort()).equals(
            (fore.record_table["valid_datetime"].unique().sort())
        )

        # Replace if not null value by 1.00 for precipitation to test the aggregation
        obs.record_table = obs.record_table.with_columns(
            when(col("precipitation").is_not_null())
            .then(lit(1.0))
            .otherwise(col("precipitation"))
            .alias("precipitation")
        )

        fore.record_table = fore.record_table.with_columns(
            when(col("precipitation").is_not_null())
            .then(lit(1.0))
            .otherwise(col("precipitation"))
            .alias("precipitation")
        )

        # Generate a range from  2024, 2, 20, 0, 0 to 2024, 7, 13, 0, 0 (daily)
        features_datetimes = obs.record_table.filter(
            col("valid_datetime").dt.minute() == 0
        )["valid_datetime"].to_list()

        # 3 cases tested in termes of aggregation: observations only, forecasts only, both
        # 1/ case: observations only -------------------------------------------
        # init the feature generator
        featgen_obs_only = FeatureGenerator(
            [
                Aggregation(
                    col("precipitation").sum(), observation_period=timedelta(hours=2)
                ),
            ]
        )

        features_obs_only = featgen_obs_only.generate_features(
            features_datetimes,
            observations=obs,
            control_aggregations=False,
        )

        # Check features values
        # missing values should be there because of the 2-hour aggregation: first hour has not aggregation
        assert features_obs_only["precipitation_sum_obs_7200s"].is_null().sum() == (
            Aggregation.get_steps_in_agg_periods(timedelta(hours=2)) - 1
        )
        # Check values : since precipitation values were replaced by 1.0, the sum over 2 hours should be 2.0
        assert features_obs_only.filter(
            col("precipitation_sum_obs_7200s").is_not_null()
        )["precipitation_sum_obs_7200s"].unique().to_list() == [2.0]

        # 2/ case: forecast only -----------------------------------------------
        # init the feature generator
        features_fore_only = FeatureGenerator(
            [
                Aggregation(
                    col("precipitation").sum(), forecast_period=timedelta(hours=2)
                ),
            ]
        )

        features_fore_only = features_fore_only.generate_features(
            features_datetimes,
            forecasts=fore,
            control_aggregations=False,
        )

        # Check features values
        # missing values should be there because of the 2-hour aggregation: first hour has not aggregation
        assert features_fore_only["precipitation_sum_fore_7200s"].is_null().sum() == (
            Aggregation.get_steps_in_agg_periods(timedelta(hours=2)) - 1
        )
        # # Check values : since precipitation values were replaced by 1.0, the sum over 2 hours should be 2.0
        # assert features_fore_only.filter(
        #     col("precipitation_sum_fore_7200s").is_not_null()
        # )["precipitation_sum_fore_7200s"].unique().to_list() == [2.0]
