from polars import col, DataFrame
from datetime import timedelta, datetime, timezone
from mlweather.collection import Observations, prepare_hourly_raw_observations
from mlweather.aggregation import AggConfig, aggregate, get_dur_step


def test_AggConfig():
    agg_config = AggConfig(col("temperature_2m").mean(), timedelta(days=1))
    # assert agg_config.operation == col("temperature_2m").mean()
    assert agg_config.period == timedelta(days=1)
    assert agg_config.get_var_label() == "temperature_2m_mean_1d"


def test_get_dur_step():
    df = DataFrame(
        {
            "time": ["2023-10-01T00:00", "2023-10-01T01:00"],
            "precipitation": [0.0, 1.2],
            "temperature_2m": [15.0, 14.5],
        }
    )
    assert get_dur_step(prepare_hourly_raw_observations(df)["dt_target"]) == timedelta(
        hours=1
    )


def test_aggregate():
    obs = Observations.get_obs(
        (0.0, 0.0),
        ["precipitation", "temperature_2m"],
        datetime(2023, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
        datetime(2023, 12, 31, 0, 0, 0, tzinfo=timezone.utc),
        verbose=True,
    )

    agg_config = [
        AggConfig(col("precipitation").sum(), timedelta(days=1)),
        AggConfig(col("temperature_2m").mean(), timedelta(days=1)),
    ]

    aggregated_df = aggregate(obs.values, agg_config)

    assert aggregated_df.shape[0] == (365 * 24)  # One year of hourly data
    assert aggregated_df[0:23, agg_config[0].get_var_label()].is_null().any()
    assert aggregated_df[24:, agg_config[0].get_var_label()].is_not_null().any()
