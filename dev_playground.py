from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv
from polars import col, concat
import logging
import os

from mlweather.collection.forecasts import Forecasts
from mlweather.collection.observations import Observations
from mlweather.aggregation import Aggregation
from mlweather.aggregation import FeatureGenerator
from mlweather.collection.records import Records
from mlweather.collection.variables import ADMISSIBLE_VARIABLES


load_dotenv()
# Set the root logger level to Info
logging.basicConfig(level=logging.INFO)

loc = (49.875255, -4.121925)  # Paris coordinates
start_period = datetime(2025, 10, 1, tzinfo=timezone.utc)
end_period = datetime(2026, 1, 9, tzinfo=timezone.utc)


start_count = datetime.now()
# Observations collection
observations = Observations.collect(
    loc,
    start=start_period,
    end=end_period,
    variables_names=["temperature_2m", "precipitation"],
    query_by_period_slices=True,
    verbose=True,
    api_key=os.getenv("OPENMETEO_API_KEY"),
)
print("collected in ", datetime.now() - start_count)


start_count = datetime.now()
# Forecasts collection
forecasts = Forecasts.collect(
    loc,
    forecast_horizon_days_max=7,
    start=start_period,
    end=end_period,
    variables_names=["temperature_2m", "precipitation"],
    verbose=True,
    api_key=os.getenv("OPENMETEO_API_KEY"),
)
print("collected in ", datetime.now() - start_count)

fore = Forecasts.collect(
    lat_lon=(52.52, 13.41),
    variables_names=["temperature_2m", "precipitation"],
    forecast_horizon_days_max=2,
    start=datetime(2024, 2, 1, 0, 0, 0, tzinfo=timezone.utc),
    end=datetime(2024, 3, 1, 1, 0, 0, tzinfo=timezone.utc),
    cache_enabled=False,  # Disable cache for testing
    query_by_period_slices=True,
    period_slice_days=7,
)
# Source des forecasts passée : https://open-meteo.com/en/docs/historical-forecast-api

# Feature generation
feat_gen = FeatureGenerator(
    [
        Aggregation(
            col("temperature_2m").sum() / 24,
            observation_period=timedelta(days=30),
            forecast_period=None,
        ),
        # Degree days: (daily min temperature + daily max temperature) / 2 summed over the observation period (30 days)
        Aggregation(
            (
                (
                    col("temperature_2m")
                    .min()
                    .over(col("valid_datetime").dt.truncate("1d"))
                    + col("temperature_2m")
                    .max()
                    .over(col("valid_datetime").dt.truncate("1d"))
                )
                / 2
            ).sum()
            / 24,
            observation_period=timedelta(days=30),
            forecast_period=None,
        ),
    ],  # Daily mean temperature]
)

# Daily range from start_period to end_period

features_datetimes = [
    start_period + timedelta(days=i)
    for i in range((end_period - start_period).days + 1)
]

feat_gen.generate_features(
    features_datetimes=features_datetimes,
    observations=observations,
)
