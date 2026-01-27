from datetime import datetime, timezone, timedelta
from polars import col


from mlweather.collection.forecasts import Forecasts
from mlweather.collection.observations import Observations
from mlweather.aggregation import Aggregation
from mlweather.aggregation import FeatureGenerator
from mlweather.collection.variables import ADMISSIBLE_VARIABLES


loc = (47.875255, -4.121925)  # Paris coordinates
start_period = datetime(2025, 10, 1, tzinfo=timezone.utc)
end_period = datetime(2026, 1, 9, tzinfo=timezone.utc)

# Observations collection
observations = Observations.collect(
    loc,
    start=start_period,
    end=end_period,
    variables_names=["temperature_2m", "precipitation"],
)


# Forecasts collection
forecasts = Forecasts.collect(
    loc,
    forecast_horizon_days_max=7,
    start=start_period,
    end=end_period,
    variables_names=["temperature_2m", "precipitation"],
)
# Source des forecasts passée : https://open-meteo.com/en/docs/historical-forecast-api

# Feature generation
feat_gen = FeatureGenerator(
    [
        Aggregation(
            col("temperature_2m").sum() / 24,
            observation_period=timedelta(days=30),
            forecast_period=timedelta(days=7),
        ),
        # Aggregation(
        #     col("temperature_2m").sum(),
        #     observation_period=timedelta(days=30),
        #     forecast_period=timedelta(days=7),
        # ),
    ],  # Daily mean temperature]
)

features_datetimes = [
    datetime(2025, 12, 1, tzinfo=timezone.utc) + timedelta(days=7),
]
