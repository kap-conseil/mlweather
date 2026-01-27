from datetime import datetime, timezone, timedelta
from polars import col, concat


from mlweather.collection.forecasts import Forecasts
from mlweather.collection.observations import Observations
from mlweather.aggregation import Aggregation
from mlweather.aggregation import FeatureGenerator
from mlweather.collection.records import Records
from mlweather.collection.variables import ADMISSIBLE_VARIABLES


loc = (49.875255, -4.121925)  # Paris coordinates
start_period = datetime(2024, 10, 1, tzinfo=timezone.utc)
end_period = datetime(2026, 1, 9, tzinfo=timezone.utc)


start_count = datetime.now()
# Observations collection
observations = Observations.collect(
    loc,
    start=start_period,
    end=end_period,
    variables_names=["temperature_2m", "precipitation"],
    query_by_period_slices=True,
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


from requests_cache import CachedSession

# Create a persistent session
session = CachedSession(
    cache_name="api_cache",  # SQLite file
    backend="sqlite",
    expire_after=86400,  # 1 day TTL
)

url = "https://httpbin.org/get"

session.prepare_request
# Use the session instead of requests
response = session.get(url)
print(response.from_cache)

# First request (should fetch from the web)
response = session.get(url)
print("First request, from cache?", response.from_cache)
print(response.json())

# Second request (should hit the cache)
response = session.get(url)
print("Second request, from cache?", response.from_cache)
print(response.json())

reponse
