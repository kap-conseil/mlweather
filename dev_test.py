from polars import (
    col,
    datetime_range,
)

from functools import reduce
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv
import os


from mlweather.collection.observations import Observations
from mlweather.collection.forecasts import Forecasts
from mlweather.aggregation import (
    Aggregation,
    FeatureGenerator,
)

load_dotenv()

# FETCH OBSERVATIONS ##############################################################
start_dt = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
end_dt = start_dt + timedelta(days=366 / 2)
selected_vars = [
    "precipitation",
    "temperature_2m",
]

# Collect data
obs = Observations.collect(
    (48.0, -2.0),
    selected_vars,
    start_dt,
    end_dt,
    api_key=os.getenv("OPENMETEO_API_KEY"),
    verbose=False,
)

fore = Forecasts.collect(
    (48.0, -2.0),
    selected_vars,
    start_dt,
    end_dt,
    (0, 7),
    api_key=os.getenv("OPENMETEO_API_KEY"),
)

# Prepare transformations
aggs = [
    # Aggregation(col("precipitation").sum(), observation_period=timedelta(days=7)),
    # Aggregation(col("precipitation").sum(), forecast_period=timedelta(days=7)),
    Aggregation(
        col("temperature_2m").sum(),
        observation_period=timedelta(hours=2),
    ),
    # Aggregation(
    #     col("precipitation").sum(),
    #     observation_period=timedelta(hours=2),
    #     forecast_period=timedelta(hours=2),
    # ),
]

# Generate a range from  2024, 2, 20, 0, 0 to 2024, 7, 13, 0, 0 (daily)
features_datetimes = datetime_range(
    start_dt,
    end_dt,
    "1d",
    eager=True,
).to_list()


featgen = FeatureGenerator(aggs)

temp = featgen.generate_features(
    features_datetimes,
    observations=obs,
    forecasts=fore,
    control_aggregations=False,
)
