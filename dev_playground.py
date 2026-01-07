from polars import (
    arange,
    col,
    datetime_range,
    when,
    lit,
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
    (1, 7),
    api_key=os.getenv("OPENMETEO_API_KEY"),
)

# Prepare transformations
aggs = [
    Aggregation(col("precipitation").sum(), observation_period=timedelta(days=7)),
    Aggregation(col("precipitation").sum(), forecast_period=timedelta(days=7)),
    Aggregation(
        col("temperature_2m").sum(),
        observation_period=timedelta(hours=2),
    ),
    Aggregation(
        col("precipitation").sum(),
        observation_period=timedelta(hours=2),
        forecast_period=timedelta(hours=2),
    ),
]

# Check that they have the same valid_datetimes
if not (obs.record_table["valid_datetime"].unique().sort()).equals(
    (fore.record_table["valid_datetime"].unique().sort())
):
    raise ValueError("Observations and Forecasts must have the same valid_datetimes")

# Replace if not null value by 1.00 for precipitation to test the aggregation
obs.record_table = obs.record_table.with_columns(
    when(col("precipitation").is_not_null())
    .then(lit(1.0))
    .otherwise(col("precipitation"))
    .alias("precipitation")
).filter(arange(obs.record_table.shape[0]) < 10)

fore.record_table = fore.record_table.with_columns(
    when(col("precipitation").is_not_null())
    .then(lit(1.0))
    .otherwise(col("precipitation"))
    .alias("precipitation")
)

# Generate a range from  2024, 2, 20, 0, 0 to 2024, 7, 13, 0, 0 (daily)
features_datetimes = obs.record_table.filter(col("valid_datetime").dt.minute() == 0)[
    "valid_datetime"
].to_list()

# Case obs only
# init the feature generator
featgen_obs_only = FeatureGenerator(
    [
        Aggregation(col("precipitation").sum(), observation_period=timedelta(hours=2)),
    ]
)

features_obs_only = featgen_obs_only.generate_features(
    features_datetimes,
    observations=obs,
    control_aggregations=False,
)

# Check features values
# missing values should be there because of the 2-hour aggregation: first hour has not aggregation
features_obs_only["precipitation_sum_obs_7200s"].is_null().sum() == (
    Aggregation.get_steps_in_agg_periods(timedelta(hours=2)) - 1
)
# Check values : since precipitation values were replaced by 1.0, the sum over 2 hours should be 2.0
assert features_obs_only.filter(col("precipitation_sum_obs_7200s").is_not_null())[
    "precipitation_sum_obs_7200s"
].unique().to_list() == [2.0]
