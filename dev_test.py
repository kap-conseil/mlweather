from polars import (
    DataFrame,
    Expr,
    LazyFrame,
    col,
    Datetime,
    date_range,
    datetime_range,
    datetime_ranges,
    len as length,
    lit,
    selectors as cs,
    concat,
    collect_all,
)
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv
import os
import timeit

from mlweather.collection.observations import Observations
from mlweather.collection.forecasts import Forecasts
from mlweather.aggregation import (
    Aggregation,
    FeatureGenerator,
)

load_dotenv()

# FETCH OBSERVATIONS ##############################################################
start_dt = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
end_dt = start_dt + timedelta(days=366)
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
    Aggregation(col("precipitation").sum(), observation_period=timedelta(days=7)),
    Aggregation(col("precipitation").sum(), forecast_period=timedelta(days=7)),
    Aggregation(
        col("temperature_2m").sum(),
        observation_period=timedelta(days=7),
    ),
    Aggregation(
        col("precipitation").sum(),
        observation_period=timedelta(days=7),
        forecast_period=timedelta(days=7),
    ),
]

# Generate a range from  2024, 2, 20, 0, 0 to 2024, 7, 13, 0, 0 (daily)
features_datetimes = datetime_range(
    datetime(2024, 2, 20, 0, 0, 0, tzinfo=timezone.utc),
    datetime(2024, 7, 13, 0, 0, 0, tzinfo=timezone.utc),
    "1d",
    eager=True,
).to_list()

def check_features_datetimes(
    features_datetimes: list[Datetime], all_records: LazyFrame
) -> bool:
    """Check that all focal valid_datetime are present in the LazyFrame."""
    # Valid datetimes present in the LazyFrame of weather records
    weather_valid_datetimes = set(
        (
            all_records.select("valid_datetime")
            .unique()
            .sort("valid_datetime")
            .collect()["valid_datetime"]
            .to_list()
        )
    )
    # Find datetime expected in the features that are not in the weather records
    missing_datetimes = set(features_datetimes).difference(weather_valid_datetimes)
   XXX Add error
    return add return

XXX => Transform the source (obs and fore) dateties to UTC
XXX => Then check the 

# 1/ Generate all records mixing the observations and forecasts (transformed to hourly init_datetime)
all_records = FeatureGenerator._prepare_all_records(obs.record_table, fore.record_table)
# 2/ Work by periods set, filter and apply the aggregation expressions over all focal valid_datetime => Collect all results
start_timer = datetime.now(timezone.utc)
temp = []
for same_period_aggs in FeatureGenerator(aggs).get_same_period_aggregations():
    plans_for_features = []
    obs_period = same_period_aggs[0].observation_period
    fore_period = same_period_aggs[0].forecast_period
    expressions = [agg.expression for agg in same_period_aggs]
    for focal_valid_datetime in features_datetimes:
        plans_for_features.append(
            FeatureGenerator._filter_apply(
                all_records,
                obs_period,
                fore_period,
                expressions,
                focal_valid_datetime,
            )
        )
    temp.append(concat(collect_all(plans_for_features), how="diagonal"))


print("Timer:", datetime.now(timezone.utc) - start_timer)





all_records.select("valid_datetime").unique().sort("valid_datetime").collect()
# 3/ Aggregate over the results to have a single row per focal valid_datetime

time = timeit.timeit(
    stmt="all_records.select(col('valid_datetime')).unique().sort('valid_datetime').collect()",
    globals=globals(),
    number=10,
)

print(f"Average: {time / 10_000:.9f} s")
