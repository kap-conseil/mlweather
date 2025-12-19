from functools import reduce
from polars import (
    DataFrame,
    col,
    Datetime,
    int_range,
    len as length,
    Float64,
    UInt32,
    Expr,
    Duration,
    Series,
    lit,
    when,
    len as pl_len,
    selectors as cs,
    concat,
)
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo
from dotenv import load_dotenv
import os
import re

from mlweather.collection.observations import Observations
from mlweather.collection.forecasts import Forecasts
from mlweather.aggregation import (
    Aggregation,
    FeatureGenerator,
)

load_dotenv()

# FETCH OBSERVATIONS ##############################################################
start_time = datetime.now(timezone.utc)

start_dt = datetime(2024, 5, 1, 0, 0, 0, tzinfo=timezone.utc)
end_dt = datetime(2024, 6, 1, 0, 0, 0, tzinfo=timezone.utc)

# Collect data
obs = Observations.collect(
    (48.0, -2.0),
    [
        "precipitation",
        "temperature_2m",
    ],
    start_dt,
    end_dt,
    # api_key=os.getenv("OPENMETEO_API_KEY"),
    api_key=os.getenv("WEATHERAPI_API_KEY"),
    verbose=False,
)

fore = Forecasts.collect(
    (48.0, -2.0),
    [
        "precipitation",
        "temperature_2m",
    ],
    start_dt,
    end_dt,
    (0, 7),
    api_key=os.getenv("WEATHERAPI_API_KEY"),
)

# Insert forecast path into all observations entries to replicate forecast path by hour (not only dayly init_times).
# Do so by merging the forecast path (based on midnight computation, not hourly) with the observations based on the next day midnight into each observation valid_datetime.
left_side = (
    obs.record_table.sort("valid_datetime")
    .with_row_index()
    # Create a synthétic init_time on the obs side to inject the forecast paths
    .with_columns(
        (col("valid_datetime").dt.truncate("1d") + timedelta(days=1)).alias(
            "init_datetime@fore"
        )
    )
    .rename(lambda x: x + "@obs" if x != "init_datetime@fore" else x)
).with_columns(lit(1).alias("test@obs"))

# Add suffix to right side columns
right_side = (
    fore.record_table.sort(["init_datetime", "valid_datetime"])
    .with_row_index()
    .rename(lambda x: x + "@fore")
).with_columns(lit(1).alias("test@fore"))

# Insert the forecast on left side by
expanded_right_side = left_side.join(
    right_side,
    left_on="init_datetime@fore",
    right_on="init_datetime@fore",
    how="left",
    suffix="@fore",
)

obs_temp = (
    expanded_right_side.select(cs.ends_with("@obs"))
    .rename(lambda x: x.replace("@obs", ""))
    .with_columns(source=lit("obs"))
)
fore_temp = (
    expanded_right_side.select(cs.ends_with("@fore"))
    .rename(lambda x: x.replace("@fore", ""))
    .filter(col("valid_datetime").is_not_null())
    .with_columns(source=lit("fore"))
    .sort(["valid_datetime", "past_day", "init_datetime"])
    .with_columns(
        (col("init_datetime") + (int_range(pl_len()) * timedelta(hours=1))).alias(
            "init_datetime_intraday"
        )
    )
)


tt = concat(
    [
        obs_temp,
        fore_temp,
    ],
    how="diagonal",
)

ttt = tt.filter(
    (col("valid_datetime") >= datetime(2024, 5, 20))
    & (col("valid_datetime") < datetime(2024, 5, 24))
)

# Create a test dataframe with datetime every hour for 2 days
test_df = (
    DataFrame(
        {
            "valid_datetime": [
                datetime(2024, 5, 20, 0, 0, 0) + timedelta(hours=hour)
                for hour in range(48)
            ],
        }
    )
    .with_columns(val=int_range(48))
    .with_row_index()
)

(
    test_df.rolling(
        index_column="valid_datetime",
        period=timedelta(hours=2),
    )
    .agg(
        col("valid_datetime").min().alias("min_valid_datetime"),
        col("valid_datetime").max().alias("max_valid_datetime"),
        col("val").sum(),
        # Return the values within the rolling window as a list
        col("val").alias("values_in_window"),
        col("valid_datetime").alias("dates_in_window"),
    )
    .with_row_index()
).explode(["values_in_window", "dates_in_window"])
