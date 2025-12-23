from functools import reduce
from polars import (
    DataFrame,
    LazyFrame,
    col,
    Datetime,
    date_ranges,
    datetime_ranges,
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
    map_batches,
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
start_dt = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
end_dt = start_dt + timedelta(days=366)

# Collect data
obs = Observations.collect(
    (48.0, -2.0),
    [
        "precipitation",
        "temperature_2m",
    ],
    start_dt,
    end_dt,
    api_key=os.getenv("OPENMETEO_API_KEY"),
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
    api_key=os.getenv("OPENMETEO_API_KEY"),
)

def _prepare_records(obs: Observations | None, fore: Forecasts | None, *, lazy_output: bool) -> DataFrame | LazyFrame:
    # Replicate the forecast data to hourly frequency by self merging on the hourly spread dates
    # Add hours to dates based on init_datetime
    # Insert date ranges (hours within the day) withi the cells of a new column then explode
    non_meteo_vars = [
        "init_datetime",
        "valid_datetime",
        "past_day",
        "init_datetime_raw",
        "init_datetime_raw_end",
    ]

    if obs is None:
        obs_side = DataFrame().lazy()
    else:
        # Add id to observations
        obs_side = (
            obs.record_table.lazy().sort("valid_datetime")
            .with_row_index("id")
            .with_columns("observation-" + col("id").cast(str).alias("id"))
        )

    if fore is None:
        forecast_side = DataFrame().lazy()
    else:
        forecast_side = (
            # Add id
            fore.record_table.lazy().with_row_index("id")
            .with_columns( "forecast-" + col("id").cast(str).alias("id"))
            # Replicate the  daily computeed forecast to intraday steps (hourly)
            .rename({"init_datetime": "init_datetime_raw"})
            # Generate datetime ranges for each init_datetime (24 hour with the daily init_datetime)
            .with_columns(
                (col("init_datetime_raw") + timedelta(hours=23)).alias("init_datetime_raw_end")
            )
            .with_columns(
                datetime_ranges(
                    "init_datetime_raw", "init_datetime_raw_end", interval=timedelta(hours=1)
                ).alias("init_datetime")
            )
            # From compact (nested) form to long form
            .explode("init_datetime")
            .select(*non_meteo_vars, cs.exclude(non_meteo_vars))
            .sort(["valid_datetime", "init_datetime"])
        )


    # Assemble
    all_records = concat([obs_side, forecast_side], how="diagonal")
    # Treat non lazy
    if not lazy_output:
        all_records = all_records.collect()

    return all_records

def _filter_apply(
    all_records: LazyFrame | DataFrame,
    focal_valid_datetime: datetime,
    observation_period: timedelta | None,
    forecast_period: timedelta | None,
) -> LazyFrame | DataFrame:
    # Deal with 0 length periods
    # Check that at least one period is not none
    if observation_period is None and forecast_period is None:
        raise ValueError(
            "At least one of observation_period or forecast_period must be not None"
        )
    if observation_period is None:
        observation_period = timedelta(0)
    if forecast_period is None:
        forecast_period = timedelta(0)
    return (
        all_records.filter(
        # In obs:
        # from the valid_time - the forecast period (included),
        # back to included to the observation period and forecast period (bound excluded)
        (
            col("init_datetime").is_null()      # only for observations
            & (col("valid_datetime") <= (focal_valid_datetime - forecast_period))
            & (
                col("valid_datetime")
                > (focal_valid_datetime - forecast_period - observation_period)
            )
        ) |
        # In forecast:
        # for the init_times that match perfectly the first valid time
        # from the valid_time (included),
        # back to included to the observation period period (bound excluded)
        (   
            col("init_datetime").is_not_null()      # only for forecasts
            & (col("init_datetime") ==(focal_valid_datetime - forecast_period))
            & (col("valid_datetime") <= (focal_valid_datetime))
            & (
                col("valid_datetime")
                > (focal_valid_datetime - forecast_period)
            )
        )
    ).with_columns(
        lit(focal_valid_datetime).alias("focal_valid_datetime"),
        # Number of rows in the output
        length().alias("n_rows_in_filtered")
    )
    .sort(["valid_datetime", "init_datetime"])
    )


all_records = _prepare_records(obs, fore, lazy_output=True)

start_timer = datetime.now(timezone.utc)

def get_focal_valid_datetimes(all_records: LazyFrame | DataFrame) -> list[datetime]:
    # Restrict the usefull serie (compatible with lazy mode)
    focal_valid_datetimes = (
        all_records
        .select("valid_datetime")
        .filter(col("valid_datetime").is_not_null())
        .sort("valid_datetime")
        .unique()
    )
    if isinstance(focal_valid_datetimes, LazyFrame):
        focal_valid_datetimes = focal_valid_datetimes.collect()

    return focal_valid_datetimes["valid_datetime"].to_list()

get_focal_valid_datetimes(all_records)

concat(
    map(
        lambda dt: _filter_apply(
                all_records,
                dt,
                timedelta(hours=2),
                timedelta(hours=2),
        ),
        get_focal_valid_datetimes(all_records)
    )
)

for dt in get_focal_valid_datetimes(all_records):
    _filter_apply(
        all_records,
        dt,
        timedelta(hours=2),
        timedelta(hours=2),
    )
print("Timer:", datetime.now(timezone.utc) - start_timer)

start_timer = datetime.now(timezone.utc)
concatenated = concat(collected_results, how="diagonal")
if isinstance(all_records, LazyFrame):
    concatenated = concatenated.collect()
print("Timer:", datetime.now(timezone.utc) - start_timer)

all_records.collect()
all_records = all_records.drop("init_datetime_raw", "init_datetime_raw_end", "past_day")


 len(
    all_records_lazy.select("valid_datetime")
    .filter(col("valid_datetime").is_not_null())
    .unique()
    .collect()["valid_datetime"]
    .to_list()
)


for valid_datetime in all_records["valid_datetime"].unique():
    all_records.filter(
        (col("init_datetime").is_null()) & (col("valid_datetime") < valid_datetime)
    )


(
    filter_past_weather(
        pw_agg,
        agg_conf.measure_name,
        # At the end of the first duration period
        dt - duration[2] - Hour(1),
        duration[1],
    ),
)
met_vars = ["precipitation", "temperature_2m"]
test = (
    all_records.sort("valid_datetime", "init_datetime")
    .rolling(
        index_column="valid_datetime",
        period=timedelta(hours=8),
    )
    .agg(
        col("id").alias("ids_in_window"),
        col("valid_datetime").min().alias("min_valid_datetime"),
        col("valid_datetime").max().alias("max_valid_datetime"),
        pl_len().alias("length"),
        # # Return the values within the rolling window as a list
        *[col(met_var).alias(f"{met_var}_values_in_window") for met_var in met_vars],
        col("valid_datetime").alias("valid_datetimes_in_window"),
        col("init_datetime").alias("init_datetimes_in_window"),
    )
    .with_row_index("agg_group_id")
    .explode(
        [
            "ids_in_window",
            "valid_datetimes_in_window",
            "init_datetimes_in_window",
            *[f"{met_var}_values_in_window" for met_var in met_vars],
        ]
    )
)

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
