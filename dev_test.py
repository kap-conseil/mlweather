import polars as pl
from polars import (
    DataFrame,
    col,
    Datetime,
    len as length,
    Float64,
    Expr,
    Duration,
    Series,
    when,
)
import numpy as np
from datetime import datetime, timezone, timedelta

from zoneinfo import ZoneInfo
from dotenv import load_dotenv
import os
import re


from mlweather.collection import (
    Measure,
    ADMISSIBLE_MEASURES,
    Observations,
    prepare_hourly_raw_observations,
)
from mlweather.aggregation import AggConfig, aggregate

load_dotenv()

# FETCH PAST DATA ##############################################################
obs = Observations.get_obs(
    (0.0, 0.0),
    [
        "precipitation",
        "temperature_2m",
    ],
    datetime(2010, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
    datetime(2024, 1, 2, 0, 0, 0, tzinfo=timezone.utc),
    api_key=os.getenv("OPENMETEO_API_KEY"),
    # api_key="z",
    verbose=True,
)

# Define the aggregations
agg_config = [
    AggConfig(col("precipitation").sum(), timedelta(days=1)),
    AggConfig(col("precipitation").sum(), timedelta(days=7)),
    AggConfig(col("temperature_2m").mean(), timedelta(days=1)),
    AggConfig(col("temperature_2m").mean(), timedelta(days=7)),
]


start_time = datetime.now(timezone.utc)
temp = aggregate(obs.values, agg_config)
print(f"Aggregation completed in {datetime.now(timezone.utc) - start_time} seconds.")


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
