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


from mlweather.records import (
    Variable,
    ADMISSIBLE_VARIABLES,
    Observations,
    Forecasts,
)

load_dotenv()

# FETCH OBSERVATIONS ##############################################################
start_time = datetime.now(timezone.utc)

temp = Observations.collect(
    (48.0, -2.0),
    [
        "precipitation",
        "temperature_2m",
    ],
    datetime(2010, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
    datetime(2024, 1, 2, 0, 0, 0, tzinfo=timezone.utc),
    # api_key=os.getenv("OPENMETEO_API_KEY"),
    api_key=None,
    verbose=False,
)

temp = Forecasts.collect(
    (48.0, -2.0),
    [
        "precipitation",
        "temperature_2m",
    ],
    datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
    datetime(2025, 1, 2, 0, 0, 0, tzinfo=timezone.utc),
    (0, 2),
)

READY FOR PREPARE INIT_DATETIME (rounded to day) 