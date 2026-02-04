from abc import ABC, abstractmethod
import warnings
from polars import DataFrame, col, selectors as cs
from requests import Session
from requests_cache import CachedSession, logger, orjson_serializer

from mlweather.collection.variables import ADMISSIBLE_VARIABLES

# GENERIC CLASS FOR ALL WEATHER RECORDS ########################################
# Used as base class for Observations and Forecasts to
# 1/ give general behaviour and 2/ allow specialization in child classes


@abstractmethod
class Records(ABC):
    """
    Data class representing weather observations with metadata.
    Records are hourly data.

    Attributes:
        lat_lon (tuple): Latitude and longitude coordinates as (float, float).
        elevation (float): Elevation of the location in meters
        units (dict): Dictionary mapping measurement names to their units.
        record_table (DataFrame): Polars DataFrame containing the weather records.
    """

    lat_lon: tuple[float, float]
    elevation: float
    units: dict[str, str]
    record_table: DataFrame

    def __init__(
        self,
        lat_lon: tuple[float, float],
        elevation: float,
        units: dict[str, str],
        record_table: DataFrame,
    ) -> None:
        super().__init__()
        self.lat_lon = lat_lon
        self.elevation = elevation
        self.units = units
        self.record_table = record_table

    @abstractmethod
    def __repr__(self):
        pass

    @staticmethod
    def are_var_names_valid(variable_names: list[str]) -> bool:
        admissible_var_names = [v.name for v in ADMISSIBLE_VARIABLES]
        invalid_var_names = list(
            filter(lambda v: v not in admissible_var_names, variable_names)
        )
        if invalid_var_names:
            raise ValueError(
                f"The following variable names are not admissible: {invalid_var_names}"
            )
        return True

    @staticmethod
    def get_openmeteo(
        base_url: str,
        params: dict,
        *,
        cache_enabled: bool = True,
        cache_expire_after: int = 86400 * 31,
        cache_sqlite_filename: str = "api_cache.sqlite",
        retry_attempts: int = 3,
        verbose: bool = False,
    ) -> dict:
        # For caching strategy, prepare the session: either with cached or not (base requests)
        if cache_enabled:
            # Create a persistent session
            session = CachedSession(
                cache_name=cache_sqlite_filename,  # SQLite file
                backend="sqlite",
                expire_after=cache_expire_after,  # TTL
                stale_if_error=True,
                serializer=orjson_serializer,
            )
            # Query
            response = session.get(base_url, params=params)
        else:
            # Using base requests session
            session = Session()

        # if verbose, tell if from cache or not (only for cached session)
        if verbose and cache_enabled:
            if response.from_cache:
                logger.info("Response retrieved from cache")
            else:
                logger.info("Response retrieved from server")
        # Retry logic
        attempts_number = 1
        while attempts_number <= retry_attempts:
            try:
                # Run the request
                response = session.get(base_url, params=params)
                # If verbose info URL
                if verbose:
                    logger.info(f"Request URL: {response.url}")
                # Parse response here to cover JSONDecodeError in the try block
                content = response.json()
                # Exit because the job is nicely done
                break
            # If failed, retry if not too many attempts
            except Exception as e:
                # Log the retry attempt
                warnings.warn(
                    f"Request failed at attempt {attempts_number}, retrying...",
                    RuntimeWarning,
                )
                # Iterate on the attempt number
                attempts_number += 1
                # If too many attempts, raise the error and stop iterations
                if attempts_number > retry_attempts:
                    raise e
                # Go to next iteration to retry
                continue
        # Explicit error if error in response json
        if "error" in content:
            raise ValueError(f"Request error: {content['reason']}")
        response.raise_for_status()

        return content

    @staticmethod
    def is_regular_time(record_table: DataFrame) -> None:
        # All valid_datetimes are unique within each init_datetime
        if not (
            # For each init_datetime, make sure we have as many rows as unique values of valid_datetime
            record_table.group_by(["init_datetime"])
            .agg(col("valid_datetime").n_unique() == col("valid_datetime").len())[
                "valid_datetime"
            ]
            .all()
        ):
            raise ValueError("Observations do not have regular time intervals")

        # The diff of ordered valid_datetime within each init_datetime is constant and unique
        record_table = (
            record_table.sort("init_datetime", "valid_datetime")
            .with_columns(
                (col("valid_datetime") - col("valid_datetime").shift(1))
                .over("init_datetime")
                .alias("diff"),
            )
            .filter(col("diff").is_not_null())
        )

        diffs = record_table["diff"].value_counts()

        if diffs.shape[0] != 1:
            raise ValueError(
                "Observations do not have regular time intervals within init_datetimes"
                f"found time steps {diffs}"
            )

        return None

    @staticmethod
    def prepare_hourly_records(resp_dict: dict) -> DataFrame:
        # records
        hourly_values = (
            DataFrame(resp_dict["hourly"])
            .rename({"time": "valid_datetime"})
            .with_columns(
                col("valid_datetime").str.to_datetime(
                    format="%Y-%m-%dT%H:%M", time_zone="UTC"
                )
            )
            # Reorder columns to have valid_datetime first
            .select(
                "valid_datetime",
                cs.exclude("valid_datetime"),
            )
        )

        return hourly_values

    @staticmethod
    # Reorder with valid and init datetimes first
    def reorder_columns(hourly_values: DataFrame) -> DataFrame:
        return hourly_values.select(
            "init_datetime",
            "valid_datetime",
            cs.exclude("init_datetime", "valid_datetime"),
        )

    @staticmethod
    def select_api_url(
        free_access_url: str, commercial_access_api: str, api_key: str | None
    ) -> str:
        # Build query
        if api_key is None:
            base_url = free_access_url
        elif isinstance(api_key, str):
            base_url = commercial_access_api
        else:
            raise ValueError("API key must be a string or None")

        return base_url
