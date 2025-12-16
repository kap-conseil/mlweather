from abc import ABC, abstractmethod
from polars import DataFrame, col, int_range, len as pl_len, selectors as cs
from requests import Request, Session

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

    def __repr__(self) -> str:
        return f"""
        Records at {self.lat_lon} with {self.record_table.shape[0]} hourly entries.
        Elevation: {self.elevation} m
        Units: {self.units}
        Record Table:
        {self.record_table}
        """

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
    def get_openmeteo(base_url: str, params: dict, verbose: bool = False) -> dict:
        # Prepare the request
        req = Request("GET", base_url, params=params)
        prepared = req.prepare()
        # Print the full URL before sending the request
        if verbose:
            print("Query URL:", prepared.url)
        with Session() as session:
            response = session.send(prepared)
        # Raise error if status code not 200
        if response.status_code != 200:
            raise ValueError(
                f"Error fetching data: {response.status_code} - {response.text}"
            )
        content = response.json()
        # Explicit error if error in response json
        if "error" in content:
            raise ValueError(f"Request error: {content['reason']}")
        response.raise_for_status()

        return content

    @staticmethod
    def is_obs_regular_time(record_table: DataFrame) -> None:
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
        diffs = (
            record_table.sort("init_datetime", "valid_datetime")
            .with_columns(
                (col("valid_datetime") - col("valid_datetime").shift(1)).alias("diff"),
                by="init_datetime",
            )  # Drop first row with of group (using the row number)
            .with_columns((int_range(pl_len()).alias("index")), by="init_datetime")
            .remove(col("index") == 0)
        )["diff"].value_counts()

        if diffs.shape[0] != 1:
            raise ValueError(
                "Observations do not have regular time intervals within init_datetimes"
            )

        return None

    @staticmethod
    def prepare_hourly_records(resp_dict: dict) -> DataFrame:
        # records
        hourly_values = (
            DataFrame(resp_dict["hourly"])
            .rename({"time": "valid_datetime"})
            .with_columns(
                col("valid_datetime").str.to_datetime(format="%Y-%m-%dT%H:%M")
            )
            # Reorder columns to have valid_datetime first
            .select(
                "valid_datetime",
                cs.exclude("valid_datetime"),
            )
        )

        return hourly_values

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
