import re
from polars import (
    DataFrame,
    col,
    Datetime,
    int_range,
    lit,
    selectors as cs,
    len as pl_len,
)
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from requests import Request, Session
from functools import reduce


# METEOROLOGICAL VARIABLES #####################################################


class Variable:
    """
    Data class representing a weather variables with associated metadata.

    Attributes:
        name (str): The name of the measurement.
        unit (str): The unit in which the measurement is expressed.
        value_type (type): The expected data type of the measurement value.
        valid_time (str): The time or period for which the measurement is valid.
    """

    name: str
    unit: str
    value_type: type
    measurement: str

    def __init__(self, name: str, unit: str, value_type: type, measurement: str):
        self.name = name
        self.unit = unit
        self.value_type = value_type
        self.measurement = measurement

    @classmethod
    def init_from_name(cls, variable_name: str) -> "Variable":
        """
        Retrieve the meteorological Variable given its name.

        Args:
            name (str): The name of the meteorological variable to retrieve.
        Returns:
            Variable: The corresponding Variable instance.
        """
        for var in ADMISSIBLE_VARIABLES:
            if var.name == variable_name:
                return var
        raise ValueError(
            f"Variable name '{variable_name}' not found in admissible meteorological Variables."
        )


# Definition of all the admissible measures
ADMISSIBLE_VARIABLES = set(
    [
        Variable("precipitation", "mm", float, "Preceding hour sum"),
        Variable("temperature_2m", "°C", float, "Instant"),
        Variable("wind_speed_10m", "km/h", float, "Instant"),
        Variable("wind_direction_10m", "°", float, "Instant"),
        Variable("relative_humidity_2m", "%", float, "Instant"),
        Variable("shortwave_radiation", "W/m²", float, "Preceding hour mean"),
        Variable("global_tilted_irradiance", "W/m²", float, "Preceding hour mean"),
        Variable("et0_fao_evapotranspiration", "mm", float, "Preceding hour sum"),
        Variable("dew_point_2m", "°C", float, "Instant"),
        Variable("apparent_temperature", "°C", float, "Instant"),
        Variable("pressure_msl", "hPa", float, "Instant"),
        Variable("surface_pressure", "hPa", float, "Instant"),
        Variable("rain", "mm", float, "Preceding hour sum"),
        Variable("snowfall", "cm", float, "Preceding hour sum"),
        Variable("cloud_cover", "%", float, "Instant"),
        Variable("cloud_cover_low", "%", float, "Instant"),
        Variable("cloud_cover_mid", "%", float, "Instant"),
        Variable("cloud_cover_high", "%", float, "Instant"),
        Variable("direct_radiation", "W/m²", float, "Preceding hour mean"),
        Variable("direct_normal_irradiance", "W/m²", float, "Preceding hour mean"),
        Variable("diffuse_radiation", "W/m²", float, "Preceding hour mean"),
        Variable("sunshine_duration", "s", float, "Preceding hour sum"),
        Variable("wind_speed_100m", "km/h", float, "Instant"),
        Variable("wind_direction_100m", "°", float, "Instant"),
        Variable("wind_gusts_10m", "km/h", float, "Instant"),
        Variable("weather_code", "WMO code", int, "Instant"),
        Variable("snow_depth", "m", float, "Instant"),
        Variable("vapour_pressure_deficit", "kPa", float, "Instant"),
        Variable("soil_temperature_0_to_7cm", "°C", float, "Instant"),
        Variable("soil_temperature_7_to_28cm", "°C", float, "Instant"),
        Variable("soil_temperature_28_to_100cm", "°C", float, "Instant"),
        Variable("soil_temperature_100_to_255cm", "°C", float, "Instant"),
        Variable("soil_moisture_0_to_7cm", "m³/m³", float, "Instant"),
        Variable("soil_moisture_7_to_28cm", "m³/m³", float, "Instant"),
        Variable("soil_moisture_28_to_100cm", "m³/m³", float, "Instant"),
        Variable("soil_moisture_100_to_255cm", "m³/m³", float, "Instant"),
    ]
)


def to_utc_safe(dt: datetime, assume_tz=ZoneInfo("UTC")) -> datetime:
    if dt.tzinfo is None or dt.tzinfo.utcoffset(dt) is None:
        dt = dt.replace(tzinfo=assume_tz)
    return dt.astimezone(timezone.utc)


# DEFINITION OF THE STRUCTURES #################################################
# Parent and ABC method for weather data collections
# Inherit methods to child and force child level implementation using ABC
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
        Records at {self.lat_lon} with {self.record_table} hourly entries.
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
        content = response.json()
        print(content)
        print("error" in content)
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


class Observations(Records):
    """
    Class representing weather observations with metadata.
    Records are hourly data.

    Attributes:
        lat_lon (tuple): Latitude and longitude coordinates as (float, float).
        elevation (float): Elevation in meters.
        units (dict): Dictionary mapping measurement names to their units.
        records (DataFrame): Polars DataFrame containing the weather records.
    """

    def __init__(
        self,
        lat_lon: tuple[float, float],
        elevation: float,
        units: dict[str, str],
        record_table: DataFrame,
    ) -> None:
        # First construct as parent
        super().__init__(lat_lon, elevation, units, record_table)
        # Child specific attributes can be added here if needed

    @classmethod
    def collect(
        cls,
        lat_lon: tuple[float, float],
        variables_names: list[str],
        start_date: datetime,
        end_date: datetime,
        api_key: str | None = None,
        verbose: bool = False,
    ):
        """
        Retrieve weather observations from Open-Meteo API (hourly data) for a specified location and a time period.

        This class method fetches historical weather data from the Open-Meteo archive API,
        supporting both free and commercial API access. It validates input parameters,
        constructs the API request, handles the response, and returns structured observations.

        Args:
            lat_lon (tuple[float, float]): Latitude and longitude coordinates as a tuple.
            variables_names (list[str]): List of weather measurement variable names to retrieve.
                Must be from the ADMISSIBLE_VARIABLES list.
            start_date (datetime): Start date for the data retrieval period. By default supposes UTC if no timezone provided.
            end_date (datetime): End date for the data retrieval period. By default supposes UTC if no timezone provided.
            api_key (str | None, optional): API key for commercial access. If None,
                uses the free archive API. Defaults to None.
            verbose (bool, optional): If True, prints the query URL for debugging.
                Defaults to False.

        Returns:
            cls: An instance of the class containing the retrieved weather observations
            with location coordinates, elevation, measurement units, hourly values,
            and regularity information.
        """
        # Check arguments
        cls.are_var_names_valid(variables_names)

        params = {
            "latitude": lat_lon[0],
            "longitude": lat_lon[1],
            "start_date": to_utc_safe(start_date).strftime("%Y-%m-%d"),
            "end_date": to_utc_safe(end_date).strftime("%Y-%m-%d"),
            "hourly": ",".join(variables_names),
            "timezone": "GMT",
        }

        # If an API key is provided, add it to the request parameters
        if api_key is not None:
            params["apikey"] = api_key

        # Parse and prepare output
        resp_dict = Observations.get_openmeteo(
            Records.select_api_url(
                "https://archive-api.open-meteo.com/v1/archive",
                "https://customer-archive-api.open-meteo.com/v1/archive",
                api_key,
            ),
            params,
            verbose,
        )

        # Prepare Observations init
        units = resp_dict["hourly_units"]
        units.pop("time", None)

        # Add init datetime as a copy of valid_datetime beause they are Observations
        hourly_values = Observations.prepare_hourly_records(resp_dict).with_columns(
            lit(None).cast(Datetime).alias("init_datetime")
        )

        # Check regular time grid within init_datetime
        Records.is_obs_regular_time(hourly_values)

        # Reorder with valid and intit datetimes first
        hourly_values = hourly_values.select(
            "init_datetime",
            "valid_datetime",
            cs.exclude("init_datetime", "valid_datetime"),
        )

        # Prepare the observations with the hourly table
        obs = cls(
            (resp_dict["latitude"], resp_dict["longitude"]),
            resp_dict["elevation"],
            units,
            hourly_values.sort("init_datetime", "valid_datetime"),
        )

        return obs


def validate_past_day_range(past_days_range: tuple[int, int]) -> tuple[int, int]:
    if (
        not isinstance(past_days_range, tuple)
        or len(past_days_range) != 2
        or not all(isinstance(day, int) for day in past_days_range)
        or past_days_range[0] < 0
        or past_days_range[1] < past_days_range[0]
    ):
        raise ValueError(
            "past_days_range must be a tuple of two non-negative integers (start_day, end_day) with end_day >= start_day."
        )

    return past_days_range


class Forecasts(Records):
    """
    Class representing weather forecasts with metadata.
    Records are hourly data.

    Attributes:
        lat_lon (tuple): Latitude and longitude coordinates as (float, float).
        elevation (float): Elevation in meters.
        units (dict): Dictionary mapping measurement names to their units.
        records (DataFrame): Polars DataFrame containing the weather records.
    """

    def __init__(
        self,
        lat_lon: tuple[float, float],
        elevation: float,
        units: dict[str, str],
        past_days_range: tuple[int, int],
        record_table: DataFrame,
    ) -> None:
        # First construct as parent
        super().__init__(lat_lon, elevation, units, record_table)
        # Child specific attributes can be added here if needed
        self.past_days_range = validate_past_day_range(past_days_range)

    @staticmethod
    def rename_day0_columns(
        record_table: DataFrame, queried_past_day_variables: list[str]
    ):
        # Get varaiables in day0
        days0_weather_variables = (
            set(record_table.columns)
            - {"valid_datetime"}
            - set(queried_past_day_variables)
        )
        # Create renaming mapping
        renaming_dict = {var: f"{var}_previous_day0" for var in days0_weather_variables}

        # Effectively rename
        return record_table.rename(renaming_dict)

    @staticmethod
    def melt_by_weather_variable(hourly_records: DataFrame):
        # Identify the weather variables (without the _dayX suffix)
        groups_for_melt = {
            c.split("_day")[0]
            for c in hourly_records.columns
            if re.search(r"_day\d+$", c)
        }
        # By variable, melt the corresponding columns
        melted_records = [
            hourly_records.select(["valid_datetime", cs.starts_with(g)])
            .unpivot(
                cs.starts_with(g),
                index="valid_datetime",
                variable_name="past_day",
                value_name=g,
            )
            # Extract the past day number from the column name
            .with_columns(col("past_day").str.extract(r"(\d+)$").cast(int))
            for g in groups_for_melt
        ]

        return melted_records

    @classmethod
    def collect(
        cls,
        lat_lon: tuple[float, float],
        variables_names: list[str],
        start_date: datetime,
        end_date: datetime,
        past_forecast_days_range: tuple[int, int],
        api_key: str | None = None,
        verbose: bool = False,
    ):
        # Check arguments
        cls.are_var_names_valid(variables_names)
        past_days_range = validate_past_day_range(past_forecast_days_range)

        # Define the vairable names to query for all vars and previous days
        all_measures_with_past_days = [
            f"{m}_previous_day{pd}"
            for m in variables_names
            for pd in range(past_days_range[0], past_days_range[1] + 1)
        ]

        # Build query params
        params = {
            "latitude": lat_lon[0],
            "longitude": lat_lon[1],
            "start_date": to_utc_safe(start_date).strftime("%Y-%m-%d"),
            "end_date": to_utc_safe(end_date).strftime("%Y-%m-%d"),
            "hourly": ",".join(all_measures_with_past_days),
            "timezone": "GMT",
        }
        # If an API key is provided, add it to the request parameters
        if api_key is not None:
            params["apikey"] = api_key

        # Parse and prepare output
        resp_dict = Forecasts.get_openmeteo(
            Records.select_api_url(
                "https://previous-runs-api.open-meteo.com/v1/forecast",
                "https://customer-previous-runs-api.open-meteo.com/v1/forecast",
                api_key,
            ),
            params,
            verbose,
        )

        # Prepare Observations init
        units = resp_dict["hourly_units"]
        units.pop("time", None)

        # Preapre the hourly records and transform to long per meteo var (along the previous days)
        hourly_values = Forecasts.prepare_hourly_records(resp_dict)

        hourly_values = Forecasts.rename_day0_columns(
            hourly_values, all_measures_with_past_days
        )

        hourly_values = Forecasts.melt_by_weather_variable(hourly_values)

        hourly_values = reduce(
            lambda left, right: left.join(
                right, on=["valid_datetime", "past_day"], how="inner"
            ),
            hourly_values,
        )
        # Prepare the observations with the hourly table
        fore = cls(
            (resp_dict["latitude"], resp_dict["longitude"]),
            resp_dict["elevation"],
            units,
            past_days_range,
            hourly_values,
        )

        return fore
