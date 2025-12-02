from polars import DataFrame, col, Datetime
from dataclasses import dataclass
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from requests import get, Request, Session
from requests_cache import CachedSession
from logging import debug, warning, error, info, critical


# MEASURE AND ADMISSIBLE WEATHER MEASURES ######################################


@dataclass
class Measure:
    """
    Data class representing a weather measurement with associated metadata.

    Attributes:
        name (str): The name of the measurement.
        unit (str): The unit in which the measurement is expressed.
        value_type (type): The expected data type of the measurement value.
        valid_time (str): The time or period for which the measurement is valid.
    """

    name: str
    unit: str
    value_type: type
    valid_time: str


# Definition of all the admissible measures
ADMISSIBLE_MEASURES = [
    Measure("precipitation", "mm", float, "Preceding hour sum"),
    Measure("temperature_2m", "°C", float, "Instant"),
    Measure("wind_speed_10m", "km/h", float, "Instant"),
    Measure("wind_direction_10m", "°", float, "Instant"),
    Measure("relative_humidity_2m", "%", float, "Instant"),
    Measure("shortwave_radiation", "W/m²", float, "Preceding hour mean"),
    Measure("global_tilted_irradiance", "W/m²", float, "Preceding hour mean"),
    Measure("et0_fao_evapotranspiration", "mm", float, "Preceding hour sum"),

    Measure("dew_point_2m", "°C", float, "Instant"),
    Measure("apparent_temperature", "°C", float, "Instant"),
    Measure("pressure_msl", "hPa", float, "Instant"),
    Measure("surface_pressure", "hPa", float, "Instant"),

    Measure("rain", "mm", float, "Preceding hour sum"),
    Measure("snowfall", "cm", float, "Preceding hour sum"),

    Measure("cloud_cover", "%", float, "Instant"),
    Measure("cloud_cover_low", "%", float, "Instant"),
    Measure("cloud_cover_mid", "%", float, "Instant"),
    Measure("cloud_cover_high", "%", float, "Instant"),

    Measure("direct_radiation", "W/m²", float, "Preceding hour mean"),
    Measure("direct_normal_irradiance", "W/m²", float, "Preceding hour mean"),
    Measure("diffuse_radiation", "W/m²", float, "Preceding hour mean"),

    Measure("sunshine_duration", "s", float, "Preceding hour sum"),

    Measure("wind_speed_100m", "km/h", float, "Instant"),
    Measure("wind_direction_100m", "°", float, "Instant"),
    Measure("wind_gusts_10m", "km/h", float, "Instant"),

    Measure("weather_code", "WMO code", int, "Instant"),

    Measure("snow_depth", "m", float, "Instant"),
    Measure("vapour_pressure_deficit", "kPa", float, "Instant"),

    Measure("soil_temperature_0_to_7cm", "°C", float, "Instant"),
    Measure("soil_temperature_7_to_28cm", "°C", float, "Instant"),
    Measure("soil_temperature_28_to_100cm", "°C", float, "Instant"),
    Measure("soil_temperature_100_to_255cm", "°C", float, "Instant"),

    Measure("soil_moisture_0_to_7cm", "m³/m³", float, "Instant"),
    Measure("soil_moisture_7_to_28cm", "m³/m³", float, "Instant"),
    Measure("soil_moisture_28_to_100cm", "m³/m³", float, "Instant"),
    Measure("soil_moisture_100_to_255cm", "m³/m³", float, "Instant"),
]


# PAST WEATHER OBSERVATIONS ####################################################


def to_utc_safe(dt: datetime, assume_tz=ZoneInfo("UTC")) -> datetime:
    if dt.tzinfo is None or dt.tzinfo.utcoffset(dt) is None:
        dt = dt.replace(tzinfo=assume_tz)
    return dt.astimezone(timezone.utc)


def prepare_hourly_raw_observations(hourly_dict: dict[str, list[float]]) -> DataFrame:
    """
    Prepare hourly raw observations from the Observations instance.

    Returns:
        DataFrame: A Polars DataFrame containing the hourly observations.
    """
    return (
        DataFrame(hourly_dict)
        .rename({"time": "dt_target"})
        .with_columns(col("dt_target").str.to_datetime(format="%Y-%m-%dT%H:%M"))
    )


@dataclass
class Observations:
    """
    Data class representing weather observations with metadata.
    Records are hourly data.

    Attributes:
        lat_lon (tuple): Latitude and longitude coordinates as (float, float).
        elevation (float): Elevation in meters.
        units (dict): Dictionary mapping measurement names to their units.
        records (DataFrame): Polars DataFrame containing the weather records.
        regular_time (bool): Whether the time series has regular intervals.
    """

    lat_lon: tuple[float, float]
    elevation: float
    units: list[str]
    values: DataFrame
    regular_time: bool

    @staticmethod
    def is_regular_time(values: DataFrame):
        # Evaluate if regular time
        # Test the type of dt_target
        if not isinstance(values["dt_target"].dtype, Datetime):
            raise TypeError("Observation dt_target must be of type Datetime.")
        # Test uniqueness
        if values.shape[0] != values["dt_target"].n_unique():
            raise ValueError("Observation dt_target are not unique.")

        # Test if missing in dates
        if values["dt_target"].is_null().any():
            raise ValueError("Observation dt_target have one or more missing value.")

        # Test if all the differences at delta (1) are one unique duration

        return values["dt_target"].diff().drop_nulls().value_counts().shape[0] == 1

    @classmethod
    def get_obs(
        cls,
        lat_lon: tuple[float, float],
        measure_names: list[str],
        start_date: datetime,
        end_date: datetime,
        api_key: str | None = None,
        max_retries: int = 2,
        retry_delay: float = 2.0,
        caching: bool = False,
        verbose: bool = False,
    ):
        """
        Retrieve weather observations from Open-Meteo API (hourly data) for a specified location and time period.

        This class method fetches historical weather data from the Open-Meteo archive API,
        supporting both free and commercial API access. It validates input parameters,
        constructs the API request, handles the response, and returns structured observations.

        Args:
            lat_lon (tuple[float, float]): Latitude and longitude coordinates as a tuple.
            measure_names (list[str]): List of weather measurement names to retrieve.
                Must be from the ADMISSIBLE_MEASURES measures list.
            start_date (datetime): Start date for the data retrieval period. By default supposes UTC if no timezone provided.
            end_date (datetime): End date for the data retrieval period. By default supposes UTC if no timezone provided.
            api_key (str | None, optional): API key for commercial access. If None,
                uses the free archive API. Defaults to None.
            max_retries (int, optional): Maximum number of retry attempts for failed requests.
                Defaults to 2.
            retry_delay (float, optional): Delay in seconds between retry attempts.
                Defaults to 2.0.
            verbose (bool, optional): If True, prints the query URL for debugging.
                Defaults to False.

        Returns:
            cls: An instance of the class containing the retrieved weather observations
            with location coordinates, elevation, measurement units, hourly values,
            and regularity information.
        """
        # Check arguments
        # Is it an admissible measure?
        diff = set(measure_names) - set([n.name for n in ADMISSIBLE_MEASURES])
        if diff:
            raise ValueError(f"Not admissible measure names: {diff}")

        # Build query
        if api_key is None:
            base_url = "https://archive-api.open-meteo.com/v1/archive"
        elif isinstance(api_key, str):
            base_url = "https://customer-archive-api.open-meteo.com/v1/archive"
        else:
            raise ValueError("API key must be a string or None")
        params = {
            "latitude": lat_lon[0],
            "longitude": lat_lon[1],
            "start_date": to_utc_safe(start_date).strftime("%Y-%m-%d"),
            "end_date": to_utc_safe(end_date).strftime("%Y-%m-%d"),
            "hourly": ",".join(measure_names),
            "timezone": "GMT",
        }
        # If an API key is provided, add it to the request parameters
        if api_key is not None:
            params["apikey"] = api_key

        # Prepare the request
        req = Request("GET", base_url, params=params)
        prepared = req.prepare()
        # Print the full URL before sending the request
        if verbose:
            info("Query URL:", prepared.url)

        if not caching:
            with Session() as session:
                response = session.send(prepared)
        else:
            # Use a shared session instance for caching
            if not hasattr(cls, "_cached_session"):
                cls._cached_session = CachedSession(
                    backend="memory", expire_after=3600 * 24
                )

            response = cls._cached_session.send(prepared)
        # Explicit error if error in response json
        try:
            if "error" in response.json():
                raise ValueError(f"Request error: {response.json()['reason']}")
        finally:
            response.raise_for_status()

        # Parse and prepare output
        resp_dict = response.json()

        # Prepare Observations init
        # unit
        units = resp_dict["hourly_units"]
        units.pop("time", None)
        # values
        hourly_values = prepare_hourly_raw_observations(resp_dict["hourly"])

        # Prepare the observations with the hourly table
        obs = cls(
            (resp_dict["latitude"], resp_dict["longitude"]),
            resp_dict["elevation"],
            units,
            hourly_values,
            Observations.is_regular_time(hourly_values),  # Evaluated in post init
        )

        return obs
