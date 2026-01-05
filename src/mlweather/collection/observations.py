from datetime import datetime
from polars import DataFrame, Datetime, lit
from mlweather.collection.records import Records
from mlweather.collection.utils import to_utc_safe


# A SPECIALIZATION OF RECORDS FOR OBSERVATIONS #################################


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

    def __repr__(self) -> str:
        return (
            f"Observations of location {self.lat_lon} with {self.record_table.shape[0]:,} hourly entries.\n"
            f"Elevation: {self.elevation} m\n"
            f"Units: {self.units}\n"
            f"Record Table:\n"
            f"{self.record_table}"
        )

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
            lit(None).cast(Datetime("us", "UTC")).alias("init_datetime")
        )
        # Reorder and sort columns
        hourly_values = Records.reorder_columns(hourly_values).sort(
            "init_datetime", "valid_datetime"
        )

        # Check regular time grid within init_datetime
        Records.is_obs_regular_time(hourly_values)

        # Prepare the observations with the hourly table
        obs = cls(
            (resp_dict["latitude"], resp_dict["longitude"]),
            resp_dict["elevation"],
            units,
            hourly_values,
        )

        return obs
