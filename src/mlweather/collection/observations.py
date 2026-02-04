from datetime import datetime, timedelta
from polars import DataFrame, Datetime, concat, lit
from mlweather.collection.records import Records
from mlweather.collection.utils import to_utc_safe


# A SPECIALIZATION OF RECORDS FOR OBSERVATIONS #################################


class Observations(Records):
    """
    Class representing weather observations with metadata.
    Records are hourly data.

    Attributes:
        lat_lon (tuple): Latitude and longitude coordinates as (float, float). They are the effective locations used for the data collection, not the queried ones. To allow to monitor the real sampling location of observations.
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
            f"elevation: {self.elevation} m\n"
            f"units: {self.units}\n"
            f"record_table:\n"
            f"{self.record_table}"
        )

    @staticmethod
    def bind_observations(observations_slices: list["Observations"]) -> "Observations":
        # Combine all slices into a single record_table
        combined_record_table = concat(
            [obs_slice.record_table for obs_slice in observations_slices],
            how="vertical",
        ).sort("valid_datetime")

        # Create a new Observations instance with the combined data
        combined_observations = Observations(
            lat_lon=observations_slices[0].lat_lon,
            elevation=observations_slices[0].elevation,
            units=observations_slices[0].units,
            record_table=combined_record_table,
        )

        # Check regular time grid
        Records.is_regular_time(combined_record_table)

        return combined_observations

    @classmethod
    def collect(
        cls,
        lat_lon: tuple[float, float],
        variables_names: list[str],
        start: datetime,
        end: datetime,
        *,
        api_key: str | None = None,
        cache_enabled: bool = True,
        cache_expire_after: int = 86400 * 8,
        cache_sqlite_filename: str = "api_cache.sqlite",
        query_by_period_slices: bool = False,
        period_slice_days: int = 31 * 3,
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
            start (datetime): Start date for the data retrieval period. By default supposes UTC if no timezone provided.
            end (datetime): End date for the data retrieval period. By default supposes UTC if no timezone provided.
            api_key (str | None, optional): API key for commercial access. If None,
                uses the free archive API. Defaults to None.
            cache_enabled (bool, optional): If True, enables caching of API responses to
                reduce redundant network calls. Defaults to True.
            cache_expire_after (int, optional): Time in seconds after which the cached
                responses expire. Defaults to 86400 * 8 (8 days).
            cache_sqlite_filename (str, optional): Filename for the SQLite cache database.
                Defaults to "api_cache.sqlite".
            query_by_period_slices (bool, optional): If True, queries data in slices of periods.
                Defaults to False.
            period_slice_days (int, optional): Number of days for each period slice when
                querying by period slices. Defaults to 31 * 3 (approximately 3 months).
            verbose (bool, optional): If True, enables verbose logging of the data retrieval process.
        Returns:
            cls: An instance of the class containing the retrieved weather observations
            with location coordinates, elevation, measurement units, hourly values,
            and regularity information.
        """
        # Check arguments
        # start and end should be datetime objects
        if not isinstance(start, datetime) or not isinstance(end, datetime):
            raise TypeError("start and end must be datetime objects.")
        # control the variable names
        cls.are_var_names_valid(variables_names)
        # control the length of period_slice_days
        if period_slice_days < 7:
            raise ValueError("period_slice_days must be at least 7 days.")

        # Prepare time slices: cut into periods if requested
        if query_by_period_slices:
            # Extrem points of the full period
            full_period_start = to_utc_safe(start)
            full_period_end = to_utc_safe(end)
            # Prepare period slices
            period_slices = []
            slice_start = full_period_start
            while slice_start < full_period_end:
                slice_end = min(
                    slice_start + timedelta(days=period_slice_days),
                    full_period_end,
                )
                period_slices.append((slice_start, slice_end))
                slice_start = slice_end + timedelta(days=1)
        else:
            period_slices = [(to_utc_safe(start), to_utc_safe(end))]

        # Collect and prepare all slices (only one single if no slicing)
        observations_for_slices = []
        for slice_start, slice_end in period_slices:
            # Prepare params
            params = {
                "latitude": lat_lon[0],
                "longitude": lat_lon[1],
                "start_date": slice_start.strftime("%Y-%m-%d"),
                "end_date": slice_end.strftime("%Y-%m-%d"),
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
                cache_enabled=cache_enabled,
                cache_expire_after=cache_expire_after,
                cache_sqlite_filename=cache_sqlite_filename,
                verbose=verbose,
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

            # Check regular time grid within init_datetime (within the slice)
            Records.is_regular_time(hourly_values)

            # Prepare the observations instance and collect it
            obs = cls(
                (resp_dict["latitude"], resp_dict["longitude"]),
                resp_dict["elevation"],
                units,
                hourly_values,
            )
            observations_for_slices.append(obs)

        return Observations.bind_observations(observations_for_slices)
