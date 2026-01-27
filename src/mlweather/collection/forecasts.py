from datetime import datetime, timedelta
from functools import reduce
import re
from polars import DataFrame, col, concat, selectors as cs
from mlweather.collection.records import Records
from mlweather.collection.utils import to_utc_safe


class Forecasts(Records):
    """
    Class representing weather forecasts with metadata.
    Records are hourly data.

    Attributes:
        lat_lon (tuple): Latitude and longitude coordinates as (float, float). They are the effective locations used for the data collection, not the queried ones. To allow to monitor the real sampling location of observations.
        elevation (float): Elevation in meters.
        units (dict): Dictionary mapping measurement names to their units.
        forecast_horizon_days_max (int): Maximum number of past days of forecast horizon to retrieve. Forecast collection start from current day forecast (horizon = 0 day) to forecast with horizon up to forecast_horizon_days_max (included).
        records (DataFrame): Polars DataFrame containing the weather records.
    """

    def __init__(
        self,
        lat_lon: tuple[float, float],
        elevation: float,
        units: dict[str, str],
        forecast_horizon_days_max: int,
        record_table: DataFrame,
    ) -> None:
        # First construct as parent
        super().__init__(lat_lon, elevation, units, record_table)
        # Child specific attributes can be added here if needed
        self.forecast_horizon_days_max = Forecasts.validate_forecast_horizon_days_max(
            forecast_horizon_days_max
        )

    def __repr__(self) -> str:
        return (
            f"Forecasts of location {self.lat_lon} with {self.record_table.shape[0]:,} hourly entries.\n"
            f"for forecast horizons of days ranging from 0 to {self.forecast_horizon_days_max}.\n"
            f"elevation: {self.elevation} m\n"
            f"units: {self.units}\n"
            f"record_table:\n"
            f"{self.record_table}"
        )

    @staticmethod
    def validate_forecast_horizon_days_max(
        forecast_horizon_days_max: int,
    ) -> int:
        """
        Validate the past days max for forecasts.
        Args:
            forecast_horizon_days_max (int): Maximum number of past forecast days to retrieve. Forecast collection start from current day forecast (horizon = 0 day) to forecast with horizon up to forecast_horizon_days_max (included).
        Returns:
            int: The validated past days max.
        Raises:
            ValueError: If the forecast_horizon_days_max is not a non-negative integer.
        """
        if (
            not isinstance(forecast_horizon_days_max, int)
            or forecast_horizon_days_max < 0
        ):
            raise ValueError(
                "forecast_horizon_days_max must be a non-negative integer."
            )

        return forecast_horizon_days_max

    @staticmethod
    def rename_day0_columns(
        record_table: DataFrame, queried_with_horizon_variables: list[str]
    ):
        """
        Rename the columns corresponding to previous_day0 to have a consistent naming
        convention with the other forecast horizons, i.e.revious days (i.e., adding _previous_day0 suffix).
        Args:
            record_table (DataFrame): The DataFrame containing the weather records.
            queried_with_horizon_variables (list[str]): List of variable names that were queried
                for previous days (with _previous_dayX suffix).
        Returns:
            DataFrame: The DataFrame with renamed columns for day 0 with now suffix previous_day0 variables (consistent var naming with other past days).
        """
        # Identify variables in day0: not the valid_datetime and not in queried past days (that include XXX_previous_day0 at the query, not into the hourly reords where the suffix is missing for day 0)
        days0_weather_variables = (
            set(record_table.columns)
            - {"valid_datetime"}
            - set(queried_with_horizon_variables)
        )
        # Create renaming mapping
        renaming_dict = {var: f"{var}_previous_day0" for var in days0_weather_variables}

        # Effectively rename
        return record_table.rename(renaming_dict)

    @staticmethod
    def melt_by_weather_variable(hourly_records: DataFrame):
        """
        Melt the hourly records DataFrame (wide) to have one row per valid_datetime and forecast_horizon (new long side)
        for each weather variable.
        Args:
            hourly_records (DataFrame): The DataFrame containing the weather records.
        Returns:
            list[DataFrame]: A list of DataFrames, each corresponding to a weather variable,
                melted by forecast_horizon.
        """
        # Identify the weather variables (without the _dayX suffix)
        groups_for_melt = {
            c.split("_day")[0]
            for c in hourly_records.columns
            if re.search(r"_day\d+$", c)
        }
        # By weather variable, melt the corresponding columns
        # 1/ select the valid_datetime and the columns for that weather var)
        # 2/ unpivot them to have valid_datetime, past_day, value
        # 3/ extract the past day number from the column name
        melted_records = [
            hourly_records.select(["valid_datetime", cs.starts_with(g)])
            .unpivot(
                cs.starts_with(g),
                index="valid_datetime",
                variable_name="days_forecast_horizon",
                value_name=g,
            )
            # Extract the horizon (past day) number from the column name
            .with_columns(col("days_forecast_horizon").str.extract(r"(\d+)$").cast(int))
            for g in groups_for_melt
        ]

        return melted_records

    @staticmethod
    def add_initial_datetime(hourly_values: DataFrame) -> DataFrame:
        """
        Add the initial datetime column to the hourly values DataFrame,
        based on the valid_datetime and days_forecast_horizon columns.
        It rounds down the valid_datetime to the zero hour of the day (UTC everywhere) and subtracts the past day (forecast horizon).
        Args:
            hourly_values (DataFrame): The DataFrame containing the weather records.
        Returns:
            DataFrame: The DataFrame with the added init_datetime column.
        """
        # Round down valid_datetime to day and subtract past days
        hourly_values = hourly_values.with_columns(
            (
                (
                    col("valid_datetime").dt.truncate("1d")
                    - (col("days_forecast_horizon") * timedelta(days=1))
                ).alias("init_datetime")
                # Remove technical column
            )
        ).drop("days_forecast_horizon")

        return hourly_values

    @staticmethod
    def bind_forecasts(forecasts_slices: list["Forecasts"]) -> "Forecasts":
        # Combine all slices into a single record_table
        combined_record_table = concat(
            [forecast_slice.record_table for forecast_slice in forecasts_slices],
            how="vertical",
        ).sort(["init_datetime", "valid_datetime"])

        # Create a new Forecasts instance with the combined data
        combined_forecasts = Forecasts(
            lat_lon=forecasts_slices[0].lat_lon,
            elevation=forecasts_slices[0].elevation,
            units=forecasts_slices[0].units,
            forecast_horizon_days_max=forecasts_slices[0].forecast_horizon_days_max,
            record_table=combined_record_table,
        )

        # Check regular time grid within init_datetime
        Records.is_regular_time(combined_record_table)

        return combined_forecasts

    @classmethod
    def collect(
        cls,
        lat_lon: tuple[float, float],
        variables_names: list[str],
        start: datetime,
        end: datetime,
        forecast_horizon_days_max: int,
        api_key: str | None = None,
        *,
        cache_enabled: bool = True,
        cache_expire_after: int = 86400 * 8,
        cache_sqlite_filename: str = "api_cache.sqlite",
        query_by_period_slices: bool = False,
        period_slice_days: int = 31 * 3,
    ):
        """
        Retrieve weather forecasts from Open-Meteo API (hourly data) for a specified location and a time period,
        given for the past days horizons of forecast.
        This class method fetches forecast weather data from the Open-Meteo previous runs API,
        supporting both free and commercial API access. It validates input parameters,
        constructs the API request, handles the response, and returns structured forecasts.
        Args:
            lat_lon (tuple[float, float]): Latitude and longitude coordinates as a tuple.
            variables_names (list[str]): List of weather measurement variable names to retrieve.
                Must be from the ADMISSIBLE_VARIABLES list.
            start (datetime): Start date for the data retrieval period. By default supposes UTC if no timezone provided.
            end (datetime): End date for the data retrieval period. By default supposes UTC if no timezone provided.
            forecast_horizon_days_max (int): Maximum number of past forecast days to retrieve. Forecast collection start from current day forecast (horizon = 0 day) to forecast with horizon up to forecast_horizon_days_max (inclusive).
            api_key (str | None, optional): API key for commercial access. If None,
                uses the free previous runs API. Defaults to None.
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
        Returns:
            cls: An instance of the class containing the retrieved weather forecasts
            with location coordinates, elevation, measurement units, hourly values,
            and past days range information.
        """
        # Check arguments
        # control the variable names
        cls.are_var_names_valid(variables_names)

        # control the length of period_slice_days
        if period_slice_days < 7:
            raise ValueError("period_slice_days must be at least 7 days.")

        # Define the variable names to query for all vars and previous days (all the non day0 variables)
        forecast_horizons_range = (
            0,
            Forecasts.validate_forecast_horizon_days_max(forecast_horizon_days_max),
        )
        # To query the open meteo API for forecasts at variable horizons, you must pass the weather variable with a suffix "_previous_dayX", including for day0
        all_measures_with_horizons = [
            f"{m}_previous_day{pd}"
            for m in variables_names
            for pd in range(forecast_horizons_range[0], forecast_horizons_range[1] + 1)
        ]

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
        forecasts_for_slices = []
        for slice_start, slice_end in period_slices:
            # Build query params
            params = {
                "latitude": lat_lon[0],
                "longitude": lat_lon[1],
                "start_date": slice_start.strftime("%Y-%m-%d"),
                "end_date": slice_end.strftime("%Y-%m-%d"),
                "hourly": ",".join(all_measures_with_horizons),
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
                cache_enabled=cache_enabled,
                cache_expire_after=cache_expire_after,
                cache_sqlite_filename=cache_sqlite_filename,
            )

            # Prepare Observations init
            units = resp_dict["hourly_units"]
            units.pop("time", None)

            # Prepare the hourly records and transform to long per meteo var (along the previous days)
            hourly_values = Forecasts.prepare_hourly_records(resp_dict)
            # Rename day0 variables and transform into long (over the past days), with several dataframes (one per weather variable)
            hourly_values = Forecasts.rename_day0_columns(
                hourly_values, all_measures_with_horizons
            )
            hourly_values = Forecasts.melt_by_weather_variable(hourly_values)
            # Join all the melted dataframes on valid_datetime and past_day,
            # to have a wide table but not column by past day
            hourly_values = reduce(
                lambda left, right: left.join(
                    right, on=["valid_datetime", "days_forecast_horizon"], how="inner"
                ),
                hourly_values,
            )

            # Add the init_datetime column: round down valid_datetime to day and subtract past days
            hourly_values = Forecasts.add_initial_datetime(hourly_values)
            # Clean columns and names
            # drop the useless past_day column
            hourly_values = (
                hourly_values
                # drop the "_previous_day0" suffix in columns names (only if ending with it)
                .rename(
                    {
                        c: c.removesuffix("_previous")
                        for c in hourly_values.columns
                        if c.endswith("_previous")
                    }
                )
            )

            # Reorder and sort columns
            hourly_values = Records.reorder_columns(hourly_values).sort(
                "init_datetime", "valid_datetime"
            )
            # Check regular time grid within init_datetime (within the slice    )
            Records.is_regular_time(hourly_values)

            # Prepare the forecasts instance and collect it
            fore = cls(
                (resp_dict["latitude"], resp_dict["longitude"]),
                resp_dict["elevation"],
                units,
                forecast_horizon_days_max,
                hourly_values,
            )
            forecasts_for_slices.append(fore)

        return Forecasts.bind_forecasts(forecasts_for_slices)
