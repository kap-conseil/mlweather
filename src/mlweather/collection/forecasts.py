from datetime import datetime, timedelta
from functools import reduce
import re
from polars import DataFrame, col, selectors as cs
from mlweather.collection.records import Records
from mlweather.collection.utils import to_utc_safe


class Forecasts(Records):
    """
    Class representing weather forecasts with metadata.
    Records are hourly data.

    Attributes:
        lat_lon (tuple): Latitude and longitude coordinates as (float, float).
        elevation (float): Elevation in meters.
        units (dict): Dictionary mapping measurement names to their units.
        forecast_horizon_days_range (tuple[int, int]): Tuple indicating the range of horizons of forecast. First value of tuple given the smallest desired horizon. 0 is the min. The second value give the largest desired horizon (common max is 7, on Open-meteo). For instance (0, 3) means the variables are expected for forecast at 0 day ahead, 1 day ahead, 2 days ahead, and 3 days ahead.
        records (DataFrame): Polars DataFrame containing the weather records.
    """

    def __init__(
        self,
        lat_lon: tuple[float, float],
        elevation: float,
        units: dict[str, str],
        forecast_horizon_days_range: tuple[int, int],
        record_table: DataFrame,
    ) -> None:
        # First construct as parent
        super().__init__(lat_lon, elevation, units, record_table)
        # Child specific attributes can be added here if needed
        self.forecast_horizon_days_range = (
            Forecasts.validate_forecast_horizon_days_range(forecast_horizon_days_range)
        )

    def __repr__(self) -> str:
        return (
            f"Forecasts of location {self.lat_lon} with {self.record_table.shape[0]:,} hourly entries.\n"
            f"for forecast horizons of days ranging from {self.forecast_horizon_days_range[0]} to {self.forecast_horizon_days_range[1]}.\n"
            f"Elevation: {self.elevation} m\n"
            f"Units: {self.units}\n"
            f"Record Table:\n"
            f"{self.record_table}"
        )

    @staticmethod
    def validate_forecast_horizon_days_range(
        forecast_horizon_days_range: tuple[int, int],
    ) -> tuple[int, int]:
        """
        Validate the past days range for forecasts.
        Args:
            forecast_horizon_days_range (tuple[int, int]): Tuple indicating the range of past forecast days (start_day, end_day). The common max for end_day is 7.
        Returns:
            tuple[int, int]: The validated past days range.
        Raises:
            ValueError: If the forecast_horizon_days_range is not a tuple of two non-negative integers
                with end_day >= start_day.
        """
        if (
            not isinstance(forecast_horizon_days_range, tuple)
            or len(forecast_horizon_days_range) != 2
            or not all(isinstance(day, int) for day in forecast_horizon_days_range)
            or forecast_horizon_days_range[0] < 0
            or forecast_horizon_days_range[1] < forecast_horizon_days_range[0]
        ):
            raise ValueError(
                "forecast_horizon_days_range must be a tuple of two non-negative integers (start_day, end_day) with end_day >= start_day."
            )

        return forecast_horizon_days_range

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

    @classmethod
    def collect(
        cls,
        lat_lon: tuple[float, float],
        variables_names: list[str],
        start: datetime,
        end: datetime,
        past_forecast_days_range: tuple[int, int],
        api_key: str | None = None,
        verbose: bool = False,
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
            past_forecast_days_range (tuple[int, int]): Tuple indicating the range of past forecast days to retrieve (start_day, end_day).
            api_key (str | None, optional): API key for commercial access. If None,
                uses the free previous runs API. Defaults to None.
            verbose (bool, optional): If True, prints the query URL for debugging.
                Defaults to False.
        Returns:
            cls: An instance of the class containing the retrieved weather forecasts
            with location coordinates, elevation, measurement units, hourly values,
            and past days range information.
        """
        # Check arguments
        cls.are_var_names_valid(variables_names)
        forecast_horizons_range = Forecasts.validate_forecast_horizon_days_range(
            past_forecast_days_range
        )

        # Define the variable names to query for all vars and previous days (all the non day0 variables)
        # To query the open meteo API for forecasts at variable horizons, you must pass the weather variable with a suffix "_previous_dayX", including for day0
        all_measures_with_horizons = [
            f"{m}_previous_day{pd}"
            for m in variables_names
            for pd in range(forecast_horizons_range[0], forecast_horizons_range[1] + 1)
        ]

        # Build query params
        params = {
            "latitude": lat_lon[0],
            "longitude": lat_lon[1],
            "start_date": to_utc_safe(start).strftime("%Y-%m-%d"),
            "end_date": to_utc_safe(end).strftime("%Y-%m-%d"),
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
            verbose,
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
        # Check regular time grid within init_datetime
        Records.is_regular_time(hourly_values)

        return cls(
            (resp_dict["latitude"], resp_dict["longitude"]),
            resp_dict["elevation"],
            units,
            forecast_horizons_range,
            hourly_values,
        )
