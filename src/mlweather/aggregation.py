from functools import reduce
from polars import (
    Expr,
    DataFrame,
    LazyFrame,
    col,
    collect_all,
    concat,
    datetime_ranges,
    lit,
    when,
    Series,
    selectors as cs,
)
from polars._utils.convert import parse_as_duration_string
import re
from datetime import datetime, timedelta
from geopy.distance import geodesic

from mlweather.collection.observations import Observations
from mlweather.collection.forecasts import Forecasts


# DEFINE THE TRANSFORMATIONS TO PERFORM ON THE WEATHER DATA ####################
class Aggregation:
    """
    Class for defining a single aggregation operation of a given weather variable.
    It contains the polar expression to apply to the variable and the aggregation period.
    Aggregation period should be an integer multiple of the source data period (1 hour).
    Args:
        expression (Expr): Polars Expression defining the aggregation operation. See [Polars Expressions](https://docs.pola.rs/user-guide/concepts/expressions-and-contexts/).
        observation_period (timedelta | None): Aggregation period for observations as a timedelta object.
        forecast_period (timedelta | None): Aggregation period for forecasts as a timedelta object.
    """

    def __init__(
        self,
        expression: Expr,
        *,
        observation_period: timedelta | None = None,
        forecast_period: timedelta | None = None,
    ) -> None:
        self.expression = expression
        self.observation_period = observation_period
        self.forecast_period = forecast_period

    def __repr__(self) -> str:
        return (
            f"Aggregation(\n"
            f"    expression={self.expression},\n"
            f"    observation_period={self.observation_period},\n"
            f"    forecast_period={self.forecast_period}\n"
            f")"
        )

    @staticmethod
    def get_steps_in_agg_periods(period: timedelta) -> int:
        """
        Get the number of source periods in the aggregation period assuming source data period is 1 hour.
        Args:
            period (timedelta): Aggregation period as a timedelta object.
        Returns:
            int: Number of source periods in the aggregation period.
        Raises:
            ValueError: If the aggregation period is not an integer multiple of the source data period.
        """
        # Check that the timedalta is > 0
        if period < timedelta(0):
            raise ValueError(
                f"The aggregation period {period} must be equal or greater than zero"
            )
        # Assuming source data period is 1 hour
        source_period = timedelta(hours=1)
        if period % source_period != timedelta(0):
            raise ValueError(
                f"The aggregation period {period} is not an integer multiple of the source data period {source_period}"
            )

        return int(period / source_period)

    def get_steps_in_obs_period(self) -> int:
        """
        Get the number of source periods in the observation aggregation period.
        Returns:
            int: Number of source periods in the observation aggregation period.
        Raises:
            ValueError: If observation_period is None.
        """
        if self.observation_period is None:
            raise ValueError("observation_period is None")

        return self.get_steps_in_agg_periods(self.observation_period)

    def get_steps_in_fore_period(self) -> int:
        """
        Get the number of source periods in the forecast aggregation period.
        Returns:
            int: Number of source periods in the forecast aggregation period.
        Raises:
            ValueError: If forecast_period is None.
        """
        if self.forecast_period is None:
            raise ValueError("forecast_period is None")

        return self.get_steps_in_agg_periods(self.forecast_period)


class FeatureGenerator:
    """
    Class for defining a feature generator that applies multiple aggregation operations.
    Args:
        aggregations (list[Aggregation]): List of Aggregation instances defining the operations to apply.
    """

    def __init__(self, aggregations: list[Aggregation]) -> None:
        if len(aggregations) == 0:
            raise ValueError("aggregations cannot be an empty list")
        # All itm of list[Aggregation] should be Aggregation instances
        for agg in aggregations:
            if not isinstance(agg, Aggregation):
                raise TypeError(
                    "All items in aggregations must be instances of Aggregation"
                )
        self.aggregations = aggregations

    def __repr__(self) -> str:
        # Display all aggregations
        aggs_str = ",\n  ".join([repr(agg) for agg in self.aggregations])

        return f"FeatureGenerator(\nAggregations:\n  {aggs_str}\n)"

    @staticmethod
    def _prepare_all_records(
        obs: DataFrame | None, fore: DataFrame | None
    ) -> LazyFrame:
        # Replicate the forecast data to hourly frequency by self merging on the hourly spread dates
        # Add hours to dates based on init_datetime
        # Insert date ranges (hours within the day) withi the cells of a new column then explode
        non_meteo_vars = [
            "init_datetime",
            "valid_datetime",
            "init_datetime_raw",
            "init_datetime_raw_end",
        ]

        # Lazify the dataframe and handle None cases
        # First prevent that both obs and forecast are None at the same time
        if obs is None and fore is None:
            raise ValueError(
                "Both observations and forecasts cannot be None. At least one of them should be provided."
            )
        # Obs side
        if obs is None:
            obs_side = DataFrame().lazy()
        else:
            # Add id to observations
            obs_side = (
                obs.lazy()
                .sort("valid_datetime")
                .with_row_index("id")
                .with_columns(("observation-" + col("id").cast(str)).alias("id"))
            )

        # Forecast side: with hourly replication of the daily forecasts:
        # daily forecasts (at 00:00) are carried forward over the day
        if fore is None:
            forecast_side = DataFrame().lazy()
        else:
            forecast_side = (
                # Add id
                fore.lazy()
                .with_row_index("id")
                .with_columns(("forecast-" + col("id").cast(str)).alias("id"))
                # Gerenate hourly init_datetime from daily init_datetime using datetime ranges, then exploded vertically
                # Replicate the  daily computeed forecast to intraday steps (hourly)
                .rename({"init_datetime": "init_datetime_raw"})
                # Generate datetime ranges for each init_datetime (24 hour with the daily init_datetime)
                .with_columns(
                    (col("init_datetime_raw") + timedelta(hours=23)).alias(
                        "init_datetime_raw_end"
                    )
                )
                .with_columns(
                    datetime_ranges(
                        "init_datetime_raw",
                        "init_datetime_raw_end",
                        interval=timedelta(hours=1),
                        time_zone="UTC",
                    ).alias("init_datetime")
                )
                # From compact (nested) form to long form
                .explode("init_datetime")
                # Reorder to have non weather variables first
                .select(*non_meteo_vars, cs.exclude(non_meteo_vars))
                # Then sorted by valid and init datetime
                .sort(["valid_datetime", "init_datetime"])
            )

        # Assemble both sides
        all_records = concat([obs_side, forecast_side], how="diagonal")

        return all_records

    def unique_periods_sets(self) -> set:
        periods = [
            (agg.observation_period, agg.forecast_period) for agg in self.aggregations
        ]

        return set(periods)

    def get_same_period_aggregations(
        self,
    ) -> list[list[Aggregation]]:
        # Create collector of same periods groups
        groups_for_aggs = []
        # Iterate over all aggregations
        for period_set in self.unique_periods_sets():
            # Extract current aggregations for the period set
            current_aggregations = [
                op
                for op in self.aggregations
                if (op.observation_period, op.forecast_period) == period_set
            ]
            groups_for_aggs.append(current_aggregations)

        return groups_for_aggs

    @staticmethod
    def label_periods(
        observation_period: timedelta | None = None,
        forecast_period: timedelta | None = None,
    ) -> str:
        """
        Create a label for the aggregation periods.
        Args:
            observation_period (timedelta | None): Observation aggregation period. None if no observations and only forecasts.
            forecast_period (timedelta | None): Forecast aggregation period. None if no forecasts and only observations.
        Returns:
            str: Label for both aggregation periods.
        """
        # Prepare the duration string
        if observation_period is not None:
            obs_string = f"obs_{parse_as_duration_string(observation_period)}"
        else:
            obs_string = None
        if forecast_period is not None:
            fore_string = f"fore_{parse_as_duration_string(forecast_period)}"
        else:
            fore_string = None

        # Add the duration to the label
        period_label = "_" + "_".join(filter(None, [obs_string, fore_string]))

        return period_label

    @staticmethod
    def convert_expression_to_label(expression: Expr | str) -> str:
        """
        Transform the polar expression by replacing non-alphanumeric characters with underscores
        and removing multiple consecutive underscores to get a var label for the aggregated variable.
        Returns:
            str: Label for the expression part.
        """
        # Keep alpha num only
        expr_str = re.sub(r"[^a-zA-Z0-9]+", "_", str(expression))
        # Remove col mentions and leading/trailing underscores
        expr_str = expr_str.replace("col", "").strip("_")
        # Remove multiple underscores
        expr_str = re.sub(r"_+", "_", expr_str)

        return expr_str

    @staticmethod
    def get_var_label(
        expression: Expr | str,
        observation_period: timedelta | None = None,
        forecast_period: timedelta | None = None,
    ) -> str:
        """
        Transform the polar expression by replacing non-alphanumeric characters with underscores
        and removing multiple consecutive underscores, then add duration
        to get a var label for the aggregated variable.
        Returns:
            str: Clean variable label.
        """
        # Prepare parts
        expr_str = FeatureGenerator.convert_expression_to_label(expression)
        periods_label = FeatureGenerator.label_periods(
            observation_period, forecast_period
        )

        # Add the duration to the label
        full_label = f"{expr_str}{periods_label}"

        return full_label

    @staticmethod
    def _control_expressions() -> list[Expr]:
        return [
            # Count rows for obs and forecasts
            col("valid_datetime")
            .filter(col("init_datetime").is_null())
            .count()
            .alias("control_observations_n_rows"),
            col("valid_datetime")
            .filter(col("init_datetime").is_not_null())
            .count()
            .alias("control_forecasts_n_rows"),
            # Identify the min and max dates in the aggregation of both sides
            # observations
            col("valid_datetime")
            .filter(col("init_datetime").is_null())
            .min()
            .alias("control_observations_valid_datetime_min"),
            col("valid_datetime")
            .filter(col("init_datetime").is_null())
            .max()
            .alias("control_observations_valid_datetime_max"),
            # forecasts
            col("valid_datetime")
            .filter(col("init_datetime").is_not_null())
            .min()
            .alias("control_forecasts_valid_datetime_min"),
            col("valid_datetime")
            .filter(col("init_datetime").is_not_null())
            .max()
            .alias("control_forecasts_valid_datetime_max"),
        ]

    @staticmethod
    def _check_datetimes_are_in_records(
        features_datetimes: list[datetime], record_datetimes: Series
    ) -> None:
        """Check that all focal valid_datetime are present in the record datetimes."""
        # Valid datetimes present in the LazyFrame of weather records
        weather_valid_datetimes = set((record_datetimes.to_list()))
        # Find datetime expected in the features that are not in the weather records
        missing_datetimes = set(features_datetimes).difference(weather_valid_datetimes)
        if len(missing_datetimes) > 0:
            missing_datetimes_list = sorted(missing_datetimes)
            raise ValueError(
                "The following focal valid_datetime are missing in the weather records:\n"
                + "\n".join(
                    "       " + str(dt) for dt in sorted(missing_datetimes_list)
                )
            )

        return None

    @staticmethod
    def _filter_apply(
        all_records: LazyFrame,
        observation_period: timedelta | None,
        forecast_period: timedelta | None,
        expressions: list[Expr],
        focal_valid_datetime: datetime,
        control_aggregations=False,
    ) -> LazyFrame:
        """
        Filter the all_records LazyFrame based on the observation and forecast periods
        for a given focal_valid_datetime, then apply the aggregation expressions.
        All transformations (expressions) of same period set are applied (lazily) to all_records after filtering.
        Args:
            all_records (LazyFrame): LazyFrame containing all records (observations and forecasts).
            observation_period (timedelta | None): Observation aggregation period. None if no observations and only forecasts.
            forecast_period (timedelta | None): Forecast aggregation period. None if no forecasts and only observations.
            expressions (list[Expr]): List of Polars expressions to apply after filtering. Column names inside the expressions must match those in all_records.
            focal_valid_datetime (datetime): Focal valid datetime for filtering.
            control_aggregations (bool): Whether to include control aggregations in the output.
        Returns:
            LazyFrame: LazyFrame after filtering and applying the aggregation expressions.
        Raises:            ValueError: If both observation_period and forecast_period are None.
            ValueError: If both observation_period and forecast_period are None.
        """
        # Control args
        # Deal with 0 length periods
        # Check that at least one period is not none
        if observation_period is None and forecast_period is None:
            raise ValueError(
                "At least one of observation_period or forecast_period must be not None"
            )
        if observation_period is None:
            observation_period = timedelta(0)
        if forecast_period is None:
            forecast_period = timedelta(0)
        # Check periods (raise error if not integer mutiples of source period)
        if not ((forecast_period + observation_period) > timedelta(0)):
            raise ValueError(
                "At least one of observation_period or forecast_period must be greater than zero"
            )

        # Filter
        # Preprare bounds (fixed by the focal valid datetime)
        obs_upper_inside_bound = focal_valid_datetime - forecast_period
        obs_lower_outside_bound = (
            focal_valid_datetime - forecast_period - observation_period
        )
        forecast_lower_outside_bound = focal_valid_datetime - forecast_period
        # Count expected steps
        n_expected_steps_in_obs_period = Aggregation.get_steps_in_agg_periods(
            observation_period
        )
        n_expected_steps_in_fore_period = Aggregation.get_steps_in_agg_periods(
            forecast_period
        )
        # Prepare the control expressions
        if control_aggregations:
            control_exprs = FeatureGenerator._control_expressions()
            cols_to_drop = []
            cols_to_drop = [
                "control_observations_complete",
                "control_forecasts_complete",
            ]
        else:
            control_exprs = []
            cols_to_drop = [
                "control_observations_complete_agg_sample",
                "control_forecasts_complete_agg_sample",
            ]

        # Apply bounds
        plan_for_feature = (
            all_records.filter(
                # In obs:
                # from the valid_time - the forecast period (included),
                # back to included to the observation period and forecast period (bound excluded)
                (
                    col("init_datetime").is_null()  # only for observations
                    & (col("valid_datetime") <= obs_upper_inside_bound)
                    & (col("valid_datetime") > obs_lower_outside_bound)
                )
                |  # OR to switch to forecast side
                # In forecast:
                # for the init_times that match perfectly the firstcontrol_observations_valid_datetime_steps valid time
                # from the valid_time (included),
                # back to included to the observation period period (bound excluded)
                (
                    col("init_datetime").is_not_null()  # only for forecasts
                    & (col("init_datetime") == forecast_lower_outside_bound)
                    & (col("valid_datetime") <= focal_valid_datetime)
                    & (col("valid_datetime") > forecast_lower_outside_bound)
                )
            )
            .sort(["valid_datetime", "init_datetime"])
            # Apply functions: Controls and aggregation expressions
            .select(
                # Provide the focal time back
                lit(focal_valid_datetime).alias("valid_datetime"),
                # Aggregation on the full period
                (
                    (col("init_datetime").is_null().sum())
                    == n_expected_steps_in_obs_period
                ).alias("control_observations_complete_agg_sample"),
                (
                    (col("init_datetime").is_not_null().sum())
                    == n_expected_steps_in_fore_period
                ).alias("control_forecasts_complete_agg_sample"),
                # Controls
                *control_exprs,
                # Aggregations
                *[
                    # Expressions with proper labels
                    exp.alias(FeatureGenerator.convert_expression_to_label(exp))
                    for exp in expressions
                ],
            )
            # Only return the aggreaton value if both obs and forecast samples are complete, otherwise return nulls
            .with_columns(
                *[
                    when(
                        col("control_observations_complete_agg_sample")
                        & col("control_forecasts_complete_agg_sample")
                    )
                    .then(col(FeatureGenerator.convert_expression_to_label(exp)))
                    .otherwise(lit(None))
                    .alias(FeatureGenerator.convert_expression_to_label(exp))
                    for exp in expressions
                ]
            )
            .drop(*cols_to_drop)
        )
        return plan_for_feature

    _NO_MEASURES_COLUMNS = [
        "valid_datetime",
    ]

    def generate_features(
        self,
        features_datetimes: list[datetime],
        *,
        observations: Observations | None = None,
        forecasts: Forecasts | None = None,
        control_aggregations=False,
        obs_forecast_max_distance: float = 10.0,
    ):
        # Assemble the records (observations and forecast together) in a (lazy) all_records table
        if observations is None:
            obs = None
        else:
            obs = observations.record_table
        if forecasts is None:
            fore = None
        else:
            fore = forecasts.record_table
        # If mixed obs and forecasts, check that the locations of the two does not differ more than 5 km
        if isinstance(observations, Observations) and isinstance(forecasts, Forecasts):
            if (
                geodesic(observations.lat_lon, forecasts.lat_lon).km
                > obs_forecast_max_distance
            ):
                raise ValueError(
                    f"""
                    The locations of the observations and forecasts differ by more than {obs_forecast_max_distance} km.
                    The location of Observations is {observations.lat_lon} and the location of Forecasts is {forecasts.lat_lon}.
                    The distance (in km) found between them is {geodesic(observations.lat_lon, forecasts.lat_lon).km}.
                    """
                )
        # Bind properly the two sources
        all_records = self._prepare_all_records(obs, fore)
        # Work by pair of periods (obs and forecast), filter and apply the aggregation expressions over all focal valid_datetime
        # collector for the dataframes per period set
        all_aggregations = []
        # iterate over all period sets
        for same_period_aggs in FeatureGenerator(
            self.aggregations
        ).get_same_period_aggregations():
            # store the results: as many one-rowed tables as focal features_datetimes
            plans_for_features = []
            # Extract agg periods for obs and forecast
            obs_period = same_period_aggs[0].observation_period
            fore_period = same_period_aggs[0].forecast_period
            # Collect the expressions to latter apply
            expressions = [agg.expression for agg in same_period_aggs]

            # Check the dates
            FeatureGenerator._check_datetimes_are_in_records(
                features_datetimes,
                all_records.select("valid_datetime").unique().collect().to_series(),
            )
            # By requested focal valid_datetime:
            #  1/ filter
            #  2/ apply the expressions (+ store controls if requested)
            for focal_valid_datetime in features_datetimes:
                plans_for_features.append(
                    FeatureGenerator._filter_apply(
                        all_records,
                        obs_period,
                        fore_period,
                        expressions,
                        focal_valid_datetime,
                        control_aggregations=control_aggregations,
                    )
                )
            # Collect all results for the current period set: One table (single rowed) per focal valid_datetime
            # This is the "best" optimization tradeoff found so far:
            # collect_all and bind them into a dataframe: does the job faster than a map / for loop outside the polars framework
            collected_aggregations = concat(
                collect_all(plans_for_features), how="vertical"
            )
            # Rename the columns to proper feature labels
            collected_aggregations = collected_aggregations.rename(
                {
                    col_name: FeatureGenerator.get_var_label(
                        col_name, obs_period, fore_period
                    )
                    for col_name in collected_aggregations.columns
                    if col_name not in FeatureGenerator._NO_MEASURES_COLUMNS
                }
            )

            # Store it
            all_aggregations.append(collected_aggregations)

        # Join all the aggregations on valid_datetime (only), recursively
        features: DataFrame = reduce(
            lambda left, right: left.join(right, on="valid_datetime", validate="1:1"),
            all_aggregations,
        )

        return features.sort("valid_datetime")
