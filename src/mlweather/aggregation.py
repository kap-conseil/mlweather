from dataclasses import dataclass
from polars import Expr, DataFrame, col, len as length, when, Series, Datetime
from polars._utils.convert import parse_as_duration_string
import re
from functools import reduce

from datetime import timedelta

from mlweather.collection.forecasts import Forecasts
from mlweather.collection.observations import Observations


class Aggregation:
    """
    Class for defining a single aggregation operation of a given weather variable.
    It contains the polar expression to apply to the variable and the aggregation period.
    Aggregation period should be an integer multiple of the source data period (1 hour).
    Args:
        operation (Expr): Polars Expression defining the aggregation operation. See [Polars Expressions](https://docs.pola.rs/user-guide/concepts/expressions-and-contexts/).
        observation_period (timedelta | None): Aggregation period for observations as a timedelta object.
        forecast_period (timedelta | None): Aggregation period for forecasts as a timedelta object.
    """

    def __init__(
        self,
        expression: Expr,
        observation_period: timedelta | None = None,
        forecast_period: timedelta | None = None,
    ) -> None:
        self.expression = expression
        self.observation_period = observation_period
        self.forecast_period = forecast_period

    def __repr__(self) -> str:
        return (
            f"Aggregation(\n"
            f"  expression={self.expression},\n"
            f"  observation_period={self.observation_period},\n"
            f"  forecast_period={self.forecast_period}\n"
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
        # Assuming source data period is 1 hour
        source_period = timedelta(hours=1)
        if period % source_period != timedelta(0):
            raise ValueError(
                f"The aggregation period {period} is not an integer multiple of the source data period {source_period}"
            )
        return int(period / source_period)

    def get_step_in_obs_period(self) -> int:
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

    def get_step_in_fore_period(self) -> int:
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

    def get_var_label(self) -> str:
        """
        Transform the polar expression by replacing non-alphanumeric characters with underscores
        and removing multiple consecutive underscores, then add duration
        to get a var label for the aggregated variable.
        Returns:
            str: Clean variable label.
        """
        # Keep alpha num only
        expr_str = re.sub(r"[^a-zA-Z0-9]+", "_", str(self.expression))
        # Remove col mentions and leading/trailing underscores
        expr_str = expr_str.replace("col", "").strip("_")
        # Remove multiple underscores
        expr_str = re.sub(r"_+", "_", expr_str)

        # Prepare the duration string
        if self.observation_period is not None:
            obs_string = f"obs_{parse_as_duration_string(self.observation_period)}"
        else:
            obs_string = None
        if self.forecast_period is not None:
            fore_string = f"fore_{parse_as_duration_string(self.forecast_period)}"
        else:
            fore_string = None

        # Add the duration to the label
        cleaned = f"{expr_str}_{'_'.join(filter(None, [obs_string, fore_string]))}"

        return cleaned


class FeatureGenerator:
    """
    Class for defining a feature generator that applies multiple aggregation operations.
    Args:
        aggregations (list[Aggregation]): List of Aggregation instances defining the operations to apply.
    """

    def __init__(self, aggregations: list[Aggregation]) -> None:
        self.aggregations = aggregations

    def unique_periods_sets(self) -> set:
        periods = [
            (agg.observation_period, agg.forecast_period) for agg in self.aggregations
        ]

        return set(periods)

    def generate_features(
        self,
        *,
        observations: Observations | None = None,
        forecasts: Forecasts | None = None,
    ):
        """
        Generate features based on the defined aggregations.
        Args:
            observations (Observations | None): Observations instance to aggregate.
            forecasts (Forecasts | None): Forecasts instance to aggregate.
        Returns:
            DataFrame: DataFrame containing the aggregated features.
        """
        if observations is None and forecasts is None:
            raise ValueError(
                "At least observations or forecasts must be provided in the generate function"
            )
        elif isinstance(observations, Observations) and forecasts is None:
            # Perform all individual aggregations
            # collect a dataframe by aggregation defined
            aggregated_variables = []
            # Iterate over all aggregations
            for period_set in self.unique_periods_sets():
                # Extract current aggregations for the period set
                current_aggregations = [
                    op
                    for op in self.aggregations
                    if (op.observation_period, op.forecast_period) == period_set
                ]
                # Prepare the expressions for the current period set
                ops_in_period_set = [
                    op.expression.alias(op.get_var_label())
                    for op in current_aggregations
                ]
                new_col_names = [op.get_var_label() for op in current_aggregations]

                print(ops_in_period_set)
                # for op in self.aggregations:
                # Check type for the rolling operation
                # Apply the expression, lazily until the end
                current_agg_table = (
                    # Rolling groups
                    observations.record_table.lazy().rolling(
                        index_column="valid_datetime",
                        period=period_set[0],
                        offset=None,
                        closed="right",
                        group_by="init_datetime",
                    )
                ).agg(
                    # Compute the control for dates included in the aggregation
                    col("valid_datetime")
                    .min()
                    .alias(
                        "valid_datetime_min_"
                        + f"obs_{parse_as_duration_string(period_set[0])}"
                    ),
                    col("valid_datetime")
                    .max()
                    .alias(
                        "valid_datetime_max_"
                        + f"obs_{parse_as_duration_string(period_set[0])}"
                    ),
                    # Length of agg period
                    length().alias(
                        "valid_datetime_length_"
                        + f"obs_{parse_as_duration_string(period_set[0])}"
                    ),
                    # Finally apply the expression
                    *ops_in_period_set,
                )
                # Replace incomplete aggregations with missing to make sure incomplete aggs are not kept for each period set aggregrations
                current_agg_table = current_agg_table.with_columns(
                    [
                        when(
                            col(
                                f"valid_datetime_length_obs_{parse_as_duration_string(period_set[0])}"
                            )
                            == current_aggregations[0].get_step_in_obs_period()
                        )
                        .then(col(col_name))
                        .otherwise(None)
                        .alias(col_name)
                        for col_name in new_col_names
                    ]
                ).collect()

                # Store the individual aggregations before joining them
                aggregated_variables.append(current_agg_table)
                features = reduce(
                    lambda left, right: left.join(
                        right,
                        on=["init_datetime", "valid_datetime"],
                        how="inner",
                        validate="1:1",
                        nulls_equal=True,
                    ),
                    aggregated_variables,
                )
        elif observations is None and isinstance(forecasts, Forecasts):
            pass
        else:
            raise TypeError("Both observations and forecast are not supported types")

        return features
