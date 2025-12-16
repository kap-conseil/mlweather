from dataclasses import dataclass
from polars import Expr, DataFrame, col, len as length, when, Series, Datetime
from polars._utils.convert import parse_as_duration_string
import re

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

    def generate_features(
        self,
        *,
        observations: Observations | None = None,
        forecasts: Forecasts | None = None,
    ) -> DataFrame:
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
            agg_df = []
            for op in self.aggregations:
                print(f"Applying aggregation period: {op.observation_period}")
                if op.observation_period is None:
                    raise ValueError(
                        "Observation aggregations require an observation_period value."
                    )
                # Apply the expression
                temp_agg = observations.record_table.rolling(
                    index_column="valid_datetime",
                    period=op.observation_period,
                    offset=None,
                    closed="right",
                    group_by="init_datetime",
                ).agg(length().alias("length"), op.expression.alias(op.get_var_label()))

                # Replace incomplete aggregations with missing to make sure incomplete aggs are not kept
                temp_agg = temp_agg.with_columns(
                    when(col("length") == op.get_step_in_obs_period())
                    .then(op.get_var_label())
                    .otherwise(None)
                    .alias(op.get_var_label())
                )

                print(temp_agg)
        #         # Drop the length column to allow repetited joins
        #         temp_agg = temp_agg.drop("length")

        #         # Store the individual aggregations before joining them
        #         agg_df.append(temp_agg)

        #     # Join all the individual aggregations into a single DataFrame
        #     # Init the left side with dt before succeively joining the aggregations
        #     out = obs_values.select(
        #         "dt_target"
        #     )  # Ensure dt_target is selected for joining
        #     for i in range(len(agg_df)):
        #         # All aggregations are joined to the left side
        #         # Full join
        #         out = out.join(agg_df[i], on="dt_target", how="left")
        # elif observations is None and isinstance(forecasts, Forecasts):
        #     pass
        # else:
        #     raise TypeError("Both observations and forecast are not supported types")
        # # Implementation of feature generation logic goes here
        return DataFrame()


@dataclass
class AggConfig:
    """
    Configuration for aggregation operations.
    Each entry is a tuple of (aggregation operation, list of periods).
    """

    operation: Expr
    period: str

    def get_var_label(self) -> str:
        """
        Clean the string of the polar expression by replacing non-alphanumeric characters with underscores
        and removing multiple consecutive underscores, then add duration
        """
        # Keep alpha num only
        expr_str = re.sub(r"[^a-zA-Z0-9]+", "_", str(self.operation))
        # Remove col mentions and leading/trailing underscores
        expr_str = expr_str.replace("col", "").strip("_")
        # Remove multiple underscores
        expr_str = re.sub(r"_+", "_", expr_str)

        # Add the duration to the label
        cleaned = f"{expr_str}_{parse_as_duration_string(self.period)}"

        return cleaned


# def get_source_period(dt_serie: Series):
#     """
#     Get the number of source periods in the aggregation period
#     """
#     # Check that the type of the Polar Series is a Datetime
#     if dt_serie.dtype != Datetime:
#         raise TypeError("The series must be of type Datetime")

#     # Extract unique time diffs
#     time_diffs = dt_serie.diff().drop_nulls().value_counts()["dt_target"].to_list()
#     # Check that we have only one duration
#     if len(time_diffs) != 1:
#         raise ValueError(
#             f"The time series does not have a unique duration step => {time_diffs}"
#         )
#     return time_diffs[0]


def aggregate(obs_values: DataFrame, agg_config: list[AggConfig]) -> DataFrame:
    """
    Aggregate observations based on the provided aggregation configuration.

    Parameters:
        obs_values (DataFrame): The DataFrame containing observation values.
        agg_config (list[AggConfig]): List of aggregation configurations.

    Returns:
        DataFrame: DataFrame.
    """
    # Perform all individual aggregations
    agg_df = []
    for op in agg_config:
        # Apply the expression
        temp_agg = obs_values.rolling(
            index_column="dt_target",
            period=op.period,
        ).agg(length().alias("length"), op.operation.alias(op.get_var_label()))
        # Control incomplete aggregations
        expected_length = op.period / get_dur_step(obs_values["dt_target"])
        # Check if the float is convertible to a int to ensure round periods
        if not expected_length % 1 == 0:
            raise ValueError(
                f"Period {op.period} is not evenly divisible by time step. Expected length: {expected_length}"
            )
        # Replace incomplete aggregations with missing to make sure incomplete aggs are not kept
        temp_agg = temp_agg.with_columns(
            when(col("length") == int(expected_length))
            .then(op.get_var_label())
            .otherwise(None)
            .alias(op.get_var_label())
        )

        # Drop the length column to allow repetited joins
        temp_agg = temp_agg.drop("length")

        # Store the individual aggregations before joining them
        agg_df.append(temp_agg)

    # Join all the individual aggregations into a single DataFrame
    # Init the left side with dt before succeively joining the aggregations
    out = obs_values.select("dt_target")  # Ensure dt_target is selected for joining
    for i in range(len(agg_df)):
        # All aggregations are joined to the left side
        # Full join
        out = out.join(agg_df[i], on="dt_target", how="left")

    return out
