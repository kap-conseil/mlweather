from dataclasses import dataclass
from polars import Expr, DataFrame, col, len as length, when, Series, Datetime
from polars._utils.convert import parse_as_duration_string
import re


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


def get_dur_step(dt_serie: Series):
    """
    Get the number of source periods in the aggregation period
    """
    # Check that the type of the Polar Series is a Datetime
    if dt_serie.dtype != Datetime:
        raise TypeError("The series must be of type Datetime")

    # Extract unique time diffs
    time_diffs = dt_serie.diff().drop_nulls().value_counts()["dt_target"].to_list()
    # Check that we have only one duration
    if len(time_diffs) != 1:
        raise ValueError(
            f"The time series does not have a unique duration step => {time_diffs}"
        )
    return time_diffs[0]


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
