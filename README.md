# MLWeather

**Version: 0.2.0**

A high-performance Python library for collecting and preparing ML-ready weather observations and forecasts.

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)

## Target uses

This package aims to collect and prepare complex meteorological features ready for machine learning:

-   over large periods at blazing speed,

-   with high precision data and a large variety of meteorological variables (observations and forecast),

-   with custom transformation of the variable;

This packages aims at:

-   **Collecting weather records** for past meteorological and forecasts for any location on Earth based on hourly measures for a large set of meteorological variables. The data source comes from [Open-Meteo](https://open-meteo.com/). Forecast are both current forecast (to predict) and past forecast to train you model. Then you can train you models on the data of the same nature: you train your model on forecast data because you will predict on forecast data.

-   **Aggregate forecasts, observations** or a constant mixture (e.g. X days of observations followed by Y days of forecasts) using **any function** to have aggregation (e.g. sum, mean, min,...for a given period) over a chosen period for every hour. It allows to combine **observations and forecasts** to have aggregation for periods for observations (e.g. X days) followed by forecasts (e.g. Y days). This approach provides features that exploit the last forecasts and assemble them to prior observations. It provides the average of temperatures mixing the available temperature forecasts and with the preceding observed temperature. You can have aggregations of observations and forecasts for custom meteorological ML features.

-   **Simulate forecast based aggregations** for period before available past forecast. If forecasts are used in predict operations, past forecast are needed to correctly train models (same signal in train and predict phases). Howver pas forecast do not come with long history. This package offers simulation tools for aggregation of meteorological variables that are based on past forecast, before the past forecast are available. It is based on the empirical distribution of the empirical error of the aggregation using the forecast (compared to the same aggregation based on observations only data).

## Source data

Weather data (both historical observations and forecasts - current and past ones) comes from the [Open-Meteo](https://open-meteo.com/) API, which provides access to weather data worldwide. Please consult their usage conditions to either use their free tier or commercial licenses ([Open-Meteo pricing](https://open-meteo.com/en/pricing)).

`mlweather` only use the hourly records available from Open-Meteo, as it this fine-grained records can be transformed and aggregated to larger time intervals (e.g. daily, weekly, monthly) as needed.

### Weather variables

Please note the two different key times :

`init_datetime` refers to the time when the forecast model was run or initialized.

`valid_datetime` refers the time for which 1/ the forecast predicts the state of the atmosphere (in case of current or past forecasts) and 2/ the time for which the state is observed (in case of observations).

For observations, the `init_datetime` equals the `valid_datetime` : you observe the current state (the forecast horizon is zero).

The following variables are available via `mlweather` (more [info](https://open-meteo.com/en/docs#api_documentation) on variables):

| Variable name | Unit | Type | Measurement |
|----|----|----|----|
| precipitation | mm | float | Sum of hour preceding the `valid_time` |
| temperature_2m | °C | float | Instant measure at `valid time` |
| wind_speed_10m | km/h | float | Instant measure at `valid time` |
| wind_direction_10m | ° | float | Instant measure at `valid time` |
| relative_humidity_2m | \% | float | Instant measure at `valid time` |
| shortwave_radiation | W/m² | float | Sum of hour preceding the `valid_time` |
| global_tilted_irradiance | W/m² | float | Sum of hour preceding the `valid_time` |
| et0_fao_evapotranspiration | mm | float | Sum of hour preceding the `valid_time` |
| dew_point_2m | °C | float | Instant measure at `valid time` |
| apparent_temperature | °C | float | Instant measure at `valid time` |
| pressure_msl | hPa | float | Instant measure at `valid time` |
| surface_pressure | hPa | float | Instant measure at `valid time` |
| rain | mm | float | Preceding hour sum |
| snowfall | cm | float | Preceding hour sum |
| cloud_cover | \% | float | Instant measure at `valid time` |
| cloud_cover_low | \% | float | Instant measure at `valid time` |
| cloud_cover_mid | \% | float | Instant measure at `valid time` |
| cloud_cover_high | \% | float | Instant measure at `valid time` |
| direct_radiation | W/m² | float | Mean of hour preceding the `valid_time` |
| direct_normal_irradiance | W/m² | float | Mean of hour preceding the `valid_time` |
| diffuse_radiation | W/m² | float | Mean of hour preceding the `valid_time` |
| sunshine_duration | s | float | Sum of hour preceding the `valid_time` |
| wind_speed_100m | km/h | float | Instant measure at `valid time` |
| wind_direction_100m | ° | float | Instant measure at `valid time` |
| wind_gusts_10m | km/h | float | Instant measure at `valid time` |
| weather_code | WMO code | int | Instant measure at `valid time` |
| snow_depth | m | float | Instant measure at `valid time` |
| vapour_pressure_deficit | kPa | float | Instant measure at `valid time` |
| soil_temperature_0_to_7cm | °C | float | Instant measure at `valid time` |
| soil_temperature_7_to_28cm | °C | float | Instant measure at `valid time` |
| soil_temperature_28_to_100cm | °C | float | Instant measure at `valid time` |
| soil_temperature_100_to_255cm | °C | float | Instant measure at `valid time` |
| soil_moisture_0_to_7cm | m³/m³ | float | Instant measure at `valid time` |
| soil_moisture_7_to_28cm | m³/m³ | float | Instant measure at `valid time` |
| soil_moisture_28_to_100cm | m³/m³ | float | Instant measure at `valid time` |
| soil_moisture_100_to_255cm | m³/m³ | float | Instant measure at `valid time` |

## Key objects

`Records`: values for several measures with a regularly-spaced time grid (hourly) for one given location on Earth (given by the lat and lon in WGS84 format). Records can be forecasts or observations, given by the `valid_time` and estimated at the `init_time` . `Records` can either be forecast or observations, but not both / mixed.

`Features`: transformed and aggregated records (within a `Records` and between `Records`).

## Installation

### From pypi

``` shell
pip install mlweather
```

### From source github using uv

``` bash
# SSH
uv add mlweather "git+ssh://git@github.com/kap-conseil/mlweather.git"
# HTTPS (with .netrc file configured)
uv add mlweather "git+https://github.com/kap-conseil/mlweather.git"
# For a specific branch
uv add mlweather "git+ssh://git@github.com/kap-conseil/mlweather.git@dev"
```

## Use

``` python
from datetime import datetime, timezone, timedelta
from polars import col
from mlweather.collection import Observations
from mlweather.aggregation import AggConfig, aggregate

# Collect weather data
obs = Observations.get_obs(
    (52.52, 13.41),  # Berlin (lat, lon)
    ["precipitation", "temperature_2m"],
    datetime(2023, 1, 1, tzinfo=timezone.utc),
    datetime(2023, 12, 31, tzinfo=timezone.utc),
)

# Configure aggregations
agg_config = [
    AggConfig(col("precipitation").sum(), timedelta(days=1)),
    AggConfig(col("temperature_2m").mean(), timedelta(days=7)),
]

# Generate ML features
features = aggregate(obs.values, agg_config)
```

## API Reference

### Core Classes

#### `Observations.get_obs()`

Retrieve weather observations from Open-Meteo API.

**Parameters:** - `location` (tuple): Coordinates as (latitude, longitude) - `variables` (list\[str\]): Weather variables to collect - `start_date` (datetime): Start date (UTC timezone) - `end_date` (datetime): End date (UTC timezone) - `api_key` (str, optional): Open-Meteo API key - `verbose` (bool): Enable detailed logging

**Returns:** `Observations` object with `.values` DataFrame

#### `AggConfig`

Configuration for temporal aggregations.

**Parameters:** - `operation` (polars.Expr): Aggregation operation - `period` (timedelta): Rolling window period

#### `aggregate()`

Apply temporal aggregations to weather data.

**Parameters:** - `obs_values` (DataFrame): Weather observations - `agg_config` (list\[AggConfig\]): Aggregation configurations

**Returns:** `DataFrame` with aggregated features

### Supported Variables (so far)

| Variable               | Description       | Unit |
|------------------------|-------------------|------|
| `precipitation`        | Precipitation sum | mm   |
| `temperature_2m`       | Temperature at 2m | °C   |
| `wind_speed_10m`       | Wind speed at 10m | m/s  |
| `relative_humidity_2m` | Relative humidity | \%   |
| `surface_pressure`     | Surface pressure  | hPa  |

## Configuration

Set your Open-Meteo API key:

``` bash
export OPENMETEO_API_KEY="your_api_key"
```

Or use a `.env` file:

```         
OPENMETEO_API_KEY=your_api_key_here
```

## Performance

-   Built on Polars for high-performance data processing
-   Efficient memory usage for large datasets
-   Optimized rolling window calculations