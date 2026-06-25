# MLWeather

`mlweather` is a Python toolkit for collecting raw high precision weather data into machine-learning-ready features at blazing speed.

Tired of building pipleline to collect weather data, control them, aggreagate them and prepare them for machine learning ? This package is for you. It provides a simple and efficient way to collect weather data (observations and forecasts) from the [Open-Meteo](https://open-meteo.com/) API, aggregate them over any period using any aggregation function, and prepare them for machine learning.

Fast-track feature preparation so you can focus on energy, consumption, crop growth, or any other ML project that requires highly precise localization and up-to-hourly data across a wide range of variables.

[![Version](https://img.shields.io/badge/dynamic/toml?url=https%3A%2F%2Fraw.githubusercontent.com%2Fkap-conseil%2Fmlweather%2Fmain%2Fpyproject.toml&query=%24.project.version&label=version)](https://github.com/kap-conseil/mlweather/blob/main/pyproject.toml)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://github.com/kap-conseil/mlweather/actions/workflows/tests.yml/badge.svg)](https://github.com/kap-conseil/mlweather/actions/workflows/tests.yml)
[![Ruff](https://github.com/kap-conseil/mlweather/actions/workflows/lint.yml/badge.svg)](https://github.com/kap-conseil/mlweather/actions/workflows/lint.yml)
[![Coverage Artifact](https://img.shields.io/github/actions/workflow/status/kap-conseil/mlweather/tests.yml?label=coverage%20artifact)](https://github.com/kap-conseil/mlweather/actions/workflows/tests.yml)  

## Installation

### The classic way

``` shell
pip install mlweather
```
### Using UV

``` shell
# Last release from PyPI
uv add mlweather
# Main version for the last tagged release
uv add mlweather "git+https://github.com/kap-conseil/mlweather.git"
# For a specific branch (e.g. dev)
uv add mlweather "git+ssh://git@github.com/kap-conseil/mlweather.git@dev"
```

## Target uses

This package aims to collect and prepare complex meteorological features ready for machine learning:

- over large periods of observations and forecasts at blazing speed and with ease of use,

- with high precision data and a large variety of meteorological variables (observations and forecast) based on the [Open-Meteo](https://open-meteo.com/) API (hourly data),

- with custom transformations of the variables to prepare ML features based on aggregations over any period (e.g. daily, weekly, monthly) using any aggregation function (e.g. sum, mean, min, max,...).

The packages main roles are:

-   **Collecting weather records** for past meteorological and forecasts for any location on Earth based on hourly measures for a large set of meteorological variables with profond history. The data is sourced from the high quality [Open-Meteo](https://open-meteo.com/). Forecast are both current forecast (to predict) and past forecast to train you model. Then you can train you models on the data of the same nature: you train your model on forecast data because you will predict on forecast data. It gives access to both the non-commercial access and commercial licenses of Open-Meteo. Please go to [Open-Meteo Pricing](https://open-meteo.com/en/pricing) for the conditions of use.

-   **Aggregate forecasts, observations** or a mixture (e.g. X days of observations followed by Y days of forecasts) using **any function** to have aggregation (e.g. sum, mean, min,...for a given period) over a chosen period for every hour. It allows to combine **observations and forecasts** to have aggregation for periods for observations (e.g. X days) followed by forecasts (e.g. Y days). This approach provides features that exploit the last forecasts and assemble them to prior observations. It provides the average of temperatures mixing the available temperature forecasts and with the preceding observed temperature. You can have aggregations of observations and forecasts for custom meteorological ML features.

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

### `Observations` : collect historical weather observations for a given location and period.
      
`Observations` are sets of weather records collected for a given location (latitude, longitude) over a given period (start date, end date). They contain historical weather observations (not forecasts) sourced from the Open-Meteo API, at an hourly frequency.
By default the non-commercial access of Open-Meteo is used. Please refer to [Open-Meteo pricing](https://open-meteo.com/en/pricing) for the conditions of use. Just provide a key if you have a commercial license.

```python
from datetime import datetime, timezone, timedelta
from mlweather.collection.observations import Observations

observations = Observations.collect(
    lat_lon=(52.52, 13.41),                               # Berlin (lat, lon)
    variables_names=["precipitation", "temperature_2m"],         # variables to collect 
    start=datetime(2024, 5, 1, tzinfo=timezone.utc),  # start date for collection
    end=datetime(2024, 7, 31, tzinfo=timezone.utc),  # end date for collection
    # api_key="your_open_meteo_api_key"  # optional: your Open-Meteo API key for commercial access
)
```

The resulting `observations` object contains metadata (elevation, units) and a Polars DataFrame (`record_table`) with the collected weather records.
Weather records include `init_datetime` (null because it is not a forecast and evaluated at the `valid_datetime`), `valid_datetime` (time of the observation), and the requested variables.
The view of the ouput `observations` object is as follows:
```
Observations of location (47.83831, -4.17218) with 2,424 hourly entries.
elevation: 18.0 m
units: {'temperature_2m': '°C', 'precipitation': 'mm'}
record_table:
shape: (2_424, 4)
┌───────────────────┬─────────────────────────┬────────────────┬───────────────┐
│ init_datetime     ┆ valid_datetime          ┆ temperature_2m ┆ precipitation │
│ ---               ┆ ---                     ┆ ---            ┆ ---           │
│ datetime[μs, UTC] ┆ datetime[μs, UTC]       ┆ f64            ┆ f64           │
╞═══════════════════╪═════════════════════════╪════════════════╪═══════════════╡
│ null              ┆ 2025-10-01 00:00:00 UTC ┆ 10.8           ┆ 0.0           │
│ null              ┆ 2025-10-01 01:00:00 UTC ┆ 10.6           ┆ 0.0           │
│ null              ┆ 2025-10-01 02:00:00 UTC ┆ 10.6           ┆ 0.0           │
│ null              ┆ 2025-10-01 03:00:00 UTC ┆ 10.7           ┆ 0.0           │
│ null              ┆ 2025-10-01 04:00:00 UTC ┆ 10.5           ┆ 0.0           │
│ …                 ┆ …                       ┆ …              ┆ …             │
│ null              ┆ 2026-01-09 19:00:00 UTC ┆ 5.9            ┆ 0.0           │
│ null              ┆ 2026-01-09 20:00:00 UTC ┆ 7.1            ┆ 0.0           │
│ null              ┆ 2026-01-09 21:00:00 UTC ┆ 6.9            ┆ 0.0           │
│ null              ┆ 2026-01-09 22:00:00 UTC ┆ 7.0            ┆ 0.1           │
│ null              ┆ 2026-01-09 23:00:00 UTC ┆ 7.5            ┆ 0.0           │
└───────────────────┴─────────────────────────┴────────────────┴───────────────┘
```

### Forecasts : collect weather forecasts for a given location and period (past and current forecast).
      

```python
from mlweather.collection.forecasts import Forecasts

forecasts = Forecasts.collect(
    lat_lon=(52.52, 13.41),                               # Berlin (lat, lon)
    forecast_horizon_days_max= 4,                          # max forecast horizon in days. Collect forecast from 0 days ahead (current day to 4 days ahead)
    variables_names=["precipitation", "temperature_2m"],         # variables to collect 
    start=datetime(2024, 5, 1, tzinfo=timezone.utc),  # start date for collection
    end=datetime(2024, 7, 31, tzinfo=timezone.utc),  # end date for collection
    # api_key="your_open_meteo_api_key"  # optional: your Open-Meteo API key for commercial access
)
```

The resulting `Forecasts` object contains metadata (elevation, units) and a Polars DataFrame (`record_table`) with the collected weather records.
By default the non-commercial access of Open-Meteo is used. Please refer to [Open-Meteo pricing](https://open-meteo.com/en/pricing) for the conditions of use. Just provide a key if you have a commercial license.
Weather records include `init_datetime` (timestamp of the forecasting run), `valid_datetime` (time of the observation), and the requested variables.

Past forecast (particularly uselfull to train models) are collected for different forecast horizons: from 0 days ahead forecast (intraday forecast) to 7 days ahead forecast (7 days forecast) in [Open-Meteo](https://open-meteo.com/) API.

The view of the ouput `forecasts` object is as follows:

```
Forecasts of location (52.52, 13.419998) with 11,040 hourly entries.
for forecast horizons of days ranging from 0 to 4.
elevation: 38.0 m
units: {'precipitation': 'mm', 'precipitation_previous_day1': 'mm', 'precipitation_previous_day2': 'mm', 'precipitation_previous_day3': 'mm', 'precipitation_previous_day4': 'mm', 'temperature_2m': '°C', 'temperature_2m_previous_day1': '°C', 'temperature_2m_previous_day2': '°C', 'temperature_2m_previous_day3': '°C', 'temperature_2m_previous_day4': '°C'}
record_table:
shape: (11_040, 4)
┌─────────────────────────┬─────────────────────────┬───────────────┬────────────────┐
│ init_datetime           ┆ valid_datetime          ┆ precipitation ┆ temperature_2m │
│ ---                     ┆ ---                     ┆ ---           ┆ ---            │
│ datetime[μs, UTC]       ┆ datetime[μs, UTC]       ┆ f64           ┆ f64            │
╞═════════════════════════╪═════════════════════════╪═══════════════╪════════════════╡
│ 2024-04-27 00:00:00 UTC ┆ 2024-05-01 00:00:00 UTC ┆ 0.0           ┆ 16.5           │
│ 2024-04-27 00:00:00 UTC ┆ 2024-05-01 01:00:00 UTC ┆ 0.0           ┆ 15.6           │
│ 2024-04-27 00:00:00 UTC ┆ 2024-05-01 02:00:00 UTC ┆ 0.0           ┆ 14.8           │
│ 2024-04-27 00:00:00 UTC ┆ 2024-05-01 03:00:00 UTC ┆ 0.0           ┆ 14.4           │
│ 2024-04-27 00:00:00 UTC ┆ 2024-05-01 04:00:00 UTC ┆ 0.0           ┆ 14.5           │
│ …                       ┆ …                       ┆ …             ┆ …              │
│ 2024-07-31 00:00:00 UTC ┆ 2024-07-31 19:00:00 UTC ┆ 0.0           ┆ 25.2           │
│ 2024-07-31 00:00:00 UTC ┆ 2024-07-31 20:00:00 UTC ┆ 0.0           ┆ 23.3           │
│ 2024-07-31 00:00:00 UTC ┆ 2024-07-31 21:00:00 UTC ┆ 0.0           ┆ 22.0           │
│ 2024-07-31 00:00:00 UTC ┆ 2024-07-31 22:00:00 UTC ┆ 0.0           ┆ 20.9           │
│ 2024-07-31 00:00:00 UTC ┆ 2024-07-31 23:00:00 UTC ┆ 0.0           ┆ 19.7           │
└─────────────────────────┴─────────────────────────┴───────────────┴────────────────┘
```

## Aggregation: recipe for ML features

Once observations and/or forecasts are collected, `mlweather` provides tools to prepare ML-ready features based on temporal aggregations over any period using any aggregation function.

Each feature is defined by an `Aggregation` that defines the transformation for hourly data, composition between observations and forecasts, the aggregation operation and the period over which to aggregate.

For instance to compute : 
- the sum of the observed precipitation over the last 7 days *on observations only*, you can define the first (`aggregation_1`),
- the sum of the forecasted precipitation over the next 3 days *on forecasts only*, you can define the second (`aggregation_2`),
- and then combine both aggregations to have the total precipitation over the last 7 days of observations and the next 3 days of forecasts. This case is usefull when you need forecast predictors combining the past observations and forecast at the same time. For example, you need to predict the growth of mushrooms (they love water) in two days from now based on 7 days of precipitations. You can sum the last 5 days of observed precipitations and the next 2 days of forecasted precipitations to have the total precipitations over the 7 days period. `mlweather` allows to define such complex features with ease for current values and for the past (to train properly your model with valid forecast data and not historical data only).
  

```python
from polars import col
from mlweather.aggregation import Aggregation

aggregation_1 = Aggregation(
            col("precipitation").sum(),
            observation_period=timedelta(days=7),
        )

aggregation_2 = Aggregation(
            col("precipitation").sum(),
            observation_period=timedelta(days=3),
        )

aggregation_3 = Aggregation(
            col("precipitation").sum(),
            observation_period=timedelta(days=7),
            forecast_period=timedelta(days=2),
        )
```

Any polars expressions (`Expr`) can be used to define the aggregation operation (e.g. `sum`, `mean`, `min`, `max` or your custom `Expr`...) as long as it return a single value.

### Feature generation

Once the recipes for the features are defined as `Aggregation` objects, the `FeatureGenerator` can be used to compute the features for given time points (here `features_datetimes`) based on the collected observations and forecasts.

The aggregations are safely produced in the sense that: 
- The completeness of the hourly observations and forecast is verified: no records can be missing for the requested periods,
- The aggregations are only returned when all the needed records are exhaustively present.

```python
# Instantiate the feature generator with the defined aggregations
from mlweather.aggregation import FeatureGenerator

feat_gen = FeatureGenerator(
    [
        aggregation_1,
        aggregation_2,
        aggregation_3,
    ],
)

# For which time points should we computer the features ?
features_datetimes = [
    datetime(2024, 6, 1, tzinfo=timezone.utc),
    datetime(2024, 6, 2, tzinfo=timezone.utc),
    datetime(2024, 6, 3, tzinfo=timezone.utc),
    datetime(2024, 6, 4, tzinfo=timezone.utc),
    datetime(2024, 6, 5, tzinfo=timezone.utc),
    datetime(2024, 6, 6, tzinfo=timezone.utc),
    datetime(2024, 6, 7, tzinfo=timezone.utc),
    datetime(2024, 6, 8, tzinfo=timezone.utc),
    datetime(2024, 6, 9, tzinfo=timezone.utc),
]

ml_features = feat_gen.generate_features(
    features_datetimes, observations=observations, forecasts=forecasts
)
```

It produces a Polars DataFrame with the computed features for the requested time points ready to plug into you ML pipelines:

```
shape: (9, 4)
 ┌────────────────────────┬────────────────────────┬────────────────────────┬───────────────────────┐
 │ valid_datetime         ┆ precipitation_sum_obs_ ┆ precipitation_sum_obs_ ┆ precipitation_sum_obs │
 │ ---                    ┆ 7d                     ┆ 3d                     ┆ _7d_fore_…            │
 │ datetime[μs, UTC]      ┆ ---                    ┆ ---                    ┆ ---                   │
 │                        ┆ f64                    ┆ f64                    ┆ f64                   │
 ╞════════════════════════╪════════════════════════╪════════════════════════╪═══════════════════════╡
 │ 2024-06-01 00:00:00    ┆ 34.4                   ┆ 10.0                   ┆ 42.0                  │
 │ UTC                    ┆                        ┆                        ┆                       │
 │ 2024-06-02 00:00:00    ┆ 30.6                   ┆ 9.1                    ┆ 45.8                  │
 │ UTC                    ┆                        ┆                        ┆                       │
 │ 2024-06-03 00:00:00    ┆ 39.6                   ┆ 9.7                    ┆ 37.7                  │
 │ UTC                    ┆                        ┆                        ┆                       │
 │ 2024-06-04 00:00:00    ┆ 28.3                   ┆ 9.8                    ┆ 30.6                  │
 │ UTC                    ┆                        ┆                        ┆                       │
 │ 2024-06-05 00:00:00    ┆ 20.1                   ┆ 9.7                    ┆ 39.6                  │
 │ UTC                    ┆                        ┆                        ┆                       │
 │ 2024-06-06 00:00:00    ┆ 21.1                   ┆ 2.8                    ┆ 29.4                  │
 │ UTC                    ┆                        ┆                        ┆                       │
 │ 2024-06-07 00:00:00    ┆ 12.5                   ┆ 2.6                    ┆ 22.8                  │
 │ UTC                    ┆                        ┆                        ┆                       │
 │ 2024-06-08 00:00:00    ┆ 12.6                   ┆ 2.5                    ┆ 21.6                  │
 │ UTC                    ┆                        ┆                        ┆                       │
 │ 2024-06-09 00:00:00    ┆ 12.2                   ┆ 0.2                    ┆ 12.5                  │
 │ UTC                    ┆                        ┆                        ┆                       │
 └────────────────────────┴────────────────────────┴────────────────────────┴───────────────────────┘
```

## Attribution 

This package is developed and maintained by [KAP IA](https://www.kap.bzh).
## Acknowledgements

This project is supported by:   

<a href="https://www.bretagne.bzh"><img src="https://upload.wikimedia.org/wikipedia/fr/thumb/8/83/R%C3%A9gion-bretagne-logo.svg/250px-R%C3%A9gion-bretagne-logo.svg.png?_=20190416150425" alt="Région Bretagne" width="150" /></a>  

in the context of *Plateforme pour l'IA du vivant*.