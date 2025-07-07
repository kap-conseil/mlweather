# MLWeather

**Version: 0.1.5**

A high-performance Python library for collecting and preparing ML-ready weather observations and forecasts.

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)

## Features

- **Weather Data Collection**: Retrieve historical weather observations from Open-Meteo API
- **Temporal Aggregations**: Perform rolling window aggregations (e.g. daily, weekly, monthly)
- **Data Validation**: Automatic validation of data completeness and regularity
- **High Performance**: Built on Polars for fast data processing
- **ML-Ready Output**: Optimized data structures for machine learning workflows

## Installation

### From source with uv0.1

```bash
uv add mlweather "git+ssh://git@github.com/kap-conseil/mlweather.git"
```

For specific branch:
```bash
uv add mlweather "git+ssh://git@github.com/kap-conseil/mlweather.git@dev"
```

With PAT token for CI/CD:
```bash
uv add mlweather "https://<YOUR_PAT_TOKEN>@github.com/kap-conseil/mlweather.git"
```

## Quick Start

```python
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

**Parameters:**
- `location` (tuple): Coordinates as (latitude, longitude)
- `variables` (list[str]): Weather variables to collect
- `start_date` (datetime): Start date (UTC timezone)
- `end_date` (datetime): End date (UTC timezone)
- `api_key` (str, optional): Open-Meteo API key
- `verbose` (bool): Enable detailed logging

**Returns:** `Observations` object with `.values` DataFrame

#### `AggConfig`
Configuration for temporal aggregations.

**Parameters:**
- `operation` (polars.Expr): Aggregation operation
- `period` (timedelta): Rolling window period

#### `aggregate()`
Apply temporal aggregations to weather data.

**Parameters:**
- `obs_values` (DataFrame): Weather observations
- `agg_config` (list[AggConfig]): Aggregation configurations

**Returns:** `DataFrame` with aggregated features

### Supported Variables (so far)

| Variable | Description | Unit |
|----------|-------------|------|
| `precipitation` | Precipitation sum | mm |
| `temperature_2m` | Temperature at 2m | °C |
| `wind_speed_10m` | Wind speed at 10m | m/s |
| `relative_humidity_2m` | Relative humidity | % |
| `surface_pressure` | Surface pressure | hPa |

## Configuration

Set your Open-Meteo API key:

```bash
export OPENMETEO_API_KEY="your_api_key"
```

Or use a `.env` file:
```
OPENMETEO_API_KEY=your_api_key_here
```

## Performance


- Built on Polars for high-performance data processing
- Efficient memory usage for large datasets
- Optimized rolling window calculations
