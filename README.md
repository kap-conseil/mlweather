# MLWeather: A package to collect and prepare ML-ready weather observations and forecasts

- Based on Open-meteo.com service (hourly data),
- Rely on Polars for fast data processing.

## Current version 
0.1.0

## Installation (with uv package manager)

From your private Github repository, you can install the package using the `uv` package manager:
```bash
uv add mlweather "ssh://git@github.com/kap-conseil/mlweather.git"
```

From a specific branch, you can specify the branch name in the URL:
```bash
uv add mlweather "ssh://git@github.com/kap-conseil/mlweather.git@dev"
```

With a PAT toke for automated deployments: 
```bash
uv add mlweather "https://<YOUR PAT TOKEN>@github.com/kap-conseil/mlweather.git"
```
