from unittest import result

import pytest
import os
from polars import DataFrame, Datetime, Float64, String
from dotenv import load_dotenv

from mlweather.collection.records import Records

load_dotenv()

# TESTS FOR THE RECORDS BASE CLASS #############################################


# Test the field for the init
def test_records_init_fields():
    lat_lon = (50.0, 8.0)
    elevation = 100.0
    units = {"temperature_2m": "°C", "precipitation": "mm"}
    record_table = DataFrame(
        {
            "dt_target": ["2023-10-01T00:00", "2023-10-01T01:00"],
            "temperature_2m": [15.0, 14.5],
            "precipitation": [0.0, 1.2],
        }
    )

    # Since Records is abstract, we create a simple subclass for testing
    class TestRecords(Records):
        def __repr__(self):
            return "TestRecords instance"

    records = TestRecords(lat_lon, elevation, units, record_table)
    assert records.lat_lon == lat_lon
    assert records.elevation == elevation
    assert records.units == units
    assert records.record_table.equals(record_table)


# Test if variable names (for weather measures) are correctly found and rejected if inexistant
def test_validate_var_names():
    valid_vars = ["temperature_2m", "precipitation"]
    invalid_vars = ["temperature_2m", "invalid_variable"]

    assert Records.validate_var_names(valid_vars) is None

    with pytest.raises(ValueError) as excinfo:
        Records.validate_var_names(invalid_vars)
    assert "The following variable names are not admissible" in str(excinfo.value)


# Test the base function to get data from Open-Meteo
# 1/ Without API key first, on observations
# for the equivalent of curl command
# https://archive-api.open-meteo.com/v1/era5?latitude=52.52&longitude=13.41&start_date=2021-01-01&end_date=2021-12-31&hourly=temperature_2m
def test_get_openmeteo():
    base_url = "https://archive-api.open-meteo.com/v1/era5"
    params = {
        "latitude": 52.52,
        "longitude": 13.41,
        "start_date": "2021-01-01",
        "end_date": "2021-01-02",
        "hourly": "temperature_2m,precipitation",
    }

    # Fetch
    result = Records.get_openmeteo(base_url, params, cache_enabled=False)

    # Check key existence and data length
    assert "hourly" in result
    assert "time" in result["hourly"]
    assert "temperature_2m" in result["hourly"]
    assert "precipitation" in result["hourly"]
    assert len(result["hourly"]["time"]) == 48  # 2 days of hourly data


# 2 / With API key on forecasts
# For the equivalent of the curl command
# https://customer-archive-api.open-meteo.com/v1/archive?latitude=52.52&longitude=13.41&start_date=2025-12-21&end_date=2026-01-04&hourly=temperature_2m&apikey=XXX
def test_get_openmeteo_with_api_key():
    base_url = "https://customer-archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": 52.52,
        "longitude": 13.41,
        "hourly": "temperature_2m,precipitation",
        "apikey": os.getenv("OPENMETEO_API_KEY"),  # Example API key
    }

    # Fetch
    result_from_sim_data = Records.get_openmeteo(base_url, params, cache_enabled=False)

    # Check key existence
    assert "hourly" in result_from_sim_data
    assert "time" in result_from_sim_data["hourly"]
    assert "temperature_2m" in result_from_sim_data["hourly"]
    assert "precipitation" in result_from_sim_data["hourly"]


def test_prepare_hourly_record_from_sim_data():
    # Simulated data
    resp_dict = {
        "hourly": {
            "time": [
                "2026-01-03T00:00",
                "2026-01-03T01:00",
                "2026-01-03T02:00",
                "2026-01-03T03:00",
                "2026-01-03T04:00",
                "2026-01-09T19:00",
                "2026-01-09T20:00",
                "2026-01-09T21:00",
                "2026-01-09T22:00",
                "2026-01-09T23:00",
            ],
            "temperature_2m": [7.1, 6.7, 7.3, 7.7, 7.6, 8.5, 8.6, 8.6, 8.8, 8.8],
            "precipitation": [0.1, 0.2, 0.0, 0.0, 0.0, 0.1, 0.1, 0.1, 0.1, 0.0],
        }
    }

    # Key function to test
    record_table = Records.prepare_hourly_records(resp_dict)

    # Check columns existence
    assert "valid_datetime" in record_table.columns
    assert "temperature_2m" in record_table.columns
    assert "precipitation" in record_table.columns
    # Check schema and types
    assert record_table.dtypes == [Datetime, Float64, Float64]


# Test the preparation of the hourly record table from the raw Open-Meteo response
def test_prepare_hourly_record_from_real_data():
    # Get content from Open-Meteo
    base_url = "https://customer-archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": 52.52,
        "longitude": 13.41,
        "hourly": "temperature_2m,precipitation",
        "apikey": os.getenv("OPENMETEO_API_KEY"),  # Example API key
    }

    # Fetch
    result_from_real_data = Records.get_openmeteo(base_url, params, cache_enabled=False)
    # Prepare record table
    record_table = Records.prepare_hourly_records(result_from_real_data)

    # Check columns existence
    assert "valid_datetime" in record_table.columns
    assert "temperature_2m" in record_table.columns
    assert "precipitation" in record_table.columns
