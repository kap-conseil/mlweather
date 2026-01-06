import pytest
import os
from polars import DataFrame
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
def test_are_var_names_valid():
    valid_vars = ["temperature_2m", "precipitation"]
    invalid_vars = ["temperature_2m", "invalid_variable"]

    assert Records.are_var_names_valid(valid_vars) is True

    with pytest.raises(ValueError) as excinfo:
        Records.are_var_names_valid(invalid_vars)
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
    result = Records.get_openmeteo(base_url, params, verbose=False)

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
    result = Records.get_openmeteo(base_url, params, verbose=False)

    # Check key existence
    assert "hourly" in result
    assert "time" in result["hourly"]
    assert "temperature_2m" in result["hourly"]
    assert "precipitation" in result["hourly"]


# Test the preparation of the hourly record table from the raw Open-Meteo response
def test_prepare_hourly_record_table():
    # Get content from Open-Meteo
    base_url = "https://customer-archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": 52.52,
        "longitude": 13.41,
        "hourly": "temperature_2m,precipitation",
        "apikey": os.getenv("OPENMETEO_API_KEY"),  # Example API key
    }

    # Fetch
    result = Records.get_openmeteo(base_url, params, verbose=False)
    # Prepare record table
    record_table = Records.prepare_hourly_records(result)

    # Check columns existence
    assert "valid_datetime" in record_table.columns
    assert "temperature_2m" in record_table.columns
    assert "precipitation" in record_table.columns
