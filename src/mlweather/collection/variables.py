class Variable:
    """
    Data class representing a weather variable with associated metadata.

    Attributes:
        name (str): The name of the measurement.
        unit (str): The unit in which the measurement is expressed.
        value_type (type): The expected data type of the measurement value.
        measurement (str): The description of the measurement process.
    """

    name: str
    unit: str
    value_type: type
    measurement: str

    def __init__(self, name: str, unit: str, value_type: type, measurement: str):
        self.name = name
        self.unit = unit
        self.value_type = value_type
        self.measurement = measurement

    @classmethod
    def init_from_name(cls, variable_name: str) -> "Variable":
        """
        Retrieve the meteorological Variable given its name.

        Args:
            variable_name (str): The name of the meteorological variable to retrieve.
        Returns:
            Variable: A new Variable instance copied from the admissible definition.
        """
        for var in ADMISSIBLE_VARIABLES:
            if var.name == variable_name:
                return cls(var.name, var.unit, var.value_type, var.measurement)
        raise ValueError(
            f"Variable name '{variable_name}' not found in admissible meteorological Variables."
        )


# Definition of all the admissible measures
ADMISSIBLE_VARIABLES = set(
    [
        Variable("precipitation", "mm", float, "Preceding hour sum"),
        Variable("temperature_2m", "°C", float, "Instant"),
        Variable("wind_speed_10m", "km/h", float, "Instant"),
        Variable("wind_direction_10m", "°", float, "Instant"),
        Variable("relative_humidity_2m", "%", float, "Instant"),
        Variable("shortwave_radiation", "W/m²", float, "Preceding hour mean"),
        Variable("global_tilted_irradiance", "W/m²", float, "Preceding hour mean"),
        Variable("et0_fao_evapotranspiration", "mm", float, "Preceding hour sum"),
        Variable("dew_point_2m", "°C", float, "Instant"),
        Variable("apparent_temperature", "°C", float, "Instant"),
        Variable("pressure_msl", "hPa", float, "Instant"),
        Variable("surface_pressure", "hPa", float, "Instant"),
        Variable("rain", "mm", float, "Preceding hour sum"),
        Variable("snowfall", "cm", float, "Preceding hour sum"),
        Variable("cloud_cover", "%", float, "Instant"),
        Variable("cloud_cover_low", "%", float, "Instant"),
        Variable("cloud_cover_mid", "%", float, "Instant"),
        Variable("cloud_cover_high", "%", float, "Instant"),
        Variable("direct_radiation", "W/m²", float, "Preceding hour mean"),
        Variable("direct_normal_irradiance", "W/m²", float, "Preceding hour mean"),
        Variable("diffuse_radiation", "W/m²", float, "Preceding hour mean"),
        Variable("sunshine_duration", "s", float, "Preceding hour sum"),
        Variable("wind_speed_100m", "km/h", float, "Instant"),
        Variable("wind_direction_100m", "°", float, "Instant"),
        Variable("wind_gusts_10m", "km/h", float, "Instant"),
        Variable("weather_code", "WMO code", int, "Instant"),
        Variable("snow_depth", "m", float, "Instant"),
        Variable("vapour_pressure_deficit", "kPa", float, "Instant"),
        Variable("soil_temperature_0_to_7cm", "°C", float, "Instant"),
        Variable("soil_temperature_7_to_28cm", "°C", float, "Instant"),
        Variable("soil_temperature_28_to_100cm", "°C", float, "Instant"),
        Variable("soil_temperature_100_to_255cm", "°C", float, "Instant"),
        Variable("soil_moisture_0_to_7cm", "m³/m³", float, "Instant"),
        Variable("soil_moisture_7_to_28cm", "m³/m³", float, "Instant"),
        Variable("soil_moisture_28_to_100cm", "m³/m³", float, "Instant"),
        Variable("soil_moisture_100_to_255cm", "m³/m³", float, "Instant"),
    ]
)
