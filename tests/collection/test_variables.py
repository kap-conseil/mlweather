from mlweather.collection.variables import Variable


class TestVariable:
    """Tests for the Variable class."""

    def test_variable_init(self):
        """Test Variable initialization with valid parameters."""
        var = Variable("test_var", "mm", float, "Instant")
        assert var.name == "test_var"
        assert var.unit == "mm"
        assert isinstance(var.value_type, type)
        assert var.measurement == "Instant"

    def test_variable_unit_attribute(self):
        """Test that unit attribute is correctly set."""
        var = Variable("temperature", "°C", float, "Instant")
        assert var.unit == "°C"

    def test_variable_unit_different_values(self):
        """Test unit attribute with various unit types."""
        test_cases = [
            ("mm", "mm"),
            ("°C", "°C"),
            ("km/h", "km/h"),
            ("%", "%"),
            ("W/m²", "W/m²"),
            ("hPa", "hPa"),
            ("m³/m³", "m³/m³"),
            ("kPa", "kPa"),
            ("WMO code", "WMO code"),
        ]
        for unit, expected in test_cases:
            var = Variable("test", unit, float, "Instant")
            assert var.unit == expected
