# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

# The 'fixtures/' folder contains small, purpose-built example files for each
# supported instrument format. They are kept minimal (a few metadata lines and
# ~5 data rows) but cover all relevant cases: string, numeric, list, and empty
# metadata fields, as well as the special characters (² and °C) that some
# instruments encode in a non-standard way.
# These files serve as stable, known inputs so that tests can verify the parser
# output against expected values without depending on real measurement data.

from pathlib import Path

import pytest

from firescipy.instruments import (
    detect_file_type,
    read_instrument_file,
    read_deatak_mcc_file,
    read_netzsch_sta_file,
    read_netzsch_cone_file,
    SUPPORTED_TYPES,
)

FIXTURES = Path(__file__).parent / "fixtures"


# ---------------------------------------------------------------------------
# Supported types registry
# ---------------------------------------------------------------------------

def test_supported_types_contains_all_instruments():
    assert "Netzsch STA" in SUPPORTED_TYPES
    assert "Deatak MCC" in SUPPORTED_TYPES
    assert "Netzsch Cone" in SUPPORTED_TYPES


# ---------------------------------------------------------------------------
# Auto-detection
# ---------------------------------------------------------------------------

def test_detect_netzsch_sta():
    assert detect_file_type(FIXTURES / "Netzsch_STA.csv") == "Netzsch STA"


def test_detect_deatak_mcc():
    assert detect_file_type(FIXTURES / "Deatak_MCC.txt") == "Deatak MCC"


def test_detect_netzsch_cone():
    assert detect_file_type(FIXTURES / "Netzsch_Cone.csv") == "Netzsch Cone"


# ---------------------------------------------------------------------------
# Generic reader — auto-detection path
# ---------------------------------------------------------------------------

def test_read_instrument_file_returns_correct_type():
    for fname, expected_type in [
        ("Netzsch_STA.csv", "Netzsch STA"),
        ("Deatak_MCC.txt", "Deatak MCC"),
        ("Netzsch_Cone.csv", "Netzsch Cone"),
    ]:
        file_type, _, _ = read_instrument_file(FIXTURES / fname)
        assert file_type == expected_type


def test_read_instrument_file_manual_override():
    # Explicitly specifying the type should work even when detection would also succeed.
    file_type, meta, df = read_instrument_file(
        FIXTURES / "Deatak_MCC.txt", file_type="Deatak MCC"
    )
    assert file_type == "Deatak MCC"
    assert df is not None


def test_read_instrument_file_unknown_type_raises():
    with pytest.raises(ValueError, match="Unknown file_type"):
        read_instrument_file(FIXTURES / "Deatak_MCC.txt", file_type="Unknown Device")


# ---------------------------------------------------------------------------
# Deatak MCC parser
# ---------------------------------------------------------------------------

class TestDeatakMCC:
    def setup_method(self):
        self.meta, self.df = read_deatak_mcc_file(FIXTURES / "Deatak_MCC.txt")

    def test_dataframe_shape(self):
        assert self.df.shape == (5, 5)

    def test_dataframe_columns(self):
        assert "Time (s)" in self.df.columns
        assert "Temperature (C)" in self.df.columns
        assert "HRR (W/g)" in self.df.columns

    def test_numeric_metadata(self):
        # Scalar float fields
        assert self.meta["Sample Mass (mg)"] == pytest.approx(2.0)
        assert self.meta["Heating Rate (C/s)"] == pytest.approx(1.0)
        assert self.meta["Time Shift (s)"] == pytest.approx(14.0)

    def test_list_metadata(self):
        # T Correction Coefficients must be a list of floats, not strings.
        coeffs = self.meta["T Correction Coefficients"]
        assert isinstance(coeffs, list)
        assert all(isinstance(v, float) for v in coeffs)
        assert coeffs == pytest.approx([0.30873873, 0.0136155, 0.0])

    def test_none_metadata(self):
        assert self.meta["Pre-Test Comments"] is None

    def test_string_metadata(self):
        assert self.meta["Sample ID"] == "Wood"

    def test_used_encoding_present(self):
        assert "USED_ENCODING" in self.meta


# ---------------------------------------------------------------------------
# Netzsch STA parser
# ---------------------------------------------------------------------------

class TestNetzschSTA:
    def setup_method(self):
        self.meta, self.df = read_netzsch_sta_file(FIXTURES / "Netzsch_STA.csv")

    def test_dataframe_shape(self):
        assert self.df.shape == (5, 5)

    def test_numeric_metadata(self):
        assert self.meta["SAMPLE MASS /mg"] == pytest.approx(5.28)
        assert self.meta["REFERENCE MASS /mg"] == pytest.approx(0.0)

    def test_none_metadata(self):
        assert self.meta["REMARK"] is None

    def test_decimal_and_separator_detected(self):
        assert self.meta["DECIMAL"] == "COMMA"
        assert self.meta["SEPARATOR"] == "SEMICOLON"

    def test_data_uses_correct_decimal(self):
        # Values were stored with comma as decimal — must be parsed as floats.
        assert self.df.dtypes["Temp./°C"] == "float64"

    def test_used_encoding_present(self):
        assert "USED_ENCODING" in self.meta


# ---------------------------------------------------------------------------
# Netzsch Cone parser
# ---------------------------------------------------------------------------

class TestNetzschCone:
    def setup_method(self):
        self.meta, self.df = read_netzsch_cone_file(FIXTURES / "Netzsch_Cone.csv")

    def test_dataframe_shape(self):
        assert self.df.shape == (6, 3)

    def test_dataframe_columns(self):
        assert list(self.df.columns) == ["time (s)", "HRR/a", "Mass"]

    def test_numeric_metadata_with_special_chars(self):
        # These keys contain ² and °C — tests that special characters are
        # correctly recovered from the file encoding.
        assert self.meta["Heat flux (kW/m²)"] == pytest.approx(75.0)
        assert self.meta["Ambient temperature (°C)"] == pytest.approx(25.6)
        assert self.meta["Initial mass (g)"] == pytest.approx(74.22)
        assert self.meta["Time to ignition (s)"] == pytest.approx(13.0)

    def test_none_metadata(self):
        assert self.meta["Non-scrubbed?"] is None

    def test_string_metadata(self):
        assert self.meta["Standard used"] == "ISO 5660-1"

    def test_encoding_is_latin1(self):
        assert self.meta["USED_ENCODING"] == "latin-1"
