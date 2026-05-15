# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

from io import StringIO

import pandas as pd

from .base import InstrumentFile
from .helpers import split_line, strip_empty_edges, try_convert_to_float


def read_netzsch_cone_file(file_path):
    """
    Convenience wrapper around NetzschConeParser.
    """
    parser = NetzschConeParser(file_path=file_path)
    return parser.parse()


class NetzschConeParser:
    """
    Parser for NETZSCH Cone Calorimeter CSV export files.

    Expected structure
    ------------------
    - Row 0: "General information" label in column 0, measurement column names
             starting from column 2
    - Row 1: measurement units starting from column 2 (columns 0-1 are empty)
    - Rows 2+: metadata key in column 0, metadata value in column 1,
               measurement values starting from column 2

    Unlike the NetzschSTA and Deatak parsers, metadata and measurement data
    are stored side by side in each row rather than in sequential blocks.

    Example
    -------
    General information;;time (s);O2 (%);HRR/a;...
    ;;;;;;kW/m²;...
    Test;;0;20,952;0,0;...
    Standard used;ISO 5660-1;1;20,954;0,0;...
    Date of test;27.03.2025;2;20,953;0,0;...
    ...
    """

    SEPARATOR = ";"
    DECIMAL = ","

    # latin-1 is tried first because the Cone export encodes special characters
    # (² and °) using byte sequences that are best handled at the latin-1 level.
    _ENCODINGS = ["latin-1", "utf-8", "cp1252", "utf-16", "utf-16-le", "utf-16-be"]

    # The Cone export stores U+FFFD as a placeholder for ² and °. When decoded
    # as latin-1, the three UTF-8 bytes \xef\xbf\xbd of U+FFFD appear as the
    # three-character mojibake sequence ï¿½. The replacements below undo this:
    # \xef\xbf\xbdC (ï¿½C) must be handled before \xef\xbf\xbd (ï¿½) alone so
    # that °C and ² are recovered correctly.
    _REPLACEMENTS = {
        "\x9b": "°",
        "\xef\xbf\xbdC": "°C",     # ï¿½C → °C  (degree-Celsius, e.g. in column headers)
        "\xef\xbf\xbd": "²",        # ï¿½  → ²   (superscript-2, e.g. kW/m²)
    }

    # Metadata fields whose values should be converted to numbers.
    # All other fields are kept as strings.
    NUMERIC_METADATA_KEYS = {
        "Heat flux (kW/m²)",
        "Nominal duct flow rate (l/s)",
        "Sampling interval (s)",
        "Separation (mm)",
        "E (MJ/kg)",
        "Initial mass (g)",
        "Thickness (mm)",
        "Surface area (cm²)",
        "C-factor (SI units)",
        "OD correction factor",
        "Duct diameter (m)",
        "O2 delay time (s)",
        "CO2 delay time (s)",
        "CO delay time (s)",
        "Test start time (s)",
        "Time to ignition (s)",
        "Time to flameout (s)",
        "User EOT time (s)",
        "Ambient temperature (°C)",
        "Barometric pressure (Pa)",
        "Relative humidity (%)",
    }

    def __init__(self, file_path, instrument_file=None):
        """
        Parameters
        ----------
        file_path : str or Path
            Path to the NETZSCH Cone export file.
        instrument_file : InstrumentFile, optional
            Pre-loaded InstrumentFile instance. If None, one is created.
        """
        self.file_path = file_path
        # Use an existing InstrumentFile if provided, otherwise create one with
        # the Cone-specific encoding list and character replacements.
        self.file = instrument_file or InstrumentFile(
            file_path,
            encodings=self._ENCODINGS,
            replacements=self._REPLACEMENTS,
        ).read()

        self.meta = dict()      # will hold all metadata key-value pairs
        self.units = dict()     # will hold the unit string for each measurement column
        self.data_df = None     # will hold the measurement table as a DataFrame

        self._column_names = None   # set during header parsing

    def parse(self):
        """
        Parse metadata, units, and measurement table.

        Returns
        -------
        meta : dict
            Parsed metadata.
        data_df : pandas.DataFrame
            Parsed measurement table.
        """
        self._parse_header_and_units()
        self._parse_data_rows()
        self._postprocess_metadata()

        # Record which encoding was used so the caller can inspect it.
        self.meta["USED_ENCODING"] = self.file.used_encoding
        return self.meta, self.data_df

    def _parse_header_and_units(self):
        """
        Read column names from row 0 and units from row 1.
        """
        header_cells = split_line(self.file.lines[0], sep=self.SEPARATOR)
        unit_cells   = split_line(self.file.lines[1], sep=self.SEPARATOR)

        # The first two columns of row 0 are "General information" and an empty
        # label for the metadata value column — skip them to get measurement names.
        col_names = strip_empty_edges(header_cells[2:])
        self._column_names = col_names

        # Pair each column name with its unit string (may be empty for some columns).
        for name, unit in zip(col_names, unit_cells[2:]):
            self.units[name] = unit

    def _parse_data_rows(self):
        """
        Parse metadata from columns 0-1 and measurement values from columns 2+
        for each row starting at row 2.
        """
        data_rows = []

        for raw_line in self.file.lines[2:]:
            cells = split_line(raw_line, sep=self.SEPARATOR)

            if not cells:
                continue

            # Columns 0 and 1 contain metadata (key and value).
            # Later rows may have empty metadata columns — those are skipped.
            key   = cells[0] if len(cells) > 0 else ""
            value = cells[1] if len(cells) > 1 else ""
            if key:
                # Store None for fields that have a key but no value.
                self.meta[key] = value or None

            # Columns 2 onwards contain the measurement data for this time step.
            if len(cells) > 2:
                n = len(self._column_names)
                # Convert each cell to float using the locale decimal separator.
                row = [
                    try_convert_to_float(c, self.DECIMAL)
                    for c in cells[2:2 + n]
                ]
                data_rows.append(row)

        self.data_df = pd.DataFrame(data_rows, columns=self._column_names)
        # Drop columns that are entirely empty (unused channels in the export).
        self.data_df = self.data_df.dropna(axis=1, how="all")

    def _postprocess_metadata(self):
        """
        Convert selected metadata values to numeric types where appropriate.
        """
        # Only process the keys listed in NUMERIC_METADATA_KEYS.
        for key in self.NUMERIC_METADATA_KEYS:
            if key not in self.meta:
                continue
            # try_convert_to_float handles both single strings and lists.
            self.meta[key] = try_convert_to_float(self.meta[key], self.DECIMAL)
