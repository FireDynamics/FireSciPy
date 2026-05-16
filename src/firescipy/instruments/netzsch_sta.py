# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

from io import StringIO

import pandas as pd

from .base import InstrumentFile
from .helpers import split_line, strip_empty_edges, row_has_numeric_content, try_convert_to_float


def read_netzsch_sta_file(file_path):
    """
    Convenience wrapper around NetzschSTAParser.
    """
    parser = NetzschSTAParser(file_path=file_path)
    return parser.parse()


class NetzschSTAParser:
    """
    Parser for NETZSCH STA/DSC/TGA ASCII export files.

    Expected structure
    ------------------
    - metadata lines starting with '#'
    - table header line starting with '##'
    - data rows below the table header

    Example
    -------
    #EXPORTTYPE: ;DATA ALL ;;;
    #DECIMAL:    ;COMMA    ;;;
    #SEPARATOR:  ;SEMICOLON;;;
    ...
    ##Temp./°C;Time/min;DSC/(mW/mg);Mass/%

    29,958;0;7,67E-02;99,99772
    ...
    """

    # Maps the keyword found in the DECIMAL metadata field to the actual character.
    DECIMAL_MAP = {
        "COMMA": ",",
        "POINT": ".",
        "DOT": ".",
    }

    # Maps the keyword found in the SEPARATOR metadata field to the actual character.
    SEPARATOR_MAP = {
        "SEMICOLON": ";",
        "COMMA": ",",
        "TAB": "\t",
    }

    # Metadata fields whose values should be converted to numbers.
    # All other fields are kept as strings.
    NUMERIC_METADATA_KEYS = {
        "SAMPLE MASS /mg",
        "REFERENCE MASS /mg",
        "SAMPLE CRUCIBLE MASS /mg",
        "REFERENCE CRUCIBLE MASS /mg",
    }

    def __init__(self, file_path, instrument_file=None):
        """
        Parameters
        ----------
        file_path : str or Path
            Path to the NETZSCH export file.
        instrument_file : InstrumentFile, optional
            Pre-loaded InstrumentFile instance. If None, one is created.
        """
        self.file_path = file_path
        # Use an existing InstrumentFile if provided, otherwise create one.
        self.file = instrument_file or InstrumentFile(file_path).read()

        self.meta = dict()      # will hold all metadata key-value pairs
        self.data_df = None     # will hold the measurement table as a DataFrame

        # These are set while scanning for the '##' column header line.
        self.table_header_idx = None    # line index of the '##' row
        self.table_header_line = None   # the '##' row content (without the '##' prefix)

    def parse(self):
        """
        Parse metadata and data table.

        Returns
        -------
        meta : dict
            Parsed metadata.
        data_df : pandas.DataFrame
            Parsed measurement table.
        """
        self._parse_metadata_and_find_table_header()
        self._parse_data_table()

        # Record which encoding was used so the caller can inspect it.
        self.meta["USED_ENCODING"] = self.file.used_encoding

        return self.meta, self.data_df

    def _parse_metadata_and_find_table_header(self):
        """
        Parse metadata block and locate the table header line.
        """
        for idx, raw_line in enumerate(self.file.lines):
            line = raw_line.strip()

            # Skip blank lines.
            if not line:
                continue

            # A line starting with '##' is the column header — stop scanning metadata.
            if line.startswith("##"):
                self.table_header_idx = idx
                # Strip the leading '##' to get the raw column name string.
                self.table_header_line = line[2:].strip()
                break

            # A line starting with a single '#' is a metadata line.
            if line.startswith("#"):
                parsed = self._parse_metadata_line(line)
                if parsed is not None:
                    key, value = parsed
                    self.meta[key] = value

        # Convert numeric metadata fields from strings to numbers.
        self._postprocess_metadata()

        if self.table_header_idx is None:
            raise ValueError("Could not find table header line starting with '##'.")

    def _parse_metadata_line(self, line):
        """
        Parse one metadata line of the form:
        #KEY: ;VALUE ;;;

        Returns
        -------
        tuple[str, str | list[str] | None] or None
        """
        # Remove the leading '#' characters before processing.
        cleaned = line.lstrip("#").strip()

        # Split on the column separator and remove trailing empty filler cells.
        column_sep = self._get_column_separator()
        cells = split_line(cleaned, sep=column_sep)
        cells = strip_empty_edges(cells)

        if not cells:
            return None

        first_cell = cells[0]

        # Only lines containing ":" are treated as metadata.
        if ":" not in first_cell:
            return None

        # Split "KEY: value" into key and its first value fragment.
        key, first_value = first_cell.split(":", maxsplit=1)
        key = key.strip()
        first_value = first_value.strip()

        # Collect all non-empty value fragments (some fields span multiple cells).
        values = []
        if first_value:
            values.append(first_value)

        if len(cells) > 1:
            values.extend(cell.strip() for cell in cells[1:] if cell.strip())

        # Store as None, a single string, or a list depending on how many
        # value fragments were found.
        if len(values) == 0:
            value = None
        elif len(values) == 1:
            value = values[0]
        else:
            value = values

        return key, value

    def _postprocess_metadata(self):
        """
        Convert selected metadata values to numeric types where appropriate.
        """
        decimal_sep = self._get_decimal_separator()

        # Only process the keys listed in NUMERIC_METADATA_KEYS.
        for key in self.NUMERIC_METADATA_KEYS:
            if key not in self.meta:
                continue

            value = self.meta[key]
            self.meta[key] = try_convert_to_float(value, decimal_sep)

    def _parse_data_table(self):
        """
        Parse the measurement table below the table header.
        """
        if self.table_header_idx is None or self.table_header_line is None:
            raise RuntimeError("Table header information is missing.")

        decimal_sep = self._get_decimal_separator()
        column_sep = self._get_column_separator()
        column_names = self._get_column_names(column_sep)

        # Take all lines after the column header row and join them back into
        # a single string so pandas can read it like a file.
        table_lines = self.file.lines[self.table_header_idx + 1:]
        table_text = "\n".join(table_lines)

        data_df = pd.read_csv(
            StringIO(table_text),
            sep=column_sep,
            decimal=decimal_sep,
            header=None,        # column names are provided manually via 'names'
            names=column_names,
            engine="python",
            skip_blank_lines=True,
        )

        # Clean up any stray '#' prefixes that some exports add to column names.
        data_df.columns = [col.lstrip("#").strip() for col in data_df.columns]
        # Drop columns that are entirely empty (padding artefact in some exports).
        data_df = data_df.dropna(axis=1, how="all")

        self.data_df = data_df

    def _get_decimal_separator(self):
        """
        Determine decimal separator from metadata.
        """
        # Look for a "DECIMAL" entry in metadata; fall back to "POINT" (i.e. ".").
        decimal_token = str(self.meta.get("DECIMAL", "POINT")).upper()
        return self.DECIMAL_MAP.get(decimal_token, ".")

    def _get_column_separator(self):
        """
        Determine column separator from metadata.
        """
        # Look for a "SEPARATOR" entry in metadata; fall back to semicolon.
        separator_token = str(self.meta.get("SEPARATOR", "SEMICOLON")).upper()
        return self.SEPARATOR_MAP.get(separator_token, ";")

    def _get_column_names(self, column_sep):
        """
        Parse and clean table column names.
        """
        column_names = split_line(self.table_header_line, sep=column_sep)
        # Remove any leading '#' characters from column names.
        column_names = [name.lstrip("#").strip() for name in column_names]
        return column_names
