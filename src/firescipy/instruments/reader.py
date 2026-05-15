# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

# Import the individual instrument parsers from the same package.
# Each parser knows how to read one specific file format.
from .base import InstrumentFile
from .netzsch_sta import read_netzsch_sta_file
from .deatak_mcc import read_deatak_mcc_file
from .netzsch_cone import read_netzsch_cone_file

# Central registry: maps a human-readable instrument name to its parser function.
# To add support for a new instrument, add a new entry here and provide
# a corresponding parser module in this package.
FILE_TYPE_PARSERS = {
    "Netzsch STA": read_netzsch_sta_file,
    "Deatak MCC": read_deatak_mcc_file,
    "Netzsch Cone": read_netzsch_cone_file,
}

# Convenience list of all supported instrument names, derived from the registry.
SUPPORTED_TYPES = list(FILE_TYPE_PARSERS.keys())


def detect_file_type(file_path):
    """
    Detect instrument file type by inspecting file content.

    Returns a key from FILE_TYPE_PARSERS, or None if the type is unknown.
    """
    # Read the file using the generic loader (handles encoding automatically).
    f = InstrumentFile(file_path).read()

    # An empty file cannot be identified.
    if not f.lines:
        return None

    # Look at the very first line — each format has a unique marker there.
    first_line = f.lines[0].strip()

    # Netzsch STA / DSC / TGA exports always start with "#EXPORTTYPE".
    if first_line.startswith("#EXPORTTYPE"):
        return "Netzsch STA"

    # Netzsch Cone Calorimeter exports always start with "General information".
    if first_line.startswith("General information"):
        return "Netzsch Cone"

    # DEATAK MCC exports do not have a fixed first line, but always contain
    # a line with only "@" that separates metadata from measurement data.
    for line in f.lines:
        if line.strip() == "@":
            return "Deatak MCC"

    # None of the known markers were found — file type is unknown.
    return None


def read_instrument_file(file_path, file_type=None):
    """
    Load an instrument export file, optionally with a manually specified type.

    Auto-detection is attempted when file_type is None. If detection fails and
    no file_type is provided, a ValueError is raised with the list of supported
    types.

    Parameters
    ----------
    file_path : str or Path
        Path to the instrument export file.
    file_type : str, optional
        Force a specific parser. Must be one of SUPPORTED_TYPES.
        If None, the file type is detected automatically.

    Returns
    -------
    file_type : str
        The detected or provided file type.
    meta : dict
        Parsed metadata.
    df : pandas.DataFrame
        Parsed measurement table.

    Raises
    ------
    ValueError
        If the file type cannot be detected and no file_type is given,
        or if the provided file_type is not in SUPPORTED_TYPES.
    """
    # If no file type was provided by the caller, try to detect it automatically.
    if file_type is None:
        file_type = detect_file_type(file_path)

    # If detection also failed, stop and tell the user what went wrong.
    if file_type is None:
        raise ValueError(
            f"Could not detect instrument type for '{file_path}'. "
            f"Specify manually via file_type. Supported: {SUPPORTED_TYPES}"
        )

    # Guard against a typo or unsupported type passed in manually.
    if file_type not in FILE_TYPE_PARSERS:
        raise ValueError(
            f"Unknown file_type '{file_type}'. Supported: {SUPPORTED_TYPES}"
        )

    # Look up the correct parser from the registry and run it.
    meta, df = FILE_TYPE_PARSERS[file_type](file_path)

    # Return the file type alongside the data so the caller knows what was loaded.
    return file_type, meta, df
