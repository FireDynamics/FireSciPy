# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.


def split_line(line, sep=";"):
    """
    Split a line into stripped cells.
    """
    # Split on the separator and remove leading/trailing whitespace from each cell.
    return [cell.strip() for cell in line.split(sep)]


def strip_empty_edges(cells):
    """
    Remove empty cells from the end of a row.
    """
    # Many instrument exports pad rows with trailing empty fields (e.g. ";;;").
    # Remove them so column counts are not inflated.
    while cells and cells[-1] == "":
        cells.pop()
    return cells


def row_has_numeric_content(cells, decimal="."):
    """
    Heuristic: check whether at least one cell looks numeric.
    """
    for cell in cells:
        # Normalise the decimal separator to "." before attempting conversion,
        # since European locales often use "," as the decimal character.
        candidate = cell.replace(decimal, ".").replace(",", ".")
        try:
            float(candidate)
            return True     # at least one numeric cell found
        except ValueError:
            continue        # not numeric, try the next cell
    return False


def is_mostly_empty(cells):
    # Returns True if all cells are empty strings (blank row).
    non_empty = [cell for cell in cells if cell != ""]
    return len(non_empty) == 0


def try_convert_to_float(value, decimal=","):
    # If the value is already a list, apply the conversion to each element.
    if isinstance(value, list):
        return [try_convert_to_float(v, decimal) for v in value]

    # Non-string values (e.g. int, float, None) are returned unchanged.
    if not isinstance(value, str):
        return value

    candidate = value.strip()

    # Replace the locale-specific decimal separator with "." so Python's
    # float() can parse it (e.g. "3,14" → "3.14").
    if decimal != ".":
        candidate = candidate.replace(decimal, ".")

    try:
        return float(candidate)
    except ValueError:
        # The string is not a number — return it as-is.
        return value
