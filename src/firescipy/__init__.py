# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.


from . import utils
from . import pyrolysis
from . import constants
from . import handcalculation
from importlib.metadata import PackageNotFoundError, version


# IMPORTANT: must match the name in pyproject.toml
_PKG_NAME = "firescipy"

# Get version for pyproject.toml as single source of truth
try:
    __version__ = version(_PKG_NAME)
except PackageNotFoundError:
    # Running from a source checkout without installation:
    __version__ = "0.0.0"


# from .utils import ensure_nested_dict, get_nested_value, series_to_numpy, linear_model, calculate_residuals, calculate_R_squared, calculate_RMSE
#
# __all__ = ['ensure_nested_dict', 'get_nested_value', 'series_to_numpy', 'linear_model', 'calculate_residuals', 'calculate_R_squared', 'calculate_RMSE']
