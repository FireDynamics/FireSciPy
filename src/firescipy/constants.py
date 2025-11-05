# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.


"""
Physical and chemical constants used throughout FireSciPy.

All values are in SI units unless otherwise noted.
"""

# Universal gas constant in J/(mol·K)
R = 8.31446261815324

# You could optionally alias it for clarity in different contexts
GAS_CONSTANT = R

# Optional: a central dictionary for programmatic access
CONSTANTS = {
    "R": R,
    "GAS_CONSTANT": GAS_CONSTANT,
}
