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
