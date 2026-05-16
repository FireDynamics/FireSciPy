# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

# Round-trip tests for isoconversional kinetics: synthetic data is generated
# with known kinetic parameters via solve_kinetics, fed through the analysis
# pipeline, and the recovered Ea is compared to the known input value.
# Using modelled data gives an exact reference — something real experiments
# cannot provide.

import numpy as np
import pandas as pd
import pytest

from firescipy.constants import GAS_CONSTANT
from firescipy.utils import ensure_nested_dict
from firescipy.pyrolysis.kinetics import (
    initialize_investigation_skeleton,
    compute_conversion_levels,
    compute_Ea_Friedman,
)
from firescipy.pyrolysis.modeling import (
    create_linear_temp_program,
    solve_kinetics,
)


# ---------------------------------------------------------------------------
# Shared model parameters
# ---------------------------------------------------------------------------

_A = 10**10 / 60    # pre-exponential factor in 1/s
_E = 125_400        # activation energy in J/mol (125.4 kJ/mol)
_ALPHA0 = 1e-12     # small non-zero start value to avoid division by zero
_T_START = 300      # K
_T_END = 750        # K
_N_POINTS = 451     # temperature resolution: ΔT ≈ 1 K


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_friedman_data_structure(heating_rates):
    """Build a data structure with modelled differential data for Friedman testing.

    Model data bypasses the normal raw-data pipeline
    (add_constant_heating_rate_tga / combine_repetitions / compute_conversion)
    and is inserted directly into the conversion level expected by
    compute_conversion_levels.
    """
    data_structure = initialize_investigation_skeleton(
        material="Test material",
        signal={"name": "HeatFlow", "unit": "W/g"})

    for hr_label, beta in heating_rates.items():
        hr_model = create_linear_temp_program(
            start_temp=_T_START, end_temp=_T_END,
            beta=beta, beta_unit="K/min", steps=_N_POINTS)

        t_array = hr_model["Time"]
        T_array = hr_model["Temperature"]

        t_sol, alpha_sol = solve_kinetics(
            t_array=t_array, T_array=T_array,
            A=_A, E=_E, alpha0=_ALPHA0,
            R=GAS_CONSTANT, reaction_model='nth_order',
            model_params={'n': 1.0})

        # Numerical gradient is acceptable here because the model curve is
        # smooth; the noise amplification problem only arises with real TGA data
        dAlpha_dt = np.gradient(alpha_sol, t_sol)

        hr_entry = ensure_nested_dict(
            data_structure,
            ["experiments", "TGA", "constant_heating_rate", hr_label])
        hr_entry["set_value"] = {"value": beta, "unit": "K/min"}
        hr_entry["data_type"] = "differential"
        hr_entry["conversion"] = pd.DataFrame({
            "Time": t_sol,
            "Temperature_Avg": T_array,
            "Alpha": alpha_sol,
            "dAlpha_dt": dAlpha_dt})

    return data_structure


# ---------------------------------------------------------------------------
# Friedman method tests
# ---------------------------------------------------------------------------

def test_compute_Ea_Friedman_recovers_known_Ea():
    """Friedman round-trip: modelled data with known Ea must be recovered within 1%."""
    heating_rates = {"5Kmin": 5, "10Kmin": 10, "20Kmin": 20, "40Kmin": 40}

    data_structure = _build_friedman_data_structure(heating_rates)

    conversion_levels = np.linspace(0.05, 0.95, 37)
    compute_conversion_levels(data_structure, desired_levels=conversion_levels)
    compute_Ea_Friedman(data_structure)

    Ea_recovered = data_structure["experiments"]["TGA"]["Ea_results_Friedman"]["Ea"].values

    assert np.allclose(Ea_recovered, _E, rtol=0.01), (
        f"Friedman Ea deviated more than 1% from input: "
        f"mean={np.mean(Ea_recovered) / 1000:.2f} kJ/mol, "
        f"expected={_E / 1000:.2f} kJ/mol")


def test_compute_Ea_Friedman_raises_for_integral_data():
    """compute_Ea_Friedman must raise ValueError when data_type is 'integral'."""
    data_structure = initialize_investigation_skeleton(
        material="Test material",
        signal={"name": "Mass", "unit": "mg"})

    hr_entry = ensure_nested_dict(
        data_structure,
        ["experiments", "TGA", "constant_heating_rate", "10Kmin"])
    hr_entry["set_value"] = {"value": 10, "unit": "K/min"}
    hr_entry["data_type"] = "integral"

    with pytest.raises(ValueError, match="integral"):
        compute_Ea_Friedman(data_structure)
