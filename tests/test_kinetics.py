# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""
Tests for firescipy.pyrolysis.kinetics — focusing on the Friedman
isoconversional method and the differential-data pathway in
compute_conversion.

Synthetic data strategy
-----------------------
All tests use a single-step, first-order Arrhenius decomposition:

    dα/dt = A · exp(−Ea / (R·T)) · (1 − α)

with known Ea = 120 kJ/mol and A = 1e10 1/s.  The analytical solution
for a constant-heating-rate experiment (β = dT/dt) is:

    α(T) = 1 − exp(−A/β · ∫_{T0}^{T} exp(−Ea/(R·T')) dT')

which is integrated numerically.  The differential signal dα/dt is
computed from the analytical expression and fed directly into the
database as ``data_type="differential"``, mimicking an instrument that
records the reaction rate directly (e.g. DTG).
"""

import warnings

import numpy as np
import pandas as pd
import pytest
from scipy.integrate import cumulative_trapezoid

from firescipy.pyrolysis.kinetics import (
    add_constant_heating_rate_tga,
    combine_repetitions,
    compute_conversion,
    compute_conversion_levels,
    compute_Ea_Friedman,
    compute_Ea_KAS,
    friedman_Ea,
    initialize_investigation_skeleton,
)
from firescipy.constants import GAS_CONSTANT


# ---------------------------------------------------------------------------
# Shared constants and helpers
# ---------------------------------------------------------------------------

EA_TRUE = 120_000.0   # J/mol
A_TRUE  = 1e10        # 1/s
BETAS_KMIN = [2.0, 5.0, 10.0, 20.0]   # K/min
T_START, T_END = 300.0, 900.0          # K
N_POINTS = 500
DESIRED_LEVELS = np.linspace(0.10, 0.90, 17)


def _make_run(beta_Kmin: float) -> pd.DataFrame:
    """Return a single constant-heating-rate run as a DataFrame.

    Columns: Time (s), Temperature (K), Mass (1/s).
    ``Mass`` holds dα/dt — the signal provided by a differential instrument.
    """
    beta = beta_Kmin / 60.0
    T = np.linspace(T_START, T_END, N_POINTS)
    t = (T - T_START) / beta
    k = A_TRUE * np.exp(-EA_TRUE / (GAS_CONSTANT * T))
    alpha = 1.0 - np.exp(-cumulative_trapezoid(k / beta, T, initial=0.0))
    dalpha_dt = k * (1.0 - alpha)
    return pd.DataFrame({"Time": t, "Temperature": T, "Mass": dalpha_dt})


def _build_database() -> dict:
    """Build a fully processed database with all four heating rates."""
    db = initialize_investigation_skeleton(
        material="SyntheticPolymer",
        signal={"name": "Mass", "unit": "1/s"},
    )
    for beta in BETAS_KMIN:
        key = f"{beta:.0f}_Kmin"
        add_constant_heating_rate_tga(
            db,
            condition=key,
            repetition="Rep_1",
            raw_data=_make_run(beta),
            data_type="differential",
            set_value=[beta, "K/min"],
        )
        combine_repetitions(db, condition=key)
        compute_conversion(db, condition=key)
        compute_conversion_levels(db, desired_levels=DESIRED_LEVELS, condition=key)
    return db


# ---------------------------------------------------------------------------
# friedman_Ea — unit tests
# ---------------------------------------------------------------------------

class TestFriedmanEa:
    """Unit tests for the low-level friedman_Ea function."""

    def _analytical_points(self, temperatures, A=A_TRUE, Ea=EA_TRUE):
        """Return exact dα/dt values for a first-order reaction at peak
        (α ≈ 0.5) for the given temperatures, using fixed kinetic parameters.
        This gives a set of (T, dα/dt) pairs that exactly satisfy the Friedman
        equation with the supplied Ea."""
        dalpha_dt = A * np.exp(-Ea / (GAS_CONSTANT * temperatures)) * 0.5
        return dalpha_dt

    def test_returns_correct_Ea(self):
        """friedman_Ea recovers the true activation energy from noiseless data."""
        temperatures = np.array([600.0, 700.0, 800.0, 900.0])
        dalpha_dt = self._analytical_points(temperatures)
        _, Ea_i, _ = friedman_Ea(temperatures, dalpha_dt)
        assert abs(Ea_i - EA_TRUE) / EA_TRUE < 1e-6

    def test_returns_correct_slope_sign(self):
        """Slope must be negative (Ea > 0 → m = −Ea/R < 0)."""
        temperatures = np.array([600.0, 700.0, 800.0, 900.0])
        dalpha_dt = self._analytical_points(temperatures)
        popt, _, _ = friedman_Ea(temperatures, dalpha_dt)
        m_fit = popt[0]
        assert m_fit < 0

    def test_fit_points_shape(self):
        """fit_points must contain two arrays of the same length as input."""
        temperatures = np.array([600.0, 700.0, 800.0, 900.0])
        dalpha_dt = self._analytical_points(temperatures)
        _, _, fit_points = friedman_Ea(temperatures, dalpha_dt)
        x, y = fit_points
        assert len(x) == len(temperatures)
        assert len(y) == len(temperatures)

    def test_accepts_pandas_series(self):
        """friedman_Ea must accept pandas Series as input."""
        temperatures = pd.Series([600.0, 700.0, 800.0, 900.0])
        dalpha_dt = pd.Series(self._analytical_points(temperatures.values))
        _, Ea_i, _ = friedman_Ea(temperatures, dalpha_dt)
        assert abs(Ea_i - EA_TRUE) / EA_TRUE < 1e-6

    def test_raises_on_length_mismatch(self):
        with pytest.raises(ValueError, match="same length"):
            friedman_Ea([600.0, 700.0], [0.01])

    def test_raises_on_nonpositive_temperature(self):
        with pytest.raises(ValueError, match="strictly positive"):
            friedman_Ea([0.0, 700.0], [0.01, 0.02])

    def test_raises_on_nonpositive_dalpha_dt(self):
        with pytest.raises(ValueError, match="strictly positive"):
            friedman_Ea([600.0, 700.0], [0.0, 0.02])

    def test_raises_on_negative_dalpha_dt(self):
        with pytest.raises(ValueError, match="strictly positive"):
            friedman_Ea([600.0, 700.0], [-0.01, 0.02])


# ---------------------------------------------------------------------------
# compute_Ea_Friedman — integration tests
# ---------------------------------------------------------------------------

class TestComputeEaFriedman:
    """Integration tests for the high-level compute_Ea_Friedman wrapper."""

    @pytest.fixture(scope="class")
    def db(self):
        return _build_database()

    def test_result_stored_in_database(self, db):
        """compute_Ea_Friedman stores Ea_results_Friedman in the expected location."""
        compute_Ea_Friedman(db)
        assert "Ea_results_Friedman" in db["experiments"]["TGA"]

    def test_result_is_dataframe(self, db):
        compute_Ea_Friedman(db)
        assert isinstance(db["experiments"]["TGA"]["Ea_results_Friedman"], pd.DataFrame)

    def test_result_columns(self, db):
        """Result DataFrame must contain all expected columns."""
        compute_Ea_Friedman(db)
        result = db["experiments"]["TGA"]["Ea_results_Friedman"]
        required = {"Conversion", "Ea", "m_fit", "b_fit", "R_squared", "RMSE"}
        assert required.issubset(result.columns)
        # One x/y pair per heating rate
        for i in range(1, len(BETAS_KMIN) + 1):
            assert f"x{i}" in result.columns
            assert f"y{i}" in result.columns

    def test_result_row_count(self, db):
        """One row per desired conversion level."""
        compute_Ea_Friedman(db)
        result = db["experiments"]["TGA"]["Ea_results_Friedman"]
        assert len(result) == len(DESIRED_LEVELS)

    def test_conversion_column_matches_desired_levels(self, db):
        compute_Ea_Friedman(db)
        result = db["experiments"]["TGA"]["Ea_results_Friedman"]
        np.testing.assert_allclose(result["Conversion"].values, DESIRED_LEVELS)

    def test_ea_accuracy(self, db):
        """Recovered Ea must be within 1 % of the true value at every level."""
        compute_Ea_Friedman(db)
        result = db["experiments"]["TGA"]["Ea_results_Friedman"]
        relative_error = (result["Ea"] - EA_TRUE).abs() / EA_TRUE
        assert (relative_error < 0.01).all(), (
            f"Max relative Ea error: {relative_error.max():.4f}"
        )

    def test_r_squared_near_unity(self, db):
        """R² must be ≥ 0.999 for noiseless synthetic data."""
        compute_Ea_Friedman(db)
        result = db["experiments"]["TGA"]["Ea_results_Friedman"]
        assert (result["R_squared"] >= 0.999).all()

    def test_missing_dataset_raises(self):
        """Passing a wrong data_keys path must raise ValueError."""
        db = _build_database()
        with pytest.raises(ValueError, match="Dataset not found"):
            compute_Ea_Friedman(db, data_keys=["experiments", "TGA", "nonexistent"])

    def test_missing_conversion_raises(self):
        """A condition without conversion data must raise KeyError."""
        db = initialize_investigation_skeleton(
            material="Incomplete",
            signal={"name": "Mass", "unit": "1/s"},
        )
        key = "10_Kmin"
        add_constant_heating_rate_tga(
            db, condition=key, repetition="Rep_1",
            raw_data=_make_run(10.0), data_type="differential",
            set_value=[10.0, "K/min"],
        )
        combine_repetitions(db, condition=key)
        # Deliberately skip compute_conversion and compute_conversion_levels.
        # The function will fail when it tries to access 'conversion_fractions'
        # (needed for the conversion level grid) before reaching the per-condition
        # 'conversion' check — both are valid guards for this incomplete state.
        with pytest.raises(KeyError):
            compute_Ea_Friedman(db)

    def test_nonpositive_rate_emits_warning_and_stores_nan(self):
        """A conversion level with dα/dt ≤ 0 must warn and store NaN."""
        db = _build_database()
        # Corrupt the differential signal of one condition to be negative
        # at the first conversion level by flipping the sign of all values.
        for key in db["experiments"]["TGA"]["constant_heating_rate"]:
            conv = db["experiments"]["TGA"]["constant_heating_rate"][key]["conversion"]
            conv["Mass_Avg"] = -conv["Mass_Avg"]

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            compute_Ea_Friedman(db)
            assert any("non-positive dα/dt" in str(warning.message) for warning in w)

        result = db["experiments"]["TGA"]["Ea_results_Friedman"]
        assert result["Ea"].isna().any()


# ---------------------------------------------------------------------------
# compute_conversion differential bugfix
# ---------------------------------------------------------------------------

class TestComputeConversionDifferential:
    """Regression test for the bug where compute_conversion called
    differential_conversion with only one argument instead of (time, data)."""

    def test_differential_conversion_does_not_raise(self):
        """compute_conversion must not raise TypeError for differential data."""
        db = initialize_investigation_skeleton(
            material="BugfixCheck",
            signal={"name": "Mass", "unit": "1/s"},
        )
        key = "10_Kmin"
        add_constant_heating_rate_tga(
            db, condition=key, repetition="Rep_1",
            raw_data=_make_run(10.0), data_type="differential",
            set_value=[10.0, "K/min"],
        )
        combine_repetitions(db, condition=key)
        compute_conversion(db, condition=key)  # must not raise
        conv = db["experiments"]["TGA"]["constant_heating_rate"][key]["conversion"]
        assert "Alpha" in conv.columns
        # Allow a tiny floating-point overshoot at the upper boundary.
        assert (conv["Alpha"].values >= 0.0).all()
        assert (conv["Alpha"].values <= 1.0 + 1e-9).all()

    def test_alpha_monotonically_increasing(self):
        """Alpha computed from differential data must be non-decreasing."""
        db = initialize_investigation_skeleton(
            material="MonotonicCheck",
            signal={"name": "Mass", "unit": "1/s"},
        )
        key = "10_Kmin"
        add_constant_heating_rate_tga(
            db, condition=key, repetition="Rep_1",
            raw_data=_make_run(10.0), data_type="differential",
            set_value=[10.0, "K/min"],
        )
        combine_repetitions(db, condition=key)
        compute_conversion(db, condition=key)
        alpha = db["experiments"]["TGA"]["constant_heating_rate"][key]["conversion"]["Alpha"].values
        assert np.all(np.diff(alpha) >= -1e-10)


# ---------------------------------------------------------------------------
# Consistency: Friedman vs KAS on the same data
# ---------------------------------------------------------------------------

class TestFriedmanVsKAS:
    """Sanity-check that Friedman and KAS return Ea values in the same
    order of magnitude on identical data."""

    def test_friedman_and_kas_ea_agree_within_10_percent(self):
        db = _build_database()
        compute_Ea_Friedman(db)
        compute_Ea_KAS(db)

        friedman_ea = db["experiments"]["TGA"]["Ea_results_Friedman"]["Ea"].values
        kas_ea = db["experiments"]["TGA"]["Ea_results_KAS"]["Ea"].values

        # Both should be within 10 % of each other on clean synthetic data.
        relative_diff = np.abs(friedman_ea - kas_ea) / EA_TRUE
        assert (relative_diff < 0.10).all(), (
            f"Max relative difference between Friedman and KAS: {relative_diff.max():.4f}"
        )
