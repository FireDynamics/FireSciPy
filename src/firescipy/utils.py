# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import os
import subprocess
import warnings

import numpy as np
import pandas as pd

from typing import List, Dict, Union  # for type hints in functions


def series_to_numpy(data: Union[np.ndarray, pd.Series]) -> np.ndarray:
    """
    Helper function that converts a Pandas Series to a NumPy array if necessary.

    Parameters
    ----------
    data : np.ndarray or pd.Series
        Input data to be converted. If already a NumPy array,
        it is returned unchanged.

    Returns
    -------
    np.ndarray
        The input data as a NumPy array.
    """

    if type(data) == pd.Series:
        data = data.to_numpy()
    elif type(data) == list:
        data = np.array(data)
    else:
        data

    return data


def ensure_nested_dict(nested_dict, keys):
    """
    Ensures a nested dictionary structure exists the given sequence of keys.

    Parameters
    ----------
    nested_dict: dict
        The dictionary to operate on. It will be modified in place
        to include the full nested structure.
    keys : Sequence
        Sequence of keys representing the path of nested dictionaries to create.

    Returns
    -------
    dict
        The final nested dictionary at the end of the path.
    """

    # Iterate through keys to build the nested structure
    for key in keys:
        if key not in nested_dict:
            nested_dict[key] = dict()
        nested_dict = nested_dict[key]
    # Return nested dictionary
    return nested_dict


def store_in_nested_dict(nested_dict, new_data, keys):
    """
    Stores data in a nested dictionary structure, for the given keys.
    Intermediate levels will be created if they do not already exist.

    Parameters
    ----------
    nested_dict : dict
        The dictionary to operate on. It will be modified in place
    new_data : Any
        Data to store at the specified nested location.
    keys : Sequence
        Sequence of keys representing the nested path.

    Returns
    -------
    None
        The nested dictionary is changed in place.
    """

    if not isinstance(nested_dict, dict):
        raise TypeError("Expected 'dictionary' to be of type dict.")

    if not isinstance(keys, (list, tuple)) or not keys:
        raise ValueError("Expected 'keys' to be a non-empty list or tuple.")

    storage_location = ensure_nested_dict(nested_dict, keys[:-1])
    storage_location[keys[-1]] = new_data


def get_nested_value(nested_dict, keys):
    """
    Retrieve a value from a nested dictionary, using a sequence of keys.

    Parameters
    ----------
    nested_dict : dict
        The nested dictionary to traverse.
    keys : Sequence
        A sequence of keys (e.g., list or tuple) specifying the path
        to the data.

    Returns
    -------
    Any
        The data at the specified location in the nested dictionary.
    """

    current = nested_dict
    for key in keys:
        try:
            current = current[key]
        except KeyError:
            print(f" * The key '{key}' does not exist.")
            # Gracefully return None if any key is missing in the nested path
            return None

    return current


def linear_model(x, m, b):
    # TODO: Move to numpy polyfit/polyval?
    """
    Linear model function: y = mx + b.
    """

    return m * x + b


def calculate_residuals(data_y, fit_y):
    """
    Compute the residuals between observed data and fitted values.

    Residuals are calculated as the difference between actual values (data_y)
    and predicted values (fit_y). Useful for evaluating the quality of
    regression or curve fitting results.

    Parameters
    ----------
    data_y : array-like
        The observed data values.
    fit_y : array-like
        The predicted or fitted values.

    Returns
    -------
    np.ndarray
        The residuals, calculated as data_y - fit_y.
    """

    # Element-wise subtraction of predicted values from actual data
    residuals = data_y - fit_y
    return residuals


def calculate_R_squared(residuals, data_y):
    """
    Calculate the coefficient of determination (R-squared) for a set of data.

    R-squared is defined as:

    .. math::

        R^2 = 1 - (SS_{res} / SS_{tot})

    where SS_res is the sum of squares of residuals (the differences between the
    observed and predicted values) and SS_tot is the total sum of squares
    (the differences between the observed values and their mean).
    This metric indicates the proportion of the variance in the dependent
    variable that is explained by the model.

    Parameters
    ----------
    residuals : array-like
        The residuals (errors) from the fitted model,
        typically computed as (observed - predicted).
    data_y : array-like
        The observed data values.

    Returns
    -------
    float
        The R-squared value. Higher values indicate a better fit.
        Note: R² can be negative if the model performs worse
        than a constant mean.
    """

    # Calculate the sum of squares of residuals.
    ss_res = np.sum(residuals**2)

    # Calculate the total sum of squares relative to the mean of the observed data.
    ss_tot = np.sum((data_y - np.mean(data_y))**2)

    # Compute R-squared: 1 - (sum of squares of residuals divided by total sum of squares)
    r_squared = 1 - (ss_res / ss_tot)

    return r_squared


def calculate_RMSE(residuals):
    """
    Compute the Root Mean Squared Error (RMSE) from residuals.

    RMSE is the square root of the average of the squared residuals.
    It provides an estimate of the standard deviation of the prediction errors
    and is commonly used to quantify the accuracy of a model.

    Parameters
    ----------
    residuals : array-like
        The residuals (differences between observed and predicted values).

    Returns
    -------
    float
        The RMSE value, representing the standard deviation  of residual errors.
    """

    # Compute RMSE by taking the square root of the mean squared residuals
    rmse = np.sqrt(np.mean(residuals**2))
    return rmse


def gaussian(x, mu, sigma, a=1.0):
    """
    Compute the Gaussian (normal) distribution function.

    The Gaussian function is defined as shown below.

    .. math::

        f(x) = \\frac{a}{\\sigma \\sqrt{2\\pi}} \\exp\\left( -\\frac{1}{2} \\left( \\frac{x - \\mu}{\\sigma} \\right)^2 \\right)

    Parameters
    ----------
    x : float or ndarray
        The input value(s) where the Gaussian function is evaluated.
    mu : float
        The mean (center) of the Gaussian distribution.
    sigma : float
        The standard deviation (spread) of the Gaussian distribution.
        Must be positive.
    a : float
        A scaling factor of the Gaussian distribution, default: 1.0.

    Returns
    -------
    float or ndarray
        The computed value(s) of the Gaussian function at `x`.
    """

    exponent = -0.5 * ((x - mu) / sigma) ** 2
    normalisation = a / (sigma * np.sqrt(2 * np.pi))
    f_x =  normalisation * np.exp(exponent)
    return f_x


def dynamic_local_change_simplification(time, temperature, basis_tolerance=0.2):
    """
    Simplify a time-temperature dataset by dynamically assessing local changes
    against a "basis" change.

    This function allows to remove data points from a series but preserves the
    shape. This is helpful when defining a RAMP for FDS. For example, when a
    simulation is to be conducted where the temperature development of a heater
    over time is to be used as input, e.g. TGA, or cone calorimeter.
    It works as follows:

    The first point is retained. Then, the change between the first and second
    data point is established (basis change). A range is defined around the
    basis change, using the `basis_tolerance`. Next, it is determined if the
    change between the first and third point is inside that range. If this is
    true, the point is excluded and the change between first and fourth point
    is assessed. This process is repeated until a point is outside the range.
    This point is retained and the process starts again. The last point in the
    series is always retained.


    Parameters
    ----------
    time : ndarray
        Time values.
    temperature : ndarray
        Temperature values.
    basis_tolerance : float
        Tolerance range for deviations from the basis change (e.g., 0.2 = 20%).

    Returns
    -------
    ndarray
        Indices of the retained points in the original data.
    """

    # Always keep the first point
    retained_indices = [0]

    # Establish the basis change (change between the first two points)
    start_idx = 0
    basis_change = abs(temperature[1] - temperature[0])

    for i in range(1, len(temperature)):
        # Calculate the cumulative change since the current start point
        cumulative_change = abs(temperature[i] - temperature[start_idx])

        # Calculate the allowed range around the basis change
        lower_bound = basis_change * (1 - basis_tolerance)
        upper_bound = basis_change * (1 + basis_tolerance)

        # If the cumulative change exceeds the allowed range
        if cumulative_change < lower_bound or cumulative_change > upper_bound:
            # Keep the current point
            retained_indices.append(i)

            # Reset the start point and basis change
            start_idx = i
            if i + 1 < len(temperature):  # Check to avoid index errors
                basis_change = abs(temperature[i + 1] - temperature[i])
            else:
                basis_change = cumulative_change  # Final segment uses the last change

    # Always keep the last point
    retained_indices.append(len(temperature) - 1)

    return np.array(retained_indices)


def get_git_commit_hash(path_to_git_repo):
    """
    Retrieve the short and long commit hashes of the current HEAD in a local Git repository.

    This function runs ``git rev-parse`` in the specified repository directory and
    returns both the short and full (long) forms of the current HEAD commit hash.
    Note that uncommitted changes in the working tree are not reflected in these hashes.

    Parameters
    ----------
    path_to_git_repo : str or os.PathLike
        Path to the local clone of a Git repository.

    Returns
    -------
    rev_short : str
        Short form of the current HEAD commit hash (e.g., ``a1b2c3d``).
    rev_long : str
        Full (long) form of the current HEAD commit hash (e.g., ``a1b2c3d4e5f6...``).

    Raises
    ------
    RuntimeError
        If the path is not a Git repository, Git is not installed, or HEAD
        cannot be resolved.
    """

    try:
        rev_short = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=path_to_git_repo
        ).strip().decode()

        rev_long = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=path_to_git_repo
        ).strip().decode()

    except FileNotFoundError as e:
        raise RuntimeError(
            "Git does not seem to be installed or is not available on the PATH."
        ) from e

    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"Path '{path_to_git_repo}' is not a valid Git repository or HEAD is not defined."
        ) from e

    return rev_short, rev_long


def get_window_points(phi, delta_phi):
    """
    Compute the number of data points required to cover a given window width.

    Useful for determining the window size for smoothing operations where the
    desired window is specified in physical units (e.g. temperature in K or
    time in s) rather than in number of points.

    The result is always an odd number (required by most smoothing filters)
    and at least 5.

    Parameters
    ----------
    phi : array-like
        The independent variable series (e.g. temperature or time).
    delta_phi : float
        Desired window width in the same units as phi.

    Returns
    -------
    int
        Odd number of data points covering the requested window width.
    """

    # Estimate the typical spacing between consecutive data points.
    d_phi = np.median(np.diff(phi))

    # Convert the desired window width to a number of points.
    win_pts = int(round(delta_phi / d_phi))
    win_pts = max(win_pts, 5)

    # Smoothing filters (e.g. Savitzky-Golay) require an odd window size.
    if win_pts % 2 == 0:
        win_pts += 1

    return win_pts


def get_peak_value(x_values, y_values):
    """
    Find the peak (maximum) of a data series and return its coordinates and index.

    Parameters
    ----------
    x_values : array-like or pd.Series
        The independent variable (e.g. time or temperature).
    y_values : array-like or pd.Series
        The dependent variable whose maximum is sought.

    Returns
    -------
    x_peak : float
        The x-value at the peak.
    y_peak : float
        The y-value at the peak.
    peak_idx : int
        The index of the peak in the input arrays.
    """

    x_values = series_to_numpy(x_values)
    y_values = series_to_numpy(y_values)

    peak_idx = np.argmax(y_values)

    x_peak = x_values[peak_idx]
    y_peak = y_values[peak_idx]

    return x_peak, y_peak, peak_idx


def cubic_max_exact(coeffs, x1, x2):
    """
    Find the exact maximum of a cubic polynomial on the interval [x1, x2].

    Evaluates the polynomial at its critical points (where the derivative is
    zero) and at both endpoints, then returns the location and value of the
    global maximum on that interval.

    Parameters
    ----------
    coeffs : array-like
        Polynomial coefficients in descending order, as returned by
        ``numpy.polyfit`` with degree 3.
    x1 : float
        Left boundary of the search interval.
    x2 : float
        Right boundary of the search interval.

    Returns
    -------
    x_max : float
        The x-value at which the polynomial reaches its maximum.
    y_max : float
        The maximum value of the polynomial on [x1, x2].
    """

    p   = np.poly1d(coeffs)
    dp  = np.polyder(coeffs)    # first derivative (quadratic)

    # Critical points are where the first derivative is zero.
    crit_x = np.roots(dp)
    crit_x = crit_x[np.isreal(crit_x)].real

    # Keep only critical points strictly inside the interval.
    crit_x = crit_x[(crit_x > x1) & (crit_x < x2)]

    # Evaluate at critical points and both endpoints, return the maximum.
    xs = np.array([x1, x2, *crit_x])
    ys = p(xs)

    i = np.argmax(ys)
    return xs[i], ys[i]


def interpolate_experiment_data(time, temp, signal_1, signal_2=None,
                                T_step=0.5, mode="temperature_grid",
                                nominal_heating_rate=None, non_monotonic="raise"):
    """
    Interpolate experimental data to achieve a uniform temperature or time spacing.

    Accepts up to two signal columns, for example for STA data
    such as mass and heat flow.

    Two processing modes are available:

    1) ``"temperature_grid"``:
       A temperature grid with spacing ``T_step`` is created and used
       for interpolation. This assumes strictly increasing temperature.

    2) ``"nominal_rate"``:
       Time is treated as the independent variable and a time grid is
       built from the nominal heating rate and the desired temperature
       step size ``T_step``.

    Parameters
    ----------
    time : array-like or pd.Series
        Time series from the experimental data.
    temp : array-like or pd.Series
        Temperature series from the experimental data.
    signal_1 : array-like or pd.Series
        First signal from the experimental data, e.g. mass (TGA).
    signal_2 : array-like or pd.Series, optional
        Second signal from the experimental data, e.g. heat flow (STA).
    T_step : float
        Desired temperature step size in Kelvin, e.g. 0.5 K.
    mode : str
        Interpolation mode: ``"nominal_rate"`` or ``"temperature_grid"``.
    nominal_heating_rate : float, optional
        Nominal heating rate in K/min. Required when ``mode="nominal_rate"``.
    non_monotonic : str
        How to handle non-monotonic temperature data in ``"temperature_grid"``
        mode. Options are ``"raise"``, ``"warn"``, or ``"ignore"``.

    Returns
    -------
    tuple
        ``(new_time, new_temp, new_signal_1)`` or
        ``(new_time, new_temp, new_signal_1, new_signal_2)`` if signal_2
        is provided.

    Raises
    ------
    ValueError
        If ``T_step`` is not positive, input arrays have inconsistent lengths,
        ``non_monotonic`` receives an unknown value, or the mode is unknown.
    """

    time     = series_to_numpy(time)
    temp     = series_to_numpy(temp)
    signal_1 = series_to_numpy(signal_1)

    if signal_2 is not None:
        signal_2 = series_to_numpy(signal_2)

    if T_step <= 0:
        raise ValueError("T_step must be greater than zero.")

    if not (len(time) == len(temp) == len(signal_1)):
        raise ValueError("time, temp, and signal_1 must have the same length.")

    if signal_2 is not None and len(signal_2) != len(time):
        raise ValueError("signal_2 must have the same length as time and temp.")

    if mode == "nominal_rate":

        if not np.all(np.diff(time) > 0):
            raise ValueError(
                "Time data is not strictly monotonically increasing. "
                "Time-based interpolation may be unreliable."
            )

        # Convert heating rate from K/min to K/s to match time in seconds.
        beta   = nominal_heating_rate / 60
        t_step = T_step / beta

        t_start  = time[0]
        t_end    = time[-1]
        new_time = np.arange(t_start, t_end, t_step)

        new_temp     = np.interp(new_time, time, temp)
        new_signal_1 = np.interp(new_time, time, signal_1)

        if signal_2 is not None:
            new_signal_2 = np.interp(new_time, time, signal_2)
            return new_time, new_temp, new_signal_1, new_signal_2

        return new_time, new_temp, new_signal_1

    elif mode == "temperature_grid":

        if not np.all(np.diff(temp) > 0):
            msg = (
                "Temperature data is not strictly monotonically increasing. "
                "Temperature-based interpolation may be unreliable."
            )
            if non_monotonic == "raise":
                raise ValueError(msg)
            elif non_monotonic == "warn":
                warnings.warn(msg, stacklevel=2)
            elif non_monotonic == "ignore":
                pass
            else:
                raise ValueError(f"Unknown option for non_monotonic: {non_monotonic!r}")

        T_start  = temp[0]
        T_end    = temp[-1]
        new_temp = np.arange(T_start, T_end, T_step)

        new_time     = np.interp(new_temp, temp, time)
        new_signal_1 = np.interp(new_temp, temp, signal_1)

        if signal_2 is not None:
            new_signal_2 = np.interp(new_temp, temp, signal_2)
            return new_time, new_temp, new_signal_1, new_signal_2

        return new_time, new_temp, new_signal_1

    else:
        raise ValueError(f"Mode {mode!r} is unknown. Use 'nominal_rate' or 'temperature_grid'.")


def format_export_df(data_frame, column_decimals):
    """
    Create a formatted export copy of a pandas DataFrame.

    Each specified column is converted to a fixed-decimal string
    representation, which ensures consistent formatting when writing
    results to CSV or Markdown tables.

    Parameters
    ----------
    data_frame : pd.DataFrame
        The DataFrame to format. It is not modified in place.
    column_decimals : dict
        Mapping of column name to the number of decimal places, e.g.
        ``{"Temperature (K)": 2, "HRR (kW/m²)": 1}``.

    Returns
    -------
    pd.DataFrame
        A copy of ``data_frame`` with the specified columns formatted
        as strings.
    """

    export_df = data_frame.copy()
    for col, ndigits in column_decimals.items():
        if col in export_df.columns:
            export_df[col] = export_df[col].map(
                lambda x, n=ndigits: f"{x:.{n}f}" if pd.notna(x) else ""
            )

    return export_df
