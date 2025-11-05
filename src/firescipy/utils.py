# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

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
