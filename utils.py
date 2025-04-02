import numpy as np
import pandas as pd

from typing import List, Dict, Union  # for type hints in functions


def series_to_numpy(data: Union[np.ndarray, pd.Series]) -> np.ndarray:
    """
    Helper function that converts a Pandas Series to a NumPy array if necessary.

    Parameters
    ----------
    :param data: Input data, which could be a Pandas Series or a NumPy array

    Returns
    -------
    :return: NumPy array
    """

    if type(data) == pd.Series:
        data = data.to_numpy()
    elif type(data) == list:
        data = np.array(data)
    else:
        data

    return data


def ensure_nested_dict(d, keys):
    """
    Ensures a nested dictionary structure exists for the given keys.
    Parameters:
        d (dict): The dictionary to operate on.
        keys (list): List of keys representing the nested path.
    Returns:
        dict: The final nested dictionary.
    """
    for key in keys:
        if key not in d:
            d[key] = dict()
        d = d[key]
    return d


def get_nested_value(nested_dict, keys):
    """
    Access a nested dictionary using a list of keys.

    Parameters:
        nested_dict (dict): The nested dictionary to traverse.
        keys (list): A list of keys specifying the path to the value.

    Returns:
        The value at the specified location in the nested dictionary.
    """
    current = nested_dict
    for key in keys:
        try:
            current = current[key]
        except ValueError:
            print(f" * The key '{key}' does not exist.")
            return None  # Return `None` (or another default) if the key is not found
    return current


def linear_model(x, m, b):
    """
    Linear model function: y = mx + b.
    """

    return m * x + b


def calculate_residuals(data_y, y_fit):
    """
    Compute the residuals between observed data and fitted values.

    Residuals represent the difference between actual data points (data_y) and the
    corresponding predicted values (y_fit). This function is useful for assessing
    the goodness of fit in regression or curve fitting problems.

    Parameters:
    data_y (array-like): The observed data values.
    y_fit (array-like): The predicted or fitted values.

    Returns:
    np.ndarray: The residuals, calculated as data_y - y_fit.
    """
    # Element-wise subtraction of predicted values from actual data
    residuals = data_y - y_fit
    return residuals


def calculate_R_squared(residuals, data_y):
    """
    Calculate the coefficient of determination (R-squared) for a set of data.

    R-squared is defined as:
        R² = 1 - (SS_res / SS_tot)
    where SS_res is the sum of squares of residuals (the differences between the
    observed and predicted values) and SS_tot is the total sum of squares (the
    differences between the observed values and their mean). This metric indicates
    the proportion of the variance in the dependent variable that is predictable
    from the independent variable.

    Parameters:
        residuals (array-like): The residuals (errors) from the fitted model,
                                typically computed as (observed - predicted).
        data_y (array-like): The array of observed data values.

    Returns:
        float: The R-squared value, which ranges from 0 to 1, where values closer
               to 1 indicate a better fit.
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

    RMSE is a measure of the differences between predicted and observed values.
    It provides an estimate of the standard deviation of residuals and is commonly
    used to quantify the accuracy of a model.

    Parameters:
    residuals (array-like): The residuals (differences between observed and predicted values).

    Returns:
    float: The RMSE value, representing the average magnitude of residual errors.
    """
    # Compute RMSE by taking the square root of the mean squared residuals
    rmse = np.sqrt(np.mean(residuals**2))
    return rmse


def gaussian(x, mu, sigma, a=1.0):
    """
    Compute the Gaussian (normal) distribution function.

    Parameters:
    -----------
    x : float or ndarray
        The input value(s) where the Gaussian function is evaluated.
    mu : float
        The mean (center) of the Gaussian distribution.
    sigma : float
        The standard deviation (spread) of the Gaussian distribution. Must be positive.
    a : float
        A scaling factor of the Gaussian distribution, default: 1.0.

    Returns:
    --------
    float or ndarray
        The computed value(s) of the Gaussian function at x.

    Notes:
    ------
    The Gaussian function is defined as:
        f(x) = (a / (sqrt(2 * pi))) * exp(-0.5 * ((x - mu) / sigma)^2)
    """
    exponent = -0.5 * ((x - mu) / sigma) ** 2
    normalisation = a / (sigma * np.sqrt(2 * np.pi))
    f_x =  normalisation * np.exp(exponent)
    return f_x
