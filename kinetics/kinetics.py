import numpy as np
import pandas as pd

from scipy.interpolate import interp1d
from scipy.optimize import curve_fit, minimize
from typing import List, Dict, Union  # for type hints in functions
from FireSciPy.utils import series_to_numpy, ensure_nested_dict, get_nested_value, linear_model, calculate_residuals, calculate_R_squared, calculate_RMSE
from FireSciPy.constants import GAS_CONSTANT

# gas_const = 8.31446261815324  # J / (mol * K)


# def series_to_numpy(data: Union[np.ndarray, pd.Series]) -> np.ndarray:
#     """
#     Helper function that converts a Pandas Series to a NumPy array if necessary.
#
#     Parameters
#     ----------
#     :param data: Input data, which could be a Pandas Series or a NumPy array
#
#     Returns
#     -------
#     :return: NumPy array
#     """
#
#     if type(data) == pd.Series:
#         data = data.to_numpy()
#     elif type(data) == list:
#         data = np.array(data)
#     else:
#         data
#
#     return data


def initialize_investigation_skeleton(material, investigator=None, instrument=None, date=None, notes=None):
    """
    Initialize the skeleton for an investigation data structure.

    Parameters:
        material (str): Material being investigated.
        investigator (str): Name of the investigator.
        instrument (str): Device label.
        date (str): Date of the investigation.
        notes (str): Notes of the investigation.

    Returns:
        dict: Skeleton of the investigation data structure.
    """

    skeleton = {
        "general_info": {
            "material": material,
            "investigator": investigator,
            "instrument": instrument,
            "date": date,
            "notes": notes,
        },
        "experiments": dict(),
            # Add more experiment types as needed
        }

    return skeleton

#
# def ensure_nested_dict(d, keys):
#     """
#     Ensures a nested dictionary structure exists for the given keys.
#     Parameters:
#         d (dict): The dictionary to operate on.
#         keys (list): List of keys representing the nested path.
#     Returns:
#         dict: The final nested dictionary.
#     """
#     for key in keys:
#         if key not in d:
#             d[key] = dict()
#         d = d[key]
#     return d
#
#
# def get_nested_value(nested_dict, keys):
#     """
#     Access a nested dictionary using a list of keys.
#
#     Parameters:
#         nested_dict (dict): The nested dictionary to traverse.
#         keys (list): A list of keys specifying the path to the value.
#
#     Returns:
#         The value at the specified location in the nested dictionary.
#     """
#     current = nested_dict
#     for key in keys:
#         try:
#             current = current[key]
#         except ValueError:
#             print(f" * The key '{key}' does not exist.")
#             return None  # Return `None` (or another default) if the key is not found
#     return current


def add_isothermal_tga(database, condition, repetition, raw_data, data_type=None, set_value=None):
    """
    Add raw data for an isothermal TGA experiment to the database.

    This function ensures that the hierarchical structure for storing isothermal
    TGA experiment data is present in the database. If the structure does not exist,
    it will be created. Then, the provided raw data for a specific experimental
    condition and repetition is added to the database.

    Parameters:
        database (dict): The main dictionary where all experimental data is stored.
        condition (str): The experimental condition (e.g., "300_C") under which the
                         isothermal TGA data was collected.
        repetition (str): Identifier for the specific repetition of the experiment
                          (e.g., "Rep_1", "Rep_2").
        raw_data (pd.DataFrame): The raw data collected for the specific experiment
                                 and repetition.
        data_type (str, optional): The type of data ("differential" or "integral").
                                   Defaults to None, leaving it unchanged if already defined.
        set_value (list): The nominal temperature program, value and unit [float, str] (e.g. [300, "°C"]). Defaults to None.

    Returns:
        None: Updates the dictionary in place, adding the isothermal data under the specified condition.
    """

    # Ensure the path exists
    path_keys = ["experiments", "TGA", "isothermal", condition, "raw"]
    raw_dict = ensure_nested_dict(database, path_keys)

    # Get the nominal temperature program setting
    if set_value == None:
        nominal_beta = {"Value": None, "Unit": None}
    else:
        nominal_beta = {"Value": set_value[0], "Unit": set_value[1]}
    database["experiments"]["TGA"]["isothermal"][condition]["set_value"] = nominal_beta

    # Add data type
    expected_types = ["differential", "integral"]
    if data_type not in expected_types:
        raise ValueError(f" * Either 'differential' or 'integral' needs to be provided for 'data_type'!")
    else:
        database["experiments"]["TGA"]["isothermal"][condition]["data_type"] = data_type

    # Add the raw data for the given repetition
    raw_dict[repetition] = raw_data


def add_constant_heating_rate_tga(database, condition, repetition, raw_data, data_type=None, set_value=None):
    """
    Add raw data for a constant heating rate TGA experiment to the database.

    This function ensures that the hierarchical structure for storing constant
    heating rate TGA experiment data is present in the database. If the structure
    does not exist, it will be created. Then, the provided raw data for a
    specific experimental condition and repetition is added to the database.

    Parameters:
        database (dict): The main dictionary where all experimental data is stored.
        condition (str): The experimental condition (e.g., "10_Kmin") under which the
                         constant heating rate TGA data was collected.
        repetition (str): Identifier for the specific repetition of the experiment
                          (e.g., "Rep_1", "Rep_2").
        raw_data (pd.DataFrame): The raw data collected for the specific experiment
                                 and repetition.
        data_type (str, optional): The type of data ("differential" or "integral").
                                   Defaults to None, leaving it unchanged if already defined.
        set_value (list): The nominal temperature program, value and unit [float, str] (e.g. [2.5, "K/min"]). Defaults to None.

    Returns:
        None: Updates the dictionary in place, adding the constant heating rate data under the specified condition.
    """

    # Ensure the path exists
    path_keys = ["experiments", "TGA", "constant_heating_rate", condition, "raw"]
    raw_dict = ensure_nested_dict(database, path_keys)

    # Get the nominal temperature program setting
    if set_value == None:
        nominal_beta = {"Value": None, "Unit": None}
    else:
        nominal_beta = {"Value": set_value[0], "Unit": set_value[1]}
    database["experiments"]["TGA"]["constant_heating_rate"][condition]["set_value"] = nominal_beta

    # Add data type
    expected_types = ["differential", "integral"]
    if data_type not in expected_types:
        raise ValueError(f" * Either 'differential' or 'integral' needs to be provided for 'data_type'!")
    else:
        database["experiments"]["TGA"]["constant_heating_rate"][condition]["data_type"] = data_type

    # Add the raw data for the given repetition
    raw_dict[repetition] = raw_data


def combine_isothermal_repetitions(database, condition, column_mapping=None):
    """
    Combine raw data from multiple repetitions under a specific isothermal condition.

    Parameters:
        database (dict): The main data structure storing all experimental data.
        condition (str): The isothermal condition to combine (e.g., "300_C").
        column_mapping (dict, optional): Mapping of user-defined column labels
                                         to standardised labels ('time', 'temp', 'mass').
                                         Example: {'time': 'Time (s)', 'temp': 'Temperature (deg C)', 'mass': 'Weight (mg)'}

    Returns:
        None: Updates the dictionary in place, adding the combined data under the specified condition.
    """
    # Standardised column labels
    standard_columns = {
        'time': 'Time',
        'temp': 'Temperature',
        'mass': 'Mass'
    }

    # Merge provided mappings with defaults
    column_mapping = {**standard_columns, **(column_mapping or {})}

    # Access raw data
    raw_data = database["experiments"]["TGA"]["isothermal"][condition]["raw"]

    # Step 1: Determine the longest time array
    time_col = column_mapping["time"]
    longest_time = None
    for rep in raw_data.values():
        if longest_time is None or rep[time_col].iloc[-1] > longest_time.iloc[-1]:
            longest_time = rep[time_col]

    # Interpolation reference
    reference_time = longest_time.values

    # Step 2: Interpolate all repetitions
    temp_col = column_mapping["temp"]
    mass_col = column_mapping["mass"]
    combined_data = {time_col: reference_time}
    for rep_name, rep_data in raw_data.items():

        # Ensure required columns exist in the raw data
        for col in [time_col, temp_col, mass_col]:
            if col not in rep_data.columns:
                raise ValueError(f"Column '{col}' must exist in the raw data for repetition '{rep_name}'.")

        # Interpolate temperature and mass
        combined_data[f"{temp_col}_{rep_name}"] = np.interp(reference_time, rep_data[time_col], rep_data[temp_col])
        combined_data[f"{mass_col}_{rep_name}"] = np.interp(reference_time, rep_data[time_col], rep_data[mass_col])

    # Step 3: Compute averages and standard deviations
    combined_data[f"{temp_col}_Avg"] = np.mean(
        [combined_data[key] for key in combined_data if key.startswith(temp_col + "_")], axis=0
    )
    combined_data[f"{temp_col}_Std"] = np.std(
        [combined_data[key] for key in combined_data if key.startswith(temp_col + "_")], axis=0
    )
    combined_data[f"{mass_col}_Avg"] = np.mean(
        [combined_data[key] for key in combined_data if key.startswith(mass_col + "_")], axis=0
    )
    combined_data[f"{mass_col}_Std"] = np.std(
        [combined_data[key] for key in combined_data if key.startswith(mass_col + "_")], axis=0
    )

    # Step 4: Store the combined data back into the dictionary
    database["experiments"]["TGA"]["isothermal"][condition]["combined"] = pd.DataFrame(combined_data)


def combine_constant_heating_rate_repetitions(database, condition, column_mapping=None):
    """
    Combine raw data from multiple repetitions under a specific constant heating rate condition.

    Parameters:
        database (dict): The main data structure storing all experimental data.
        condition (str): The constant heating rate condition to combine (e.g., "10_Kmin").
        column_mapping (dict, optional): Mapping of user-defined column labels
                                         to standardised labels ('time', 'temp', 'mass').
                                         Example: {'time': 'Time (s)', 'temp': 'Temperature (deg C)', 'mass': 'Weight (mg)'}

    Returns:
        None: Updates the dictionary in place, adding the combined data under the specified condition.
    """

    # Standardised column labels
    standard_columns = {
        'time': 'Time',
        'temp': 'Temperature',
        'mass': 'Mass'
    }
    time_col_default = standard_columns["time"]
    temp_col_default = standard_columns["temp"]
    mass_col_default = standard_columns["mass"]

    if column_mapping is None:
        column_mapping = standard_columns
#     # Merge provided mappings with defaults
#     column_mapping = {**standard_columns, **(column_mapping or {})}

    # Access raw data
    raw_data = database["experiments"]["TGA"]["constant_heating_rate"][condition]["raw"]

    # Step 1: Determine the longest time array
    time_col = column_mapping["time"]
    longest_time = None
    for rep in raw_data.values():
        if longest_time is None or rep[time_col].iloc[-1] > longest_time.iloc[-1]:
            longest_time = rep[time_col]

    # Interpolation reference
    reference_time = longest_time.values

    # Step 2: Interpolate all repetitions
    temp_col = column_mapping["temp"]
    mass_col = column_mapping["mass"]
    combined_data = {time_col_default: reference_time}
    for rep_name, rep_data in raw_data.items():

        # Ensure required columns exist in the raw data
        for col in [time_col, temp_col, mass_col]:
            if col not in rep_data.columns:
                raise ValueError(f"Column '{col}' must exist in the raw data for repetition '{rep_name}'.")

        # Interpolate temperature and mass
        combined_data[f"{temp_col_default}_{rep_name}"] = np.interp(reference_time, rep_data[time_col], rep_data[temp_col])
        combined_data[f"{mass_col_default}_{rep_name}"] = np.interp(reference_time, rep_data[time_col], rep_data[mass_col])

    # Step 3: Compute averages and standard deviations
    combined_data[f"{temp_col_default}_Avg"] = np.mean(
        [combined_data[key] for key in combined_data if key.startswith(temp_col_default + "_")], axis=0
    )
    combined_data[f"{temp_col_default}_Std"] = np.std(
        [combined_data[key] for key in combined_data if key.startswith(temp_col_default + "_")], axis=0
    )
    combined_data[f"{mass_col_default}_Avg"] = np.mean(
        [combined_data[key] for key in combined_data if key.startswith(mass_col_default + "_")], axis=0
    )
    combined_data[f"{mass_col_default}_Std"] = np.std(
        [combined_data[key] for key in combined_data if key.startswith(mass_col_default + "_")], axis=0
    )

    # Step 4: Store the combined data back into the dictionary
    database["experiments"]["TGA"]["constant_heating_rate"][condition]["combined"] = pd.DataFrame(combined_data)


def differential_conversion(differential_data, m_0=None, m_f=None):
    # TODO: add differential conversion computation
    raise ValueError(f" * Still under development.")
    return


def integral_conversion(integral_data, m_0=None, m_f=None):
    """
    Calculate the conversion (alpha) from integral experimental data.

    This function computes the conversion (alpha) for a given series of
    integral experimental data, such as mass or concentration, based on the
    formula:
        alpha = (m_0 - m_i) / (m_0 - m_f)
    where:
        m_0 = initial mass/concentration,
        m_i = instantaneous mass/concentration,
        m_f = final mass/concentration.

    If `m_0` and `m_f` are not provided, they default to the first and last
    values of the `integral_data` series, respectively.

    Parameters:
        integral_data (pd.Series or np.ndarray): Experimental data representing
            integral quantities (e.g., mass over time) to calculate the conversion.
        m_0 (float, optional): Initial mass/concentration. Defaults to the first
            value of `integral_data`.
        m_f (float, optional): Final mass/concentration. Defaults to the last
            value of `integral_data`.

    Returns:
        np.ndarray: Array of alpha values representing the conversion as a
        function of the provided integral data.
    """
    # Convert the input data to a numpy array for calculations
    m_i = series_to_numpy(integral_data)

    # Use the provided m_0 or default to the first value in the series
    m_0 = m_0 if m_0 is not None else m_i[0]

    # Use the provided m_f or default to the last value in the series
    m_f = m_f if m_f is not None else m_i[-1]

    # Calculate conversion (alpha) using the standard formula
    alpha = (m_0 - m_i) / (m_0 - m_f)

    return alpha


def compute_conversion(database, condition="all", setup="constant_heating_rate"):
    """
    Compute conversion for one or more experimental conditions in the database.

    Parameters:
        database (dict): The main data structure storing all experimental data.
        condition (str or list, optional): The specific experimental condition(s) to process.
            - If a string is provided, it can be a single condition (e.g., "300_C"), or "all" to process all conditions.
            - If a list is provided, it should contain multiple condition names.
        setup (str, optional): The experimental setup to process, either "isothermal" or "constant_heating_rate".
            Defaults to "constant_heating_rate".
        # m_0 (float, optional): Initial sample mass. If None, use the first mass value from the data.
        # m_f (float, optional): Final sample mass. If None, use the last mass value from the data.

    Returns:
        None: Adds conversion data directly into the database under each condition.
    """
    # Validate setup
    if setup not in {"isothermal", "constant_heating_rate"}:
        raise ValueError(f"Invalid setup '{setup}'. Must be 'isothermal' or 'constant_heating_rate'.")

    # Get all available conditions for the given setup
    available_conditions = database["experiments"]["TGA"].get(setup, {}).keys()

    # Determine which conditions to process
    if isinstance(condition, str):
        if condition == "all":
            conditions_to_process = available_conditions
        elif condition in available_conditions:
            conditions_to_process = [condition]
        else:
            raise KeyError(f"Condition '{condition}' not found in the '{setup}' setup.")
    elif isinstance(condition, list):
        # Ensure all conditions in the list exist in the database
        invalid_conditions = [cond for cond in condition if cond not in available_conditions]
        if invalid_conditions:
            raise KeyError(f"The following conditions were not found in the '{setup}' setup: {invalid_conditions}")
        conditions_to_process = condition
    else:
        raise TypeError("Condition must be a string ('all', specific condition) or a list of conditions.")

    # Helper function to process a single condition
    def process_condition(cond):
        # Check if combined data exists
        if "combined" not in database["experiments"]["TGA"][setup][cond]:
            raise KeyError(f"No 'combined' data found for condition '{cond}' under '{setup}' setup.")

        # Check for data type
        data_type = database["experiments"]["TGA"][setup][cond].get("data_type")
        if data_type not in {"integral", "differential"}:
            raise ValueError(f"Invalid or missing data type for condition '{cond}' under '{setup}' setup.")

        # Fetch combined data
        combined_data = database["experiments"]["TGA"][setup][cond]["combined"]
        time = combined_data["Time"]
        temp_avg = combined_data["Temperature_Avg"]
        mass_avg = combined_data["Mass_Avg"]


        # Compute conversion based on data type
        if data_type == "integral":
            alpha = integral_conversion(mass_avg)  # Optionally pass m_0 and m_f here
        elif data_type == "differential":
            alpha = differential_conversion(mass_avg)  # Placeholder for differential logic

        # Store the conversion data back in the database
        conversion_data = pd.DataFrame({
            "Time": time,
            "Temperature_Avg": temp_avg,
            "Mass_Avg": mass_avg,
            "Alpha": alpha})
        database["experiments"]["TGA"][setup][cond]["conversion"] = conversion_data

    # Process each condition
    for cond in conditions_to_process:
        process_condition(cond)


def compute_conversion_fractions(database, desired_points=None, setup="constant_heating_rate", condition="all"):
    """
    Interpolate conversion data to desired alpha (conversion) points for specified experimental conditions.

    Parameters:
        database (dict): The main data structure storing all experimental data.
        desired_points (array-like, optional): Desired conversion levels (alpha values) for interpolation.
            Defaults to np.linspace(0.05, 0.95, 37).
        setup (str, optional): The experimental setup to process, either "isothermal" or "constant_heating_rate".
            Defaults to "constant_heating_rate".
        condition (str or list, optional): The specific experimental condition(s) to process.
            - If a string is provided, it can be a single condition (e.g., "300_C"), or "all" to process all conditions.
            - If a list is provided, it should contain multiple condition names.

    Returns:
        None: Adds interpolated conversion fraction data directly into the database under each condition.
    """

    # Validate setup
    if setup not in {"isothermal", "constant_heating_rate"}:
        raise ValueError(f" * Invalid setup '{setup}'. Must be 'isothermal' or 'constant_heating_rate'.")

    # Default conversion levels.
    if desired_points is None:
        desired_points = np.linspace(0.05, 0.95, 37)

    # Check if desired_points is monotonic
    if not np.all(np.diff(desired_points) > 0) or not (0 <= np.min(desired_points) <= np.max(desired_points) <= 1):
        raise ValueError(" * `desired_points` must be a monotonic array of values between 0 and 1.")

    # Get all available conditions for the given setup
    available_conditions = database["experiments"]["TGA"].get(setup, {}).keys()

    # Determine which conditions to process
    if isinstance(condition, str):
        if condition == "all":
            conditions_to_process = available_conditions
        elif condition in available_conditions:
            conditions_to_process = [condition]
        else:
            raise KeyError(f"Condition '{condition}' not found in the '{setup}' setup.")
    elif isinstance(condition, list):
        # Ensure all conditions in the list exist in the database
        invalid_conditions = [cond for cond in condition if cond not in available_conditions]
        if invalid_conditions:
            raise KeyError(f" * The following conditions were not found in the '{setup}' setup: {invalid_conditions}")
        conditions_to_process = condition
    else:
        raise TypeError(" * Condition must be a string ('all', specific condition) or a list of conditions.")

    # Helper function to process a single condition
    def process_condition(cond):
        # # Check if combined data exists
        # if "combined" not in database["experiments"]["TGA"][setup][cond]:
        #     raise KeyError(f" * No 'combined' data found for condition '{cond}' under '{setup}' setup.")

#         # Check for data type
#         data_type = database["experiments"]["TGA"][setup][cond].get("data_type")
#         if data_type not in {"integral", "differential"}:
#             raise ValueError(f" * Invalid or missing data type for condition '{cond}' under '{setup}' setup.")

        # Check if conversion data exists
        if "conversion" not in database["experiments"]["TGA"][setup][cond]:
            raise KeyError(f" * Conversion data is missing for condition '{cond}'. Run `compute_conversion` first.")


        # Fetch conversion data
        conversion_data = database["experiments"]["TGA"][setup][cond]["conversion"]
        time = conversion_data["Time"]
        temp_avg = conversion_data["Temperature_Avg"]
        alpha_avg = conversion_data["Alpha"]

        # Check if desired points are within range of the provided conversion data
        if desired_points[0] < np.min(alpha_avg) or desired_points[-1] > np.max(alpha_avg):
            raise ValueError(
                f" * Desired points {desired_points} exceed the range of available alpha values: "
                f"   [{np.min(alpha_avg):.3f}, {np.max(alpha_avg):.3f}] for condition '{cond}'.")

        # Interpolate data
        new_time = np.interp(desired_points, alpha_avg, time)
        new_temp = np.interp(desired_points, alpha_avg, temp_avg)

        # Store the conversion levels back in the database
        conversion_fractions = pd.DataFrame({
            "Time": new_time,
            "Temperature_Avg": new_temp,
            "Alpha": desired_points})
        database["experiments"]["TGA"][setup][cond]["conversion_fractions"] = conversion_fractions

    # Process each condition
    for cond in conditions_to_process:
        process_condition(cond)



# def linear_model(x, m, b):
#     """
#     Linear model function: y = mx + b.
#     """
#
#     return m * x + b


def KAS_Ea(temperature, heating_rate, exponent_B=1.92, C=1.0008):
    """
    Kissinger–Akahira–Sunose method (KAS) with optional Starink improvement.
    Estimates the activation energy (E_a) and pre-exponential factor (A) for a given
    level of conversion. This estimation is based on a linear fit,
    following the isoconersional assumption.

    Reference:
    Formular 3.10 in
    ICTAC Kinetics Committee recommendations for performing kinetic
    computations on thermal analysis data
    (Vyazovkin et al., 2011, doi:10.1016/j.tca.2011.03.034)

    Parameters:
        temperature (array-like): Sample temperatures in Kelvin.
        heating_rate (array-like): Heating rates in Kelvin per second.
        exponent_B (float): Exponent for temperature (default: 1.92 for Starink improvement).
        C (float): Coefficient for activation energy calculation (default: 1.0008 for Starink).

    :return: list, containing:
        parameters of the linear fit,
        activation energy for specified level of conversion (Ea_i) in J/mol
        pre-exponential factor for specified level of conversion (A) in 1/s
        list of the points used for the linear fit

    """
    # Ensure numpy arrays
    temperature = series_to_numpy(temperature)
    heating_rate = series_to_numpy(heating_rate)

    # Input validation
    if len(temperature) != len(heating_rate):
        raise ValueError("temperature and heating_rate must have the same length.")
    if np.any(temperature <= 0) or np.any(heating_rate <= 0):
        raise ValueError("temperature and heating_rate must be positive.")
    if np.any(np.power(temperature, exponent_B) <= 0):
        raise ValueError("Temperature raised to exponent B must be positive.")

    # Prepare x and y data for the linear fit
    data_x = 1/temperature
    data_y = np.log(heating_rate / np.power(temperature, exponent_B))

    # Perform the linear fit
    popt, pcov = curve_fit(linear_model,
                           data_x, data_y,
                           maxfev=10000)

    # Extract the fitted parameters
    m_fit, b_fit = popt

    # Calculate estimate of (Ea_i), in J/mol.
    Ea_i = -(m_fit * GAS_CONSTANT) / C

    # Calculate estimate of pre-exponential factor (A)
    A_i = (Ea_i / GAS_CONSTANT) * np.exp(b_fit)

    return [popt, Ea_i, A_i, [data_x, data_y]]

#
# def calculate_residuals(data_y, y_fit):
#     """
#     Compute the residuals between observed data and fitted values.
#
#     Residuals represent the difference between actual data points (data_y) and the
#     corresponding predicted values (y_fit). This function is useful for assessing
#     the goodness of fit in regression or curve fitting problems.
#
#     Parameters:
#     data_y (array-like): The observed data values.
#     y_fit (array-like): The predicted or fitted values.
#
#     Returns:
#     np.ndarray: The residuals, calculated as data_y - y_fit.
#     """
#     # Element-wise subtraction of predicted values from actual data
#     residuals = data_y - y_fit
#     return residuals
#
#
# def calculate_R_squared(residuals, data_y):
#     """
#     Calculate the coefficient of determination (R-squared) for a set of data.
#
#     R-squared is defined as:
#         R² = 1 - (SS_res / SS_tot)
#     where SS_res is the sum of squares of residuals (the differences between the
#     observed and predicted values) and SS_tot is the total sum of squares (the
#     differences between the observed values and their mean). This metric indicates
#     the proportion of the variance in the dependent variable that is predictable
#     from the independent variable.
#
#     Parameters:
#         residuals (array-like): The residuals (errors) from the fitted model,
#                                 typically computed as (observed - predicted).
#         data_y (array-like): The array of observed data values.
#
#     Returns:
#         float: The R-squared value, which ranges from 0 to 1, where values closer
#                to 1 indicate a better fit.
#     """
#     # Calculate the sum of squares of residuals.
#     ss_res = np.sum(residuals**2)
#
#     # Calculate the total sum of squares relative to the mean of the observed data.
#     ss_tot = np.sum((data_y - np.mean(data_y))**2)
#
#     # Compute R-squared: 1 - (sum of squares of residuals divided by total sum of squares)
#     r_squared = 1 - (ss_res / ss_tot)
#
#     return r_squared
#
#
# def calculate_RMSE(residuals):
#     """
#     Compute the Root Mean Squared Error (RMSE) from residuals.
#
#     RMSE is a measure of the differences between predicted and observed values.
#     It provides an estimate of the standard deviation of residuals and is commonly
#     used to quantify the accuracy of a model.
#
#     Parameters:
#     residuals (array-like): The residuals (differences between observed and predicted values).
#
#     Returns:
#     float: The RMSE value, representing the average magnitude of residual errors.
#     """
#     # Compute RMSE by taking the square root of the mean squared residuals
#     rmse = np.sqrt(np.mean(residuals**2))
#     return rmse


def compute_Ea_KAS(database, data_keys=["experiments", "TGA", "constant_heating_rate"], **kwargs):
    """
    Wrapper function to easily compute activation energies using the Kissinger–Akahira–Sunose method.

    Parameters:
        database (dict): Nested dictionary containing the dataset.
        data_keys (list): List of keys to locate the dataset within the database.
        **kwargs: Additional arguments to pass to the KAS_Ea function.

    Returns:
        None: Stores the results in the parent dictionary of the specified dataset.
    """
    # Safely access the dataset
    dataset = get_nested_value(database, data_keys)
    if dataset is None:
        raise ValueError(f"Dataset not found at the specified keys: {data_keys}")

    # Safely access the parent dictionary to store results
    store_Ea = get_nested_value(database, data_keys[:-1])
    if store_Ea is None:
        raise ValueError(f"Unable to store results; parent keys not found: {data_keys[:-1]}")

    # Sort the keys of the dataset based on the heating rate values in the nested dictionary
    set_value_keys = sorted(dataset.keys(), key=lambda x: dataset[x]["set_value"]["Value"])

    # Sort the set values themselves
    set_values = sorted(dataset[key]["set_value"]["Value"] for key in dataset.keys())

    # Get number of conversion levels
    conversion_levels = dataset[set_value_keys[0]]["conversion_fractions"]["Alpha"]

    # Prepare data collection.
    Ea = list()
    A = list()
    m = list()
    b = list()
    r_squared = list()
    rmse = list()

    # Prepare placeholders for x and y values
    xy_data = {f"x{i+1}": [] for i in range(len(set_value_keys))}
    xy_data.update({f"y{i+1}": [] for i in range(len(set_value_keys))})

    # Iterate through conversion levels and compute results
    for conv_id, conversion_level in enumerate(conversion_levels):
        conversion_temperatures = list()
        for set_value_id, set_value_key in enumerate(set_value_keys):
            conv_temp = dataset[set_value_key]["conversion_fractions"]["Temperature_Avg"].iloc[conv_id]
            conversion_temperatures.append(conv_temp)

        # Compute activation energy
        popt, Ea_i, A_i, data_xy = KAS_Ea(conversion_temperatures, set_values, **kwargs)
        Ea.append(Ea_i)
        A.append(A_i)

        # Extract and store the fitted parameters.
        m_fit, b_fit = popt
        m.append(m_fit)
        b.append(b_fit)

        # Generate y-values from the fitted model.
        data_x, data_y = data_xy
        y_fit = linear_model(data_x, m_fit, b_fit)

        # Calculate residuals.
        residuals_i = calculate_residuals(data_y, y_fit)

        # Calculate R-squared.
        r_squared_i = calculate_R_squared(residuals_i, data_y)
        r_squared.append(r_squared_i)

        # Calculate RMSE.
        rmse_i = calculate_RMSE(residuals_i)
        rmse.append(rmse_i)

        # Store x and y values dynamically
        for i, (x_val, y_val) in enumerate(zip(data_x, data_y)):
            xy_data[f"x{i+1}"].append(x_val)
            xy_data[f"y{i+1}"].append(y_val)

    # Combine results
    Ea_results = pd.DataFrame(
        {"Conversion": conversion_levels,
         "Ea": np.array(Ea),
         "A": np.array(A),
         "m_fit": np.array(m),
         "b_fit": np.array(b),
         "R_squared": np.array(r_squared),
         "RMSE": np.array(rmse),
         **xy_data})

    # Collect results.
    store_Ea["Ea_results_KAS"] = Ea_results

#
# def reaction_rate(t, alpha, t_array, T_array, A, E, R=gas_const,
#                    reaction_model='nth_order', model_params=None):
#     """
#     Computes d(alpha)/dt at time t for a given alpha, using the
#     reaction rate constant k(T) (Arrhenius factor) and a chosen
#     reaction model f(alpha).
#     See formula (1.1) in [1].
#
#     Sergey Vyazovkin et al.; 10 June 2011
#     ICTAC Kinetics Committee recommendations for performing
#     kinetic computations on thermal analysis data
#     Thermochimica Acta, Volume 520, Issues 1–2, Pages 1-19
#     https://doi.org/10.1016/j.tca.2011.03.034
#
#     Parameters
#     ----------
#     t : float
#         Current time
#     alpha : float
#         Current conversion fraction
#     t_array : array-like
#         Times at which T_array is known
#     T_array : array-like
#         Temperatures corresponding to t_array
#     A : float
#         Pre-exponential factor
#     E : float
#         Activation energy
#     R : float
#         Gas constant
#     reaction_model : str
#         Key from the f_models dictionary
#     model_params : dict
#         Extra parameters for the chosen reaction model (e.g. {'n': 2.0})
#
#     Returns
#     -------
#     float
#         The derivative d(alpha)/dt
#     """
#     if model_params is None:
#         model_params = {}
#
#     # Interpolate temperature at current time
#     T_current = np.interp(t, t_array, T_array)
#
#     # Reaction rate constant (Arrhenius factor)
#     k_T = A * np.exp(-E / (R * T_current))
#
#     # Fetch the f(alpha) function from the dictionary
#     # f_alpha = f_models[reaction_model]
#     f_alpha = get_reaction_model(reaction_model)
#
#     # Check if generic or named model
#     if reaction_model in f_models:
#         # Evaluate f(alpha) with any extra model parameters
#         val_f_alpha = f_alpha(alpha, model_params)
#     elif reaction_model in named_models:
#         # Evaluate f(alpha) without extra model parameters
#         val_f_alpha = f_alpha(alpha)
#     else:
#         raise ValueError(f"Reaction model '{reaction_model}' not found.")
#
#     return k_T * val_f_alpha
#
# def solve_kinetics(t_array, T_array, alpha0, A, E, R=gas_const,
#                    reaction_model='nth_order', model_params=None):
#     """
#     Solve for alpha(t) over t_array using the reaction_rate ODE.
#
#     Parameters
#     ----------
#     t_array : array-like
#         Times at which to solve
#     T_array : array-like
#         Corresponding temperatures at those times
#     alpha0 : float
#         Initial conversion (e.g., 0)
#     A : float
#         Pre-exponential factor
#     E : float
#         Activation energy
#     R : float
#         Gas constant
#     model : str
#         Which f(alpha) model to use (key into f_models)
#     model_params : dict
#         Extra parameters for that model, e.g. {'n': 1.0}
#
#     Returns
#     -------
#     sol.t : array
#         The time grid of the solution
#     sol.y[0] : array
#         The computed alpha(t) at each time point
#     """
#     if model_params is None:
#         model_params = {}
#
#     # ODE wrapper for solve_ivp
#     def ode_wrapper(t, alpha):
#         return reaction_rate(t, alpha[0], t_array, T_array,
#                              A, E, R, reaction_model, model_params)
#
#     # Solve from t=0 to t=t_array[-1]
#     sol = solve_ivp(
#         ode_wrapper,
#         (t_array[0], t_array[-1]),
#         [alpha0],        # initial condition
#         t_eval=t_array,rtol=1e-8, atol=1e-10
#     )
#
#     return sol.t, sol.y[0]

#
# def gaussian(x, mu, sigma, a=1.0):
#     """
#     Compute the Gaussian (normal) distribution function.
#
#     Parameters:
#     -----------
#     x : float or ndarray
#         The input value(s) where the Gaussian function is evaluated.
#     mu : float
#         The mean (center) of the Gaussian distribution.
#     sigma : float
#         The standard deviation (spread) of the Gaussian distribution. Must be positive.
#     a : float
#         A scaling factor of the Gaussian distribution, default: 1.0.
#
#     Returns:
#     --------
#     float or ndarray
#         The computed value(s) of the Gaussian function at x.
#
#     Notes:
#     ------
#     The Gaussian function is defined as:
#         f(x) = (a / (sqrt(2 * pi))) * exp(-0.5 * ((x - mu) / sigma)^2)
#     """
#     exponent = -0.5 * ((x - mu) / sigma) ** 2
#     normalisation = a / (sigma * np.sqrt(2 * np.pi))
#     f_x =  normalisation * np.exp(exponent)
#     return f_x


def exp_difference(offset, temp_x1, temp_x2, data_y1, data_y2):
    """
    Computes the difference between two data series by means of root mean square error (RMSE).
    An offset is provided such that the difference can be minimised.
    Parameters:
        offset (float): to shift the second data series in x
        temp_x1 (numpy array): x-values of first data series
        temp_x2 (numpy array): x-values of second data series
        data_y1 (numpy array): y-values of first data series
        data_y2 (numpy array): y-values of second data series

    Returns:
        float: The RMSE value
    """
    # Interpolate data_y2 at temp_x1 shifted by offset
    interpolation = interp1d(temp_x2 + offset, data_y2, kind='linear', fill_value="extrapolate")
    data_y2_shifted = interpolation(temp_x1)

    # Compute RMSE
    residuals = calculate_residuals(data_y1, data_y2_shifted)
    RMSE = calculate_RMSE(residuals)

    return RMSE


def compute_optimal_shift(initial_guess, temp_x1, temp_x2, data_y1, data_y2, method="Powell"):
    """
    Computes the optimal shift between two data series to reduce the difference.
    Parameters:
        initial_guess (float): the initial guess value
        temp_x1 (numpy array): x-values of first data series
        temp_x2 (numpy array): x-values of second data series
        data_y1 (numpy array): y-values of first data series
        data_y2 (numpy array): y-values of second data series
        method (string): method used by scipy.optimize.minimise,
            default here "Powell", trying to avoid getting stuck in local optima

    Returns:
        float: The optimal shift that leads to the smallest RMSE
    """
    # Optimize temperature offset
    result = minimize(fun=exp_difference, x0=[initial_guess],
                      args=(temp_x1, temp_x2, data_y1, data_y2),
                      method=method)

    # Get optimal shift
    optimal_shift = result.x[0]

    return optimal_shift

#
# def get_reaction_model(model_name):
#     """
#     Unified function to retrieve the reaction rate model.
#     """
#     if model_name in named_models:
#         return named_models[model_name]
#     elif model_name in f_models:
#         return f_models[model_name]
#     else:
#         raise ValueError(f"Model '{model_name}' not found in reaction models.")
#
#
# # Define default clipping thresholds
# default_clip_values = {
#     "alpha_min": 1e-12,  # Avoids log(0) or division by zero
#     "alpha_max": 1       # Ensures alpha does not exceed 1
# }
#
# # Define function to enable the user to adjust the clipping of alpha
# def clip_alpha(alpha, clip_values):
#     """
#     Ensure alpha remains within numerical stability range.
#     Specifically, to deal with floating-point errors. They may
#     slightly push numbers outside of the expected range of alpha=[0,1].
#     This can lead to numerical errors when computations involve
#     logarithms or power laws.
#     """
#     return np.clip(alpha, clip_values["alpha_min"], clip_values["alpha_max"])
#
#
# # A dictionary of generic reaction models f(α) that take extra parameters:
# f_models = {
#     # Formula (1.9); https://doi.org/10.1016/j.tca.2011.03.034
#     'nth_order': lambda alpha, params, clip_values=default_clip_values:
#         (1 - clip_alpha(alpha, clip_values))**params['n'],
#
#     # Formula (1.8); https://doi.org/10.1016/j.tca.2011.03.034
#     'power_law': lambda alpha, params, clip_values=default_clip_values:
#         params['n'] * np.power(clip_alpha(alpha, clip_values), ((params['n']-1)/params['n'])),
#
#     # Formula (1.10); https://doi.org/10.1016/j.tca.2011.03.034
#     'Avrami_Erofeev': lambda alpha, params, clip_values=default_clip_values:
#         params['n'] * (1 - clip_alpha(alpha, clip_values)) * (-np.log(clip_alpha(1 - alpha, clip_values)))**((params['n']-1)/params['n']),
#
#     # Formulas (1.3) and (1.4); https://doi.org/10.1016/j.tca.2022.179384
#     'Sestak_Berggren': lambda alpha, params, clip_values=default_clip_values:
#         params['c'] * clip_alpha(alpha, clip_values)**params['m'] * (1 - clip_alpha(alpha, clip_values))**params['n'] * (-np.log(clip_alpha(1 - alpha, clip_values)))**params['p'],
#     # Add more named models as needed
# }
#
#
#
# # Dictionary of named models with fixed parameters
# named_models = {
#     'D3': lambda alpha, clip_values=default_clip_values: (3/2) * (1 - clip_alpha(alpha, clip_values))**(2/3) * (1 - (1 - clip_alpha(alpha, clip_values))**(1/3))**(-1),
#     'A2': lambda alpha, clip_values=default_clip_values: 2 * (1 - clip_alpha(alpha, clip_values)) * (-np.log(1 - clip_alpha(alpha, clip_values))) ** (1/2)
#     # Add more named models as needed
# }
#
#
# # ICTAC Kinetics Committee recommendations for performing kinetic computations on thermal analysis data
# # Sergey Vyazovkin et al., 2011
# # https://doi.org/10.1016/j.tca.2011.03.034
# # Table 1: Some of the kinetic models used in the solid-state kinetics.
#
# # Reaction models from Table 1.
# # Use NumPy functions, like np.exp(), to maintain np.array() compatibility.
# reaction_models = {
#     "P4": {
#         "Reaction Model": "Power law",
#         "f_alpha": lambda alpha: 4 * alpha ** (3/4),
#         "g_alpha": lambda alpha: alpha ** (1/4)
#     },
#     "P3": {
#         "Reaction Model": "Power law",
#         "f_alpha": lambda alpha: 3 * alpha ** (2/3),
#         "g_alpha": lambda alpha: alpha ** (1/3)
#     },
#     "P2": {
#         "Reaction Model": "Power law",
#         "f_alpha": lambda alpha: 2 * alpha ** (1/2),
#         "g_alpha": lambda alpha: alpha ** (1/2)
#     },
#     "P2/3": {
#         "Reaction Model": "Power law",
#         "f_alpha": lambda alpha: 2/3 * alpha ** (-1/2),
#         "g_alpha": lambda alpha: alpha ** (3/2)
#     },
#     "D1": {
#         "Reaction Model": "One-dimensional diffusion",
#         "f_alpha": lambda alpha: 1/2 * alpha ** (-1),
#         "g_alpha": lambda alpha: alpha ** 2
#     },
#     "F1": {
#         "Reaction Model": "Mampel (first order)",
#         "f_alpha": lambda alpha: 1 - alpha,
#         "g_alpha": lambda alpha: -np.log(1 - alpha)
#     },
#     "A4": {
#         "Reaction Model": "Avrami-Erofeev",
#         "f_alpha": lambda alpha: 4 * (1 - alpha) * (-np.log(1 - alpha)) ** (3/4),
#         "g_alpha": lambda alpha: (-np.log(1 - alpha)) ** (1/4)
#     },
#     "A3": {
#         "Reaction Model": "Avrami-Erofeev",
#         "f_alpha": lambda alpha: 3 * (1 - alpha) * (-np.log(1 - alpha)) ** (2/3),
#         "g_alpha": lambda alpha: (-np.log(1 - alpha)) ** (1/3)
#     },
#     "A2": {
#         "Reaction Model": "Avrami-Erofeev",
#         "f_alpha": lambda alpha: 2 * (1 - alpha) * (-np.log(1 - alpha)) ** (1/2),
#         "g_alpha": lambda alpha: (-np.log(1 - alpha)) ** (1/2)
#     },
#     "D3": {
#         "Reaction Model": "Three-dimensional diffusion",
#         "f_alpha": lambda alpha: 3/2 * (1 - alpha) ** (2/3) * (1 - (1 - alpha) ** (1/3)) ** (-1),
#         "g_alpha": lambda alpha: (1 - (1 - alpha) ** (1/3)) ** (2)
#     },
#     "R3": {
#         "Reaction Model": "Contracting sphere",
#         "f_alpha": lambda alpha: 3 * (1 - alpha) ** (2/3),
#         "g_alpha": lambda alpha: 1 - (1 - alpha) ** (1/3)
#     },
#     "R2": {
#         "Reaction Model": "Contracting cylinder",
#         "f_alpha": lambda alpha: 2 * (1 - alpha) ** (1/2),
#         "g_alpha": lambda alpha: 1 - (1 - alpha) ** (1/2)
#     },
#     "D2": {
#         "Reaction Model": "Two-dimensional diffusion",
#         "f_alpha": lambda alpha: (-np.log(1 - alpha)) ** (-1),
#         "g_alpha": lambda alpha: (1 - alpha) * np.log(1 - alpha) + alpha
#     },
# }
