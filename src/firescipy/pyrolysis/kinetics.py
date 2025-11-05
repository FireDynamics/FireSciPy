# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.


import warnings

import numpy as np
import pandas as pd

from scipy.interpolate import interp1d
from scipy.optimize import curve_fit, minimize
from typing import List, Dict, Union  # for type hints in functions
from firescipy.utils import series_to_numpy, ensure_nested_dict, get_nested_value, linear_model, calculate_residuals, calculate_R_squared, calculate_RMSE
from firescipy.constants import GAS_CONSTANT


def initialize_investigation_skeleton(material, investigator=None, instrument=None, date=None, notes=None, signal=None):
    """
    Create a base dictionary representing a new investigation.

    This function returns a standardized data structure for documenting
    experimental investigations (e.g., thermal analysis). Optional metadata
    like investigator name, instrument, and signal description can be included.

    Parameters
    ----------
    material : str
        Material being investigated.
    investigator : str, optional
        Name of the investigator.
    instrument : str, optional
        Device label (e.g., "TGA" or "TGA/DSC 3+, Mettler Toledo").
    date : str, optional
        Date of the investigation.
    notes : str, optional
        Free-form notes or comments about the investigation.
    signal : dict, optional
        Dictionary describing the primary measurement signal.
        For example: {"name": "Mass", "unit": "mg"}.
        Common alternatives include {"name": "HeatFlow"}
        or {"name": "Enthalpy"}.
        This field guides downstream analysis and unit handling.

    Returns
    -------
    dict
        A dictionary representing the initialized investigation skeleton.
    """

    skeleton = {
        "general_info": {
            "material": material,
            "investigator": investigator,
            "instrument": instrument,
            "date": date,
            "notes": notes,
            "signal": signal
        },
        "experiments": dict(),
            # Add more experiment types as needed
        }

    return skeleton


def add_isothermal_tga(database, condition, repetition, raw_data, data_type=None, set_value=None):
    """
    Add raw data for an isothermal TGA experiment to the database.

    This function ensures that the hierarchical structure for storing isothermal
    TGA experiment data is present in the database. If the structure does not
    exist, it will be created. Then, the provided raw data for a specific
    experimental condition and repetition is added to the database.

    The `database` argument is expected to be a nested dictionary
    following the structure created by `initialize_investigation_skeleton()`.

    Parameters
    ----------
    database : dict
        The main dictionary where all experimental data is stored.
    condition : str
        The experimental condition (e.g., "300_C") under which the
        isothermal TGA data was collected.
    repetition : str
        Identifier for the specific repetition of the experiment
        (e.g., "Rep_1", "Rep_2").
    raw_data : pd.DataFrame
        The raw data collected for the specific condition and repetition.
    data_type : str, optional
        Type of data: "differential" or "integral". Defaults to None.
        If not provided, existing value in database is kept.
    set_value : list of [float, str]
        The nominal temperature program, value and unit [float, str]
        (e.g. [300, "°C"]). Defaults to None.

    Returns
    -------
    None
        Updates the database in place, adding the isothermal data under
        the specified condition and repetition.
    """

    # Ensure the path exists
    path_keys = ["experiments", "TGA", "isothermal", condition, "raw"]
    raw_dict = ensure_nested_dict(database, path_keys)

    # Get the nominal temperature program setting
    if set_value == None:
        nominal_beta = {"value": None, "unit": None}
    else:
        nominal_beta = {"value": set_value[0], "unit": set_value[1]}
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

    This function ensures that the hierarchical structure needed to store
    constant heating rate TGA data exists in the `database`.
    If not, it is created automatically. The provided raw data for a
    specific experimental condition and repetition is then added
    to the database.

    The `database` argument is expected to be a nested dictionary
    following the structure created by `initialize_investigation_skeleton()`.

    Parameters
    ----------
    database : dict
        The main dictionary where all experimental data is stored.
    condition : str
        The experimental condition (e.g., "10_Kmin") under which the
        constant heating rate TGA data was collected.
    repetition : str
        Identifier for the specific repetition of the experiment
        (e.g., "Rep_1", "Rep_2").
    raw_data : pd.DataFrame
        The raw data collected for the specific condition repetition.
    data_type : str, optional
        Type of data: "differential" or "integral". Defaults to None.
        If not provided, existing value in database is kept.
    set_value : list of [float, str]
        The nominal temperature program, value and unit [float, str]
        (e.g. [2.5, "K/min"]). Defaults to None.

    Returns
    -------
    None
        Updates the dictionary in place, adding the constant heating rate data under the specified condition.
    """

    # Ensure the path exists
    path_keys = ["experiments", "TGA", "constant_heating_rate", condition, "raw"]
    raw_dict = ensure_nested_dict(database, path_keys)

    # Get the nominal temperature program setting
    if set_value == None:
        nominal_beta = {"value": None, "unit": None}
    else:
        nominal_beta = {"value": set_value[0], "unit": set_value[1]}
    database["experiments"]["TGA"]["constant_heating_rate"][condition]["set_value"] = nominal_beta

    # Add data type
    expected_types = ["differential", "integral"]
    if data_type not in expected_types:
        raise ValueError(f" * Either 'differential' or 'integral' needs to be provided for 'data_type'!")
    else:
        database["experiments"]["TGA"]["constant_heating_rate"][condition]["data_type"] = data_type

    # Add the raw data for the given repetition
    raw_dict[repetition] = raw_data


def combine_repetitions(database, condition, temp_program="constant_heating_rate", column_mapping=None):
    """
    Combine raw data from multiple repetitions for a specific condition.

    This function extracts and combines raw data for a given experimental
    condition (e.g., "300_C" or "10_Kmin") under a specified temperature
    program (e.g., "isothermal" or "constant_heating_rate").

    The `column_mapping` parameter allows data files with non-standard column
    headers to be mapped into a unified format expected by downstream functions.
    For example, :func:`compute_conversion` expects columns labeled 'Time',
    'Temperature', and the signal name defined in the investigation skeleton.

    Parameters
    ----------
    database : dict
        The main data structure storing all experimental data.
        Should follow the format initialized
        by :func:`initialize_investigation_skeleton()`.
    condition : str
        The condition to combine (e.g., "300_C" or "10_Kmin").
    temp_program : str
        The temperature program of the experiment
        (e.g. "isothermal" or "constant_heating_rate").
    column_mapping : dict, optional
        Mapping from custom column labels to standardised ones.
        Expected keys: 'time', 'temp', and 'signal'.
        Here, 'signal' indicates the recorded quantity, like 'mass' in the TGA.
        Example::

            {
                'time': 'Time (s)',
                'temp': 'Temperature (deg C)',
                'signal': 'Mass (mg)'
            }

    Returns
    -------
    None
        Updates the dictionary in place, adding the combined data
        under the specified condition and temperature program.
    """

    # Check if proper temperature program is chosen
    assert temp_program in ("isothermal", "constant_heating_rate"), f"Unknown temp_program: '{temp_program}'"

    # Check if information about the recorded quantity exists
    if "signal" not in database.get("general_info", {}):
        warnings.warn("* Signal metadata missing in general_info. Using 'signal' as default label.")

    # Get info on recorded quantity, with fallbacks if missing
    signal_meta = database["general_info"].get("signal", dict())
    signal_name = signal_meta.get("name", "Signal")
    signal_unit = signal_meta.get("unit", "")

    # Get the raw data and set
    data_root = database["experiments"]["TGA"][temp_program][condition]
    raw_data = data_root["raw"]

    # Standardised column label definitions
    standard_columns = {
        'time': 'Time',
        'temp': 'Temperature',
        'signal': signal_name}  # User-defined signal (e.g., "Mass", "HeatFlow")

    # Merge user-provided column mapping with defaults,
    # allow replacement of individual key-value pairs instead of a full dict
    # Defaults to empty dict() if `column_mapping` is None.
    column_mapping = {**standard_columns, **(column_mapping or dict())}

    # Unpack final column names to be used
    time_col_default = standard_columns["time"]
    temp_col_default = standard_columns["temp"]
    signal_col_default = standard_columns["signal"]

    # Set column mapping from user input
    time_col = column_mapping["time"]
    temp_col = column_mapping["temp"]
    signal_col = column_mapping["signal"]

    # Step 1: Find reference time from longest series
    longest_time = None
    for rep in raw_data.values():
        if longest_time is None or rep[time_col].iloc[-1] > longest_time.iloc[-1]:
            longest_time = rep[time_col]
    reference_time = longest_time.values

    # Step 2: Interpolate all repetitions to reference time
    combined_data = {time_col_default: reference_time}
    for rep_name, rep_data in raw_data.items():
        for col, col_default in [(temp_col, temp_col_default), (signal_col, signal_col_default)]:
            if col not in rep_data.columns:
                raise ValueError(f"Missing column '{col}' in repetition '{rep_name}'.")

            combined_data[f"{col_default}_{rep_name}"] = np.interp(
                reference_time, rep_data[time_col], rep_data[col])

    # Step 3: Compute mean and std dev
    for col_default in [temp_col_default, signal_col_default]:
        values = [combined_data[k] for k in combined_data if k.startswith(f"{col_default}_")]
        combined_data[f"{col_default}_Avg"] = np.mean(values, axis=0)
        combined_data[f"{col_default}_Std"] = np.std(values, axis=0)

    # Step 4: Store and return
    df_combined = pd.DataFrame(combined_data)
    data_root["combined"] = df_combined
    return df_combined


def differential_conversion(differential_data, m_0=None, m_f=None):
    # TODO: add differential conversion computation
    raise ValueError(f" * Still under development.")
    return


def integral_conversion(integral_data, m_0=None, m_f=None):
    """
    Calculate the conversion (alpha) from integral experimental data.

    This function computes the conversion (alpha) for a series of
    integral experimental data, such as mass or concentration, based on the
    formula:

    .. math::

        \\alpha = \\frac{m_0 - m_i}{m_0 - m_f}

    where:
        m_0 = initial mass/concentration,
        m_i = instantaneous mass/concentration,
        m_f = final mass/concentration.

    If `m_0` and `m_f` are not provided, they default to the first and last
    values of the `integral_data` series, respectively.

    Parameters
    ----------
    integral_data : pd.Series or np.ndarray
        Experimental data representing integral quantities
        (e.g., mass or concentration over time) to calculate the conversion.
    m_0 : float, optional
        Initial mass/concentration. Defaults to the first
        value of `integral_data`.
    m_f : float, optional
        Final mass/concentration. Defaults to the last
        value of `integral_data`.

    Returns
    -------
    np.ndarray
        Array of alpha values representing the conversion.
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
    Compute conversion values from experimental or model data for one or more
    conditions.

    This function processes combined experimental data in the given `database`
    and adds conversion values (alpha) under each specified condition.

    Requires combined data to be present under each condition, typically created
    using :func:`combine_repetitions`.

    Parameters
    ----------
    database : dict
        The main data structure storing all experimental data.
    condition : str or list, optional
        The specific experimental condition(s) to process.
        If a string is provided, it can be a single condition
        (e.g., "300_C"), or "all" to process all conditions.
        If a list is provided, it should contain multiple condition labels.
    setup : str, optional
        The experimental setup to process, either "isothermal" or "constant_heating_rate". Defaults to "constant_heating_rate".

    Returns
    -------
    None
        Updates the database in place by adding conversion data under
        each condition.
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


def compute_conversion_levels(database, desired_levels=None, setup="constant_heating_rate", condition="all"):
    """
    Interpolate conversion (alpha) data to desired conversion levels (fractions)
    for specified experimental conditions.

    This function interpolates the conversion values across different
    conditions to align them with a common set of points. This ensures
    consistency in follow-up steps such as isoconversional analysis.

    Parameters
    ----------
    database : dict
        The main data structure storing all experimental data.
        Must follow the format initialized by
        :func:`initialize_investigation_skeleton`.
    desired_levels : array-like, optional
        Desired conversion levels (fractions) for interpolation.
        Defaults to `np.linspace(0.05, 0.95, 37)`.
    setup : str, optional
        The temperature program to process, either "isothermal" or
        "constant_heating_rate".
        Defaults to "constant_heating_rate".
    condition : str or list, optional
        The specific experimental condition(s) to process.
        If a string is provided, it can be a single condition (e.g., "300_C"),
        or "all" to process all conditions.
        If a list is provided, it should contain multiple condition names.

    Returns
    -------
    None
        Updates the database in place by adding the interpolated
        conversion levels under each condition and setup.
    """

    # Validate setup
    if setup not in {"isothermal", "constant_heating_rate"}:
        raise ValueError(f" * Invalid setup '{setup}'. Must be 'isothermal' or 'constant_heating_rate'.")

    # Default conversion levels.
    if desired_levels is None:
        desired_levels = np.linspace(0.05, 0.95, 37)

    # Check if desired_levels is monotonic
    if not np.all(np.diff(desired_levels) > 0) or not (0 <= np.min(desired_levels) <= np.max(desired_levels) <= 1):
        raise ValueError(" * `desired_levels` must be a monotonic array of values between 0 and 1.")

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

        # Check if desired levels are within range of the provided conversion data
        if desired_levels[0] < np.min(alpha_avg) or desired_levels[-1] > np.max(alpha_avg):
            raise ValueError(
                f" * Desired levels {desired_levels} exceed the range of available alpha values: "
                f"   [{np.min(alpha_avg):.3f}, {np.max(alpha_avg):.3f}] for condition '{cond}'.")

        # Interpolate data
        new_time = np.interp(desired_levels, alpha_avg, time)
        new_temp = np.interp(desired_levels, alpha_avg, temp_avg)

        # Store the conversion levels back in the database
        conversion_fractions = pd.DataFrame({
            "Time": new_time,
            "Temperature_Avg": new_temp,
            "Alpha": desired_levels})
        database["experiments"]["TGA"][setup][cond]["conversion_fractions"] = conversion_fractions

    # Process each condition
    for cond in conditions_to_process:
        process_condition(cond)


def KAS_Ea(temperature, heating_rate, B=1.92, C=1.0008):
    """
    Kissinger–Akahira–Sunose method (KAS), with Starink improvement by default.
    Estimates the activation energy for a given level of conversion
    :math:`E_{\\alpha}`. This estimation is based on a
    linear regression based on the isoconversional assumption.
    By default, the Starink improvement is used, i.e.: B = 1.92, C = 1.0008
    (https://doi.org/10.1016/S0040-6031(03)00144-8).
    The baseline KAS parameters would be: B = 2.0, C = 1.0.

    The KAS equation is presented below.

    .. math::

        \\ln\\left( \\frac{\\beta_i}{T_{\\alpha,i}^{B}} \\right)
        = \\text{Const} - C \\left( \\frac{E_{\\alpha}}{R\\, T_{\\alpha,i}} \\right)

    Formula 3.10 in: Vyazovkin et al. (2011). ICTAC Kinetics Committee
    recommendations for performing kinetic computations on thermal analysis data
    *Thermochimica Acta*, 520(1–2), 1–19.
    https://doi.org/10.1016/j.tca.2011.03.034

    Parameters
    ----------
    temperature : array-like
        Sample temperatures in Kelvin.
    heating_rate : array-like
        Heating rates in Kelvin per second.
    B : float
        Exponent for the temperature term. Default is 1.92
        (Starink improvement).
    C : float
        Coefficient for activation energy expression. Default is 1.0008
        (Starink improvement).

    Returns
    -------
    tuple
        A tuple containing:

        - slope (float): Slope of the linear fit.
        - intercept (float): Intercept of the linear fit.
        - Ea_i (float): Apparent activation energy in J/mol.
        - fit_points (tuple): Data points (x, y) used for the linear fit.
    """

    # Ensure numpy arrays
    temperature = series_to_numpy(temperature)
    heating_rate = series_to_numpy(heating_rate)

    # Input validation
    if len(temperature) != len(heating_rate):
        raise ValueError("temperature and heating_rate must have the same length.")
    if np.any(temperature <= 0) or np.any(heating_rate <= 0):
        raise ValueError("temperature and heating_rate must be positive.")
    if np.any(np.power(temperature, B) <= 0):
        raise ValueError("Temperature raised to exponent B must be positive.")

    # Prepare x and y data for the linear fit
    data_x = 1/temperature
    data_y = np.log(heating_rate / np.power(temperature, B))
    fit_points = (data_x, data_y)

    # Perform the linear fit
    popt, pcov = curve_fit(linear_model,
                           data_x, data_y,
                           maxfev=10000)

    # Extract the fitted parameters
    m_fit, b_fit = popt

    # Calculate estimate of (Ea_i), in J/mol.
    Ea_i = -(m_fit * GAS_CONSTANT) / C

    # TODO: consider to use namedtuple or dataclass
    return popt, Ea_i, fit_points


def compute_Ea_KAS(database, data_keys=["experiments", "TGA", "constant_heating_rate"], **kwargs):
    """
    Wrapper function to easily compute activation energies using the Kissinger–Akahira–Sunose (KAS) method.

    This function applies the KAS method to interpolated conversion data
    (as prepared by :func:`compute_conversion_levels`) and estimates the
    activation energy (:math:`E_a`) at each conversion level. It computes
    linear regression statistics, including RMSE and R², and stores the results
    in a DataFrame.

    The final DataFrame is stored in the database at the specified location
    defined by `data_keys`.

    This function requires that :func:`compute_conversion_levels` has been
    called beforehand to prepare the input data.

    Parameters
    ----------
    database : dict
        The main data structure storing all experimental data.
        Must follow the format initialized by
        :func:`initialize_investigation_skeleton`.
    data_keys: list
        Keys that define the path to the relevant dataset inside the database.
        For example: ["experiments", "TGA", "constant_heating_rate"].
    **kwargs
        Additional arguments passed to :func:`KAS_Ea`.
        For example, to override the default Starink constants (B, C).

    Returns
    -------
    None
        Adds a DataFrame named ``Ea_results_KAS`` to the corresponding location
        in the database. The DataFrame contains:

        - ``Conversion``: Conversion level α
        - ``Ea``: Activation energy in J/mol
        - ``m_fit``: Slope of the linear fit
        - ``b_fit``: Intercept of the linear fit
        - ``R_squared``: Coefficient of determination
        - ``RMSE``: Root mean square error
        - ``fit_points``: Data points used in the linear fit
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
    set_value_keys = sorted(dataset.keys(), key=lambda x: dataset[x]["set_value"]["value"])

    # Sort the set values themselves
    set_values = sorted(dataset[key]["set_value"]["value"] for key in dataset.keys())

    # Get number of conversion levels
    conversion_levels = dataset[set_value_keys[0]]["conversion_fractions"]["Alpha"]

    # Prepare data collection.
    Ea = list()
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
        popt, Ea_i, fit_points = KAS_Ea(conversion_temperatures, set_values, **kwargs)
        Ea.append(Ea_i)

        # Extract and store the fitted parameters.
        m_fit, b_fit = popt
        m.append(m_fit)
        b.append(b_fit)

        # Generate y-values from the fitted model.
        data_x, data_y = fit_points
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
         "m_fit": np.array(m),
         "b_fit": np.array(b),
         "R_squared": np.array(r_squared),
         "RMSE": np.array(rmse),
         **xy_data})

    # Collect results.
    store_Ea["Ea_results_KAS"] = Ea_results


def exp_difference(offset, temp_x1, temp_x2, data_y1, data_y2):
    """
    Compute the RMSE between two data series after applying a shift
    to the second x-axis.

    The function shifts `temp_x2` by a given `offset`, interpolates the
    corresponding `data_y2` onto the `temp_x1` grid using linear interpolation,
    and computes the root mean square error (RMSE) between `data_y1` and the
    interpolated `data_y2`.

    This is useful for aligning two experimental or simulated curves when a
    shift in the x-dimension (e.g., time or temperature) is expected.

    Linear interpolation is done via `scipy.interpolate.interp1d` with
    extrapolation enabled outside the original `temp_x2` range.

    Parameters
    ----------
    offset : float
        Horizontal shift applied to `temp_x2` before interpolation.
    temp_x1 : array-like
        x-values of the alignment target (evaluation grid).
    temp_x2 : array-like
        x-values of the data series to be shifted.
    data_y1 : array-like
        y-values corresponding to `temp_x1`.
    data_y2 : array-like
        y-values corresponding to `temp_x2`.

    Returns
    -------
    float
        Root mean square error (RMSE) between `data_y1` and the
        interpolated, shifted version of `data_y2`.
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
    Find the optimal horizontal shift that aligns two data series by
    minimizing RMSE.

    This function optimizes the x-axis shift applied to `temp_x2` so that the
    interpolated `data_y2` best matches `data_y1`, evaluated over `temp_x1`.
    The objective function is root mean square error (RMSE), minimized using
    `scipy.optimize.minimize`.

    Parameters
    ----------
    initial_guess : float
        Initial guess for the x-axis shift.
    temp_x1 : array-like
        x-values of the alignment target (evaluation grid).
    temp_x2 : array-like
        x-values of the data series to be shifted.
    data_y1 : array-like
        y-values corresponding to `temp_x1`.
    data_y2 : array-like
        y-values corresponding to `temp_x2`.
    method : str, optional
        Optimization method passed to `scipy.optimize.minimize`.
        Default is "Powell" for robustness against local minima.

    Returns
    -------
    float
        The optimal shift that minimizes RMSE between the two series.
    """

    # Optimize temperature offset
    result = minimize(fun=exp_difference, x0=[initial_guess],
                      args=(temp_x1, temp_x2, data_y1, data_y2),
                      method=method)

    # Get optimal shift
    optimal_shift = result.x[0]

    return optimal_shift
