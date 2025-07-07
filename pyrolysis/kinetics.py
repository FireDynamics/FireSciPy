import numpy as np
import pandas as pd

from scipy.interpolate import interp1d
from scipy.optimize import curve_fit, minimize
from typing import List, Dict, Union  # for type hints in functions
from FireSciPy.utils import series_to_numpy, ensure_nested_dict, get_nested_value, linear_model, calculate_residuals, calculate_R_squared, calculate_RMSE
from FireSciPy.constants import GAS_CONSTANT



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

    Parameters
    ----------
    integral_data : pd.Series or np.ndarray
        Experimental data representing integral quantities
        (e.g., mass over time) to calculate the conversion.
    m_0 : float, optional
        Initial mass/concentration. Defaults to the first
        value of `integral_data`.
    m_f : float, optional
        Final mass/concentration. Defaults to the last
        value of `integral_data`.

    Returns
    -------
    np.ndarray
        Array of alpha values representing the conversion as a
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

    Parameters
    ----------
    database (dict): The main data structure storing all experimental data.
    condition (str or list, optional): The specific experimental condition(s) to process.
        - If a string is provided, it can be a single condition (e.g., "300_C"), or "all" to process all conditions.
        - If a list is provided, it should contain multiple condition names.
    setup (str, optional): The experimental setup to process, either "isothermal" or "constant_heating_rate".
        Defaults to "constant_heating_rate".
    # m_0 (float, optional): Initial sample mass. If None, use the first mass value from the data.
    # m_f (float, optional): Final sample mass. If None, use the last mass value from the data.

    Returns
    -------
    None
        Adds conversion data directly into the database under each condition.
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


def KAS_Ea(temperature, heating_rate, B=1.92, C=1.0008):
    """
    Kissinger–Akahira–Sunose method (KAS), with Starink improvement by default.
    Estimates the activation energy (E_a) for a given
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
        B (float): Exponent for temperature (default: 1.92 for Starink improvement).
        C (float): Coefficient for activation energy calculation (default: 1.0008 for Starink improvement).

    :return: list, containing:
        parameters of the linear fit,
        activation energy for specified level of conversion (Ea_i) in J/mol,
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
    if np.any(np.power(temperature, B) <= 0):
        raise ValueError("Temperature raised to exponent B must be positive.")

    # Prepare x and y data for the linear fit
    data_x = 1/temperature
    data_y = np.log(heating_rate / np.power(temperature, B))

    # Perform the linear fit
    popt, pcov = curve_fit(linear_model,
                           data_x, data_y,
                           maxfev=10000)

    # Extract the fitted parameters
    m_fit, b_fit = popt

    # Calculate estimate of (Ea_i), in J/mol.
    Ea_i = -(m_fit * GAS_CONSTANT) / C

    return [popt, Ea_i, [data_x, data_y]]


def compute_Ea_KAS(database, data_keys=["experiments", "TGA", "constant_heating_rate"], **kwargs):
    """
    Wrapper function to easily compute activation energies using the Kissinger–Akahira–Sunose method.

    Parameters
    ----------
        database: dict
            Nested dictionary containing the dataset.
        data_keys: list
            List of keys to locate the dataset within the database.
        **kwargs: Additional arguments to pass to the KAS_Ea function.

    Returns
    -------
        None
            Stores the results in the parent dictionary of the specified dataset.
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
        popt, Ea_i, data_xy = KAS_Ea(conversion_temperatures, set_values, **kwargs)
        Ea.append(Ea_i)

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
         "m_fit": np.array(m),
         "b_fit": np.array(b),
         "R_squared": np.array(r_squared),
         "RMSE": np.array(rmse),
         **xy_data})

    # Collect results.
    store_Ea["Ea_results_KAS"] = Ea_results


def exp_difference(offset, temp_x1, temp_x2, data_y1, data_y2):
    """
    Computes the difference between two data series by means of root mean square error (RMSE).
    An offset is provided such that the difference can be minimised.

    Parameters
    ----------
    offset : float
        Value to shift the second data series in x.
    temp_x1 : numpy.ndarray
        x-values of the first data series.
    temp_x2 : numpy.ndarray
        x-values of the second data series.
    data_y1 : numpy.ndarray
        y-values of the first data series.
    data_y2 : numpy.ndarray
        y-values of the second data series.

    Returns
    -------
    float
        The RMSE value.
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

    Parameters
    ----------
    initial_guess: float
        the initial guess value
    temp_x1: numpy.ndarray
        x-values of first data series
    temp_x2: numpy.ndarray
        x-values of second data series
    data_y1: numpy.ndarray
        y-values of first data series
    data_y2: numpy.ndarray
        y-values of second data series
    method: string
        method used by scipy.optimize.minimise, default here "Powell",
        trying to avoid getting stuck in local optima

    Returns
    -------
    float
        The optimal shift that leads to the smallest RMSE
    """
    # Optimize temperature offset
    result = minimize(fun=exp_difference, x0=[initial_guess],
                      args=(temp_x1, temp_x2, data_y1, data_y2),
                      method=method)

    # Get optimal shift
    optimal_shift = result.x[0]

    return optimal_shift
