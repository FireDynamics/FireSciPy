import numpy as np
import pandas as pd


def conversion_integral_data(res_mass_norm):
    """
    Calculate conversion for normalised integral data, e.g. TGA residual mass.

    :param res_mass_norm: NumPy array of normalised residual mass

    :return: NumPy array of the conversion from 0 to 1
    """

    alpha = 1 - res_mass_norm

    return alpha


def get_conversion_idx(alpha_i, alpha):
    """
    Gets the index of the closest conversion value
    from a NumPy array or Pandas DataSeries.

    :param alpha_i: specific value of conversion
    :param alpha: NumPy array, or Pandas DataSeries,
                  of the conversion from 0 to 1

    :return idx_closest: index of the closest conversion value
    """

    lower_idx = max(np.where(alpha < alpha_i)[0])
    upper_idx = min(np.where(alpha > alpha_i)[0])

    if type(alpha) == np.ndarray:
        lower_val = alpha[lower_idx]
        upper_val = alpha[upper_idx]
    elif type(alpha) == pd.core.series.Series:
        lower_val = alpha.iloc[lower_idx]
        upper_val = alpha.iloc[upper_idx]

    dist_lower = np.abs(alpha_i - lower_val)
    dist_upper = np.abs(alpha_i - upper_val)

    if dist_lower < dist_upper:
        idx_closest = lower_idx
    else:
        idx_closest = upper_idx

    return idx_closest


def conversion_info(alphas, hr_labels, fractions, exp_times, exp_temps):
    """
    For desired fractions of conversion, temperatures and time steps are collected
    and organised per heating rate.
    This is a pre-processing step to later determine the Arrhenius parameters.
    The information is collected in a dictionary to simplify further use.

    :param alphas: list of conversions per heating rate
    :param hr_labels: list with heating rate labels
    :param fractions: NumPy array of the desired fraction intervals
    :param exp_times: list of experiment times per heating rate
    :param exp_temps: list of sample temperatures per heating rate

    :return conv_data: dictionary with conversion data sorted by heating rate
    """

    # Collect target conversion fractions globally.
    conv_data = {
        "Conversion_fractions": fractions,
        "Conversion_combined": dict()
    }

    # Convenience.
    conv_comb = conv_data["Conversion_combined"]

    # Go over all heating rates.
    for hr_id, hr_label in enumerate(hr_labels):
        # Prepare data sets.
        alpha = alphas[hr_id].to_numpy()
        time = exp_times[hr_id].to_numpy()
        temp = exp_temps[hr_id].to_numpy()

        # Get indices closest to desired conversion fraction.
        alpha_indices = list()
        for fraction in fractions:
            idx = fsp.get_conversion_idx(fraction, alpha)
            alpha_indices.append(idx)

        # Get data points associated with the above indices.
        conv_times = time[alpha_indices]
        conv_temps = temp[alpha_indices]
        conv_fracs = alpha[alpha_indices]

        # Combine times, temperatures and fractions.
        combined = [list(line) for line in zip(conv_times, conv_temps, conv_fracs)]
        ndf = pd.DataFrame(data=np.array(combined),
                           columns=["Time", "Temperature", "Alpha"])
        # Collect result.
        conv_comb[hr_label] = ndf

    return conv_data


def get_activation_energy(conv_data):
    """
    Fits a linear function across heating rates for each conversion fraction.
    From the inclination and the gas constant R, the activation energy E is determined.
    Some statistics on the fits is performed. Both, E and statistics
    are stored in the input dictionary.

    :param conv_data: dictionary with conversion data sorted by heating rate

    :return: results are stored in the conv_data dictionary
    """

    # Initialise data collection.
    conv_data["Parameters"] = list()
    conv_data["Residuals"] = list()
    conv_data["R_squared"] = list()
    conv_data["RMSE"] = list()
    conv_data["E_alpha"] = list()

    # Go over all fractions to compute respective value for E.
    for frac_id in range(len(conv_data["Conversion_fractions"])):
        t_combined = list()
        T_combined = list()

        # Go over all heating rates.
        hr_labels = list(conv_data["Conversion_combined"])
        for hr_id, hr_label in enumerate(hr_labels):
            # Collect data points per fraction.
            toast = conv_data["Conversion_combined"][hr_label]
            t_combined.append(toast["Time"].iloc[frac_id])
            T_combined.append(toast["Temperature"].iloc[frac_id])

        # Prepare data points for fitting.
        data_x = 1 / (np.array(T_combined) + 273.15)
        data_y = np.log(1/np.array(t_combined))

        # Define the model function.
        def model_func(x, m, b):
            return m * x + b

        # Perform the curve fitting using the original y_values
        popt, pcov = curve_fit(model_func, data_x, data_y,
                               maxfev=10000)
        conv_data["Parameters"].append(popt)

        # Extract the fitted parameters
        m_fit, b_fit = popt
        # print(m_fit, b_fit)

        # Generate y-values from the fitted model
        y_fit = model_func(data_x, m_fit, b_fit)

        # Calculate residuals
        residuals = data_y - y_fit
        conv_data["Residuals"].append(residuals)

        # Calculate R-squared
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((data_y - np.mean(data_y))**2)
        r_squared = 1 - (ss_res / ss_tot)
        conv_data["R_squared"].append(r_squared)

        # Calculate RMSE
        rmse = np.sqrt(np.mean(residuals**2))
        conv_data["RMSE"].append(rmse)

        # Calculate E_alpha_i, in kJ/mol.
        E_alpha_i = -(m_fit * gas_const) / 1000
        conv_data["E_alpha"].append(E_alpha_i)


# ICTAC Kinetics Committee recommendations for performing kinetic computations on thermal analysis data
# Sergey Vyazovkin et al., 2011
# doi:10.1016/j.tca.2011.03.034
# Table 1: Some of the kinetic models used in the solid-state kinetics.

# Reaction models from Table 1.
# Use NumPy functions, like np.exp(), to maintain np.array() compatibility.
reaction_models = {
    "P4": {
        "Reaction Model": "Power law",
        "f_alpha": lambda alpha: 4 * alpha ** (3/4),
        "g_alpha": lambda alpha: alpha ** (1/4)
    },
    "P3": {
        "Reaction Model": "Power law",
        "f_alpha": lambda alpha: 3 * alpha ** (2/3),
        "g_alpha": lambda alpha: alpha ** (1/3)
    },
    "P2": {
        "Reaction Model": "Power law",
        "f_alpha": lambda alpha: 2 * alpha ** (1/2),
        "g_alpha": lambda alpha: alpha ** (1/2)
    },
    "P2/3": {
        "Reaction Model": "Power law",
        "f_alpha": lambda alpha: 2/3 * alpha ** (-1/2),
        "g_alpha": lambda alpha: alpha ** (3/2)
    },
    "D1": {
        "Reaction Model": "One-dimensional diffusion",
        "f_alpha": lambda alpha: 1/2 * alpha ** (-1),
        "g_alpha": lambda alpha: alpha ** 2
    },
    "F1": {
        "Reaction Model": "Mampel (first order)",
        "f_alpha": lambda alpha: 1 - alpha,
        "g_alpha": lambda alpha: -np.log(1 - alpha)
    },
    "A4": {
        "Reaction Model": "Avrami-Erofeev",
        "f_alpha": lambda alpha: 4 * (1 - alpha) * (-np.log(1 - alpha)) ** (3/4),
        "g_alpha": lambda alpha: (-np.log(1 - alpha)) ** (1/4)
    },
    "A3": {
        "Reaction Model": "Avrami-Erofeev",
        "f_alpha": lambda alpha: 3 * (1 - alpha) * (-np.log(1 - alpha)) ** (2/3),
        "g_alpha": lambda alpha: (-np.log(1 - alpha)) ** (1/3)
    },
    "A2": {
        "Reaction Model": "Avrami-Erofeev",
        "f_alpha": lambda alpha: 2 * (1 - alpha) * (-np.log(1 - alpha)) ** (1/2),
        "g_alpha": lambda alpha: (-np.log(1 - alpha)) ** (1/2)
    },
    "D3": {
        "Reaction Model": "Three-dimensional diffusion",
        "f_alpha": lambda alpha: 3/2 * (1 - alpha) ** (2/3) * (1 - (1 - alpha) ** (1/3)) ** (-1),
        "g_alpha": lambda alpha: (1 - (1 - alpha) ** (1/3)) ** (2)
    },
    "R3": {
        "Reaction Model": "Contracting sphere",
        "f_alpha": lambda alpha: 3 * (1 - alpha) ** (2/3),
        "g_alpha": lambda alpha: 1 - (1 - alpha) ** (1/3)
    },
    "R2": {
        "Reaction Model": "Contracting cylinder",
        "f_alpha": lambda alpha: 2 * (1 - alpha) ** (1/2),
        "g_alpha": lambda alpha: 1 - (1 - alpha) ** (1/2)
    },
    "D2": {
        "Reaction Model": "Two-dimensional diffusion",
        "f_alpha": lambda alpha: (-np.log(1 - alpha)) ** (-1),
        "g_alpha": lambda alpha: (1 - alpha) * np.log(1 - alpha) + alpha
    },
}
