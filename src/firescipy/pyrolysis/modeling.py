# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.


import numpy as np
import pandas as pd

from scipy.integrate import solve_ivp

from firescipy.constants import GAS_CONSTANT


def create_linear_temp_program(start_temp=300, end_temp=700, beta=10.0, beta_unit="K/min", steps=400):
    """
    Create a linear temperature-time program for pyrolysis modeling.

    The function generates a temperature ramp from `start_temp` to `end_temp`
    at a constant heating rate (`beta`). The time values are calculated
    accordingly based on the unit of the heating rate.

    Cooling programs (where `end_temp` < `start_temp`) are not supported.

    Parameters
    ----------
    start_temp : float
        Starting temperature in Kelvin. Default is 300 K.
    end_temp : float
        Ending temperature in Kelvin. Default is 700 K.
    beta : float
        Heating rate in specified units (float). Default is 10.0.
    beta_unit : str
        Unit of the heating rate, either "K/min" or "K/s". Default is "K/min".
    steps : int
        Number of steps for the time-temperature array. Default is 400.

    Returns
    -------
    dict
        Dictionary containing:
        - "Time": array of time values (in seconds)
        - "Temperature": array of temperature values (in Kelvin).
    """

    # Convert the heating rate into Kelvin per second.
    if beta_unit == "K/min":
        beta_s = beta / 60
    elif beta_unit == "K/s":
        beta_s = beta
    else:
        raise ValueError("Provide beta_units, and the respective beta, either in 'K/min' or 'K/s'.")

    # Check if temperature increases (no cooling)
    if start_temp >= end_temp:
        raise ValueError("start_temp must be less than end_temp.")
    # Check if heating rate is positive (no cooling)
    if beta <= 0:
        raise ValueError("beta must be greater than zero.")

    # Determine total time required for the heating process
    delta_temp = end_temp - start_temp
    heating_time = delta_temp / beta_s

    # Compute time and temperature arrays
    time = np.linspace(0, heating_time, steps)
    temperature = start_temp + beta_s * time

    # Create Pandas DataFrame.
    temp_program = {"Time": time, "Temperature": temperature}

    # Provide results
    return temp_program


def reaction_rate(t, alpha, t_array, T_array, A, E, R=GAS_CONSTANT,
                   reaction_model='nth_order', model_params=None):
    """
    Compute the reaction rate :math:`\\frac{d\\alpha}{dt}` using Arrhenius
    kinetics and a reaction model.

    The reaction rate is computed using the Arrhenius expression for the
    rate constant :math:`k(T) = A \\exp(-E / RT)` and a chosen reaction model
    :math:`f(\\alpha)`. Temperature :math:`T(t)` is obtained by interpolating
    `T_array` at the current time `t`.

    .. math::

        \\frac{d\\alpha}{dt} = A \\exp\\left(-\\frac{E}{RT(t)}\\right) f(\\alpha)

    Formula 1.1 in: Vyazovkin et al. (2011). ICTAC Kinetics Committee
    recommendations for performing kinetic computations on thermal analysis data
    *Thermochimica Acta*, 520(1–2), 1–19.
    https://doi.org/10.1016/j.tca.2011.03.034

    Parameters
    ----------
    t : float
        Current time (in seconds).
    alpha : float
        Current conversion fraction (dimensionless, between 0 and 1).
    t_array : array-like
        Time points corresponding to the known temperatures.
    T_array : array-like
        Temperatures (in Kelvin) corresponding to `t_array`.
    A : float
        Pre-exponential factor (in 1/s).
    E : float
        Activation energy (in J/mol).
    R : float
        Gas constant (in J/mol·K). Default is `GAS_CONSTANT`.
    reaction_model : str
        Key identifying the reaction model to use (e.g., `'nth_order'`).
        Must correspond to a model in the internal `f_models` dictionary.
    model_params : dict
        Additional parameters required by the reaction model
        (e.g., `{'n': 2.0}` for an nth-order reaction).

    Returns
    -------
    float
        The reaction rate :math:`\\frac{d\\alpha}{dt}` at the given time `t`.
    """

    if model_params is None:
        model_params = {}

    # Interpolate temperature at current time
    T_current = np.interp(t, t_array, T_array)

    # Reaction rate constant (Arrhenius factor)
    k_T = A * np.exp(-E / (R * T_current))

    # Fetch the f(alpha) function from the dictionary
    # f_alpha = f_models[reaction_model]
    f_alpha = get_reaction_model(reaction_model)

    # Check if generic or named model
    if reaction_model in f_models:
        # Evaluate f(alpha) with any extra model parameters
        val_f_alpha = f_alpha(alpha, model_params)
    elif reaction_model in named_models:
        # Evaluate f(alpha) without extra model parameters
        val_f_alpha = f_alpha(alpha)
    else:
        raise ValueError(f"Reaction model '{reaction_model}' not found.")

    return k_T * val_f_alpha


def solve_kinetics(t_array, T_array, A, E, alpha0=1e-12, R=GAS_CONSTANT,
                   reaction_model='nth_order', model_params=None):
    """
    Numerically solve the conversion ODE :math:`\\frac{d\\alpha}{dt}` for a
    given temperature program.

    This function solves the reaction kinetics ODE using Arrhenius temperature
    dependence and a specified reaction model :math:`f(\\alpha)`. The
    temperature is interpolated over the given `t_array`, and the solution
    is returned at the same time points.

    The underlying ODE is defined as:

    .. math::

        \\frac{d\\alpha}{dt} = A \\exp\\left(-\\frac{E}{RT(t)}\\right) f(\\alpha)

    Parameters
    ----------
    t_array : array-like
        Time values (in seconds) at which the temperature profile is defined and
        where the solution will be evaluated.
    T_array : array-like
        Corresponding temperatures (in Kelvin) at each point in `t_array`.
    A : float
        Pre-exponential factor (in 1/s).
    E : float
        Activation energy (in J/mol).
    alpha0 : float
        Initial conversion level. Defaults to 1e-12 to avoid numerical issues
        with reaction models that are undefined at :math:`\\alpha = 0`,
        such as the nth-order model.
    R : float
        Gas constant (in J/mol·K). Default is `GAS_CONSTANT`.
    reaction_model : str
        Key identifying the reaction model to use (e.g., `'nth_order'`).
        Must correspond to a model in the internal `f_models` dictionary.
    model_params : dict
        Additional parameters required by the reaction model
        (e.g., `{'n': 2.0}` for an nth-order reaction).

    Returns
    -------
    t : ndarray
        Time points (in seconds) at which the solution is evaluated.
    alpha : ndarray
        Computed conversion values :math:`\\alpha(t)` at each time point.
    """

    if model_params is None:
        model_params = {}

    # ODE wrapper for solve_ivp
    def ode_wrapper(t, alpha):
        return reaction_rate(t, alpha[0], t_array, T_array,
                             A, E, R, reaction_model, model_params)

    # Solve from t=0 to t=t_array[-1]
    sol = solve_ivp(
        ode_wrapper,
        (t_array[0], t_array[-1]),
        [alpha0],        # initial condition
        t_eval=t_array,rtol=1e-8, atol=1e-10
    )

    return sol.t, sol.y[0]


def get_reaction_model(model_name):
    """
    Retrieve a reaction model function by name.

    This function looks up a callable corresponding to a known reaction
    model name. The models can be either parameter-free (`named_models`)
    or parameterized (`f_models`), such as nth-order or Avrami-Erofeev models.

    Parameters
    ----------
    model_name : str
        Name of the reaction model to retrieve. Must be a key in either
        `named_models` or `f_models`.

    Returns
    -------
    callable
        A function f(alpha) or f(alpha, params) that computes the
        reaction model value for a given conversion alpha.

    Raises
    ------
    ValueError
        If the model name is not found in any registered model dictionaries.
    """

    if model_name in named_models:
        return named_models[model_name]
    elif model_name in f_models:
        return f_models[model_name]
    else:
        raise ValueError(f"Model '{model_name}' not found in reaction models.")


# Define default clipping thresholds
default_clip_values = {
    "alpha_min": 1e-12,  # Avoids log(0) or division by zero
    "alpha_max": 1       # Ensures alpha does not exceed 1
}

# Define function to enable the user to adjust the clipping of alpha
def clip_alpha(alpha, clip_values):
    """
    Ensure alpha remains within numerical stability range.
    Specifically, to deal with floating-point errors. They may
    slightly push numbers outside of the expected range of alpha=[0,1].
    This can lead to numerical errors when computations involve
    logarithms or power laws.
    """
    return np.clip(alpha, clip_values["alpha_min"], clip_values["alpha_max"])


# A dictionary of generic reaction models f(α) that take extra parameters:
f_models = {
    # Formula (1.9); https://doi.org/10.1016/j.tca.2011.03.034
    'nth_order': lambda alpha, params, clip_values=default_clip_values:
        (1 - clip_alpha(alpha, clip_values))**params['n'],

    # Formula (1.8); https://doi.org/10.1016/j.tca.2011.03.034
    'power_law': lambda alpha, params, clip_values=default_clip_values:
        params['n'] * np.power(clip_alpha(alpha, clip_values), ((params['n']-1)/params['n'])),

    # Formula (1.10); https://doi.org/10.1016/j.tca.2011.03.034
    'Avrami_Erofeev': lambda alpha, params, clip_values=default_clip_values:
        params['n'] * (1 - clip_alpha(alpha, clip_values)) * (-np.log(clip_alpha(1 - alpha, clip_values)))**((params['n']-1)/params['n']),

    # Formulas (1.3) and (1.4); https://doi.org/10.1016/j.tca.2022.179384
    'Sestak_Berggren': lambda alpha, params, clip_values=default_clip_values:
        params['c'] * clip_alpha(alpha, clip_values)**params['m'] * (1 - clip_alpha(alpha, clip_values))**params['n'] * (-np.log(clip_alpha(1 - alpha, clip_values)))**params['p'],
    # Add more named models as needed
}


# Dictionary of named models with fixed parameters
named_models = {
    'D3': lambda alpha, clip_values=default_clip_values: (3/2) * (1 - clip_alpha(alpha, clip_values))**(2/3) * (1 - (1 - clip_alpha(alpha, clip_values))**(1/3))**(-1),
    'A2': lambda alpha, clip_values=default_clip_values: 2 * (1 - clip_alpha(alpha, clip_values)) * (-np.log(1 - clip_alpha(alpha, clip_values))) ** (1/2)
    # Add more named models as needed
}


# ICTAC Kinetics Committee recommendations for performing kinetic computations on thermal analysis data
# Sergey Vyazovkin et al., 2011
# https://doi.org/10.1016/j.tca.2011.03.034
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
