import numpy as np

from scipy.integrate import solve_ivp


def reaction_rate(t, alpha, t_array, T_array, A, E, R=gas_const,
                   reaction_model='nth_order', model_params=None):
    """
    Computes d(alpha)/dt at time t for a given alpha, using the
    reaction rate constant k(T) (Arrhenius factor) and a chosen
    reaction model f(alpha).
    See formula (1.1) in [1].

    Sergey Vyazovkin et al.; 10 June 2011
    ICTAC Kinetics Committee recommendations for performing
    kinetic computations on thermal analysis data
    Thermochimica Acta, Volume 520, Issues 1–2, Pages 1-19
    https://doi.org/10.1016/j.tca.2011.03.034

    Parameters
    ----------
    t : float
        Current time
    alpha : float
        Current conversion fraction
    t_array : array-like
        Times at which T_array is known
    T_array : array-like
        Temperatures corresponding to t_array
    A : float
        Pre-exponential factor
    E : float
        Activation energy
    R : float
        Gas constant
    reaction_model : str
        Key from the f_models dictionary
    model_params : dict
        Extra parameters for the chosen reaction model (e.g. {'n': 2.0})

    Returns
    -------
    float
        The derivative d(alpha)/dt
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

def solve_kinetics(t_array, T_array, alpha0, A, E, R=gas_const,
                   reaction_model='nth_order', model_params=None):
    """
    Solve for alpha(t) over t_array using the reaction_rate ODE.

    Parameters
    ----------
    t_array : array-like
        Times at which to solve
    T_array : array-like
        Corresponding temperatures at those times
    alpha0 : float
        Initial conversion (e.g., 0)
    A : float
        Pre-exponential factor
    E : float
        Activation energy
    R : float
        Gas constant
    model : str
        Which f(alpha) model to use (key into f_models)
    model_params : dict
        Extra parameters for that model, e.g. {'n': 1.0}

    Returns
    -------
    sol.t : array
        The time grid of the solution
    sol.y[0] : array
        The computed alpha(t) at each time point
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
    Unified function to retrieve the reaction rate model.
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
