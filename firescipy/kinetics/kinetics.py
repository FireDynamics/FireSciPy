import numpy as np


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
