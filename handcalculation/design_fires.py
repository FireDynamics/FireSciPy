import numpy as np


def alpha_t_squared(alpha, Q_max, num_points=20):
    """
    Compute the t-squared fire growth curve.
    See for example "Enclosure Fire Dynamics" by Karlsson and Quintiere, 2000, CRC Press LLC.

    Parameters
    ----------
    alpha : str or float
        Fire growth rate [kW/s^2] or growth classification as a string.
        Accepted strings: "slow", "medium", "fast" and "ultra fast".
    Q_max : float
        Maximum heat release rate [kW].
    num_points : int
        Number of time points for discretization.

    Returns
    -------
    t_growth : numpy.ndarray
        Time array [s].
    Q_growth : numpy.ndarray
        Heat release rate array [kW].
    """

    # Predefined fire growth rates (in kW/s^2)
    alpha_mapping = {
        "slow": 0.00293,      # Slow growth
        "medium": 0.01172,    # Medium growth
        "fast": 0.04688,      # Fast growth
        "ultra fast": 0.1876  # Ultra fast growth
    }

    # Convert string to numerical alpha value if necessary
    if isinstance(alpha, str):
        alpha = alpha_mapping.get(alpha.lower())
        if alpha is None:
            raise ValueError(f"Invalid growth rate classification: '{alpha}'. Use 'slow', 'medium', 'fast', 'ultra fast' or float.")

    # Ensure alpha is a float
    if not isinstance(alpha, (float, int)):
        raise TypeError(f"Alpha must be a float or a valid string ('slow', 'medium', 'fast', or 'ultra fast'). Got: {type(alpha)}")

    # Calculate the t-squared growth curve
    t_max = np.sqrt(Q_max / alpha)
    t_growth = np.linspace(0, t_max, num_points)
    Q_growth = alpha * np.power(t_growth, 2)
    return t_growth, Q_growth


def compute_decay(Q_max, decay_model, **kwargs):
    """
    Compute the decay phase based on the selected decay model.

    Parameters
    ----------
    Q_max : float
        Maximum heat release rate [kW].
    decay_model : str
        Type of decay model.
        Options: "t_squared", "mirrored", "linear", "exponential".
    kwargs: Additional parameters for the decay model.

    Returns
    -------
    t_decay : numpy.ndarray
        Time array for the decay phase [s].
    Q_decay : numpy.ndarray
        Heat release rate array for the decay phase [kW].
    """

    if decay_model == "t_squared":
        # Use separate t_squared growth curve for decay
        num_points = kwargs.get("num_points", 20)
        alpha_decay = kwargs.get("alpha_decay", "slow")
        if alpha_decay is None:
            raise ValueError("The 'alpha_decay' parameter is required for the 't_squared' decay model.")
        t_growth, Q_growth = alpha_t_squared(alpha_decay, Q_max)
        t_decay = t_growth
        Q_decay = Q_growth[::-1]

    elif decay_model == "mirrored":
        # Use the mirrored t_squared growth curve for decay
        num_points = kwargs.get("num_points", 20)
        alpha = kwargs.get("alpha", "slow")
        t_growth, Q_growth = alpha_t_squared(alpha, Q_max)
        Q_decay = Q_growth[::-1]
        # Generate a forward-in-time decay time array
        t_decay = t_growth

    elif decay_model == "linear":
        # Use a linear model for decay
        t_end = kwargs.get("t_end", 100)  # Total decay time [s]
        t_decay = np.linspace(0, t_end, kwargs.get("num_points", 20))
        Q_decay = Q_max * (1 - t_decay / t_end)
        Q_decay[Q_decay < 0] = 0

    elif decay_model == "exponential":
        # Use an exponential model for decay
        t_decay = np.linspace(0, kwargs.get("t_end", 100), kwargs.get("num_points", 20))
        decay_constant = kwargs.get("decay_constant", 0.1)
        Q_decay = Q_max * np.exp(-decay_constant * t_decay)

    else:
        raise ValueError(f"Unsupported decay model: {decay_model}")

    return t_decay, Q_decay


def simple_design_fire(Q_max, Q_total, decay_model="t_squared", **kwargs):
    """
    Create a simple design fire based on t-squared growth and a flexible decay model.
    See for example "Enclosure Fire Dynamics" by Karlsson and Quintiere, 2000, CRC Press LLC.

    Parameters
    ----------
    Q_max : float
        Maximum heat release rate [kW].
    Q_total : float
        Total energy released by the fire [kJ].
    decay_model : str
        Decay model.
        Options: "t_squared", "mirrored", "linear", "exponential".
    kwargs: Additional parameters for growth and decay.

    Returns
    -------
    t_combined : numpy.ndarray
        Combined time array [s].
    Q_combined : numpy.ndarray
        Combined heat release rate array [kW].
    """
    # Step 0: Get the alpha for the growth and decay phase
    alpha = kwargs.get("alpha", "slow")
    if alpha is None:
        raise ValueError("The 'alpha' parameter is required for the 't_squared' growth model.")

    # Step 1: Compute fire growth phase
    t_growth, Q_growth = alpha_t_squared(alpha, Q_max)
    energy_growth = np.trapz(Q_growth, t_growth)  # Total energy during growth phase

    # Step 2: Compute fire decay phase
    t_decay, Q_decay = compute_decay(Q_max, decay_model, **kwargs)
    energy_decay = np.trapz(Q_decay, t_decay)  # Total energy during decay phase

    # Step 3: Compute steady-state phase
    energy_steady = Q_total - (energy_growth + energy_decay)
    if energy_steady < 0:
        raise ValueError("Total energy (Q_total) is insufficient to sustain the specified growth and decay phases.")
    t_steady = energy_steady / Q_max  # Duration of steady-state phase [s]

    # Step 4: Combine phases
    t_decay_shifted = t_decay + (t_growth[-1] + t_steady)  # Shift decay in time
    t_combined = np.concatenate((t_growth, t_growth[-1] + np.array([0, t_steady]), t_decay_shifted))
    Q_combined = np.concatenate((Q_growth, np.full(2, Q_max), Q_decay))

    return t_combined, Q_combined
