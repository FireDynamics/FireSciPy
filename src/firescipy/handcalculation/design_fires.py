import numpy as np


def alpha_t_squared(alpha, Q_max, num_points=20):
    """
    Compute the t-squared fire growth curve.

    This function returns the time-dependent heat release rate (HRR) for a
    t² fire growth model, up to the point where the maximum HRR `Q_max`
    is reached.
    It uses the standard alpha-t² formulation:

    .. math::

        Q(t) = \\alpha t^2

    The growth rate `alpha` can be passed either as a float (in kW/s²) or as a
    predefined growth classification string.

    Reference:
        Karlsson and Quintiere (2000), *Enclosure Fire Dynamics*, CRC Press LLC.

    Parameters
    ----------
    alpha : str or float
        Fire growth rate, in kW/s², or growth classification as a string.
        Accepted strings: "slow", "medium", "fast" and "ultra fast".
    Q_max : float
        Maximum heat release rate, in kW.
    num_points : int
        Number of time points for discretization. Default is 20.

    Returns
    -------
    t_growth : np.ndarray
        Time array in seconds, from 0 to the time at which Q_max is reached.
    Q_growth : np.ndarray
        Corresponding heat release rate array in kW.
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
    Compute the heat release rate (HRR) decay phase based on a selected model.

    This function generates the decay portion of a desigen fire curve following
    the peak heat release rate `Q_max`, using one of several common decay model
    formulations. The output is a time–HRR pair describing the decline in
    fire intensity.

    Supported decay models:
        - `"t_squared"` : A decreasing t² curve (Q ∝ (t_end − t)²).
        - `"mirrored"` : Reverses the corresponding t² growth curve.
        - `"linear"` : HRR linearly decreases from `Q_max` to 0 over time.
        - `"exponential"` : HRR decays exponentially with time.

    Optional keyword arguments (`**kwargs`) depending on the selected model:
        - `num_points` (int): Number of points to discretize the curve (default: 20).
        - `alpha_decay` (str or float): Decay rate used for `"t_squared"` model.
        - `alpha` (str or float): Growth rate to mirror (for `"mirrored"` model).
        - `t_end` (float): End time in seconds (for `"linear"` or `"exponential"` models).
        - `decay_constant` (float): Time constant for exponential decay.

    Parameters
    ----------
    Q_max : float
        Maximum heat release rate in kW at the start of the decay phase.
    decay_model : str
        Type of decay model.
        Options: "t_squared", "mirrored", "linear", "exponential".
    kwargs: Additional parameters for the decay model.

    Returns
    -------
    t_decay : np.ndarray
        Time array for the decay phase in seconds.
    Q_decay : np.ndarray
        Corresponding heat release rate array for the decay phase in kW.
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
    Generate a complete t-squared design fire with optional steady-state
    and flexible decay phases.

    This function constructs a fire heat release rate (HRR) curve consisting of:
      1. A **t-squared growth** phase to reach `Q_max`,
      2. An optional **steady-state** plateau at `Q_max` (duration computed automatically),
      3. A **decay** phase using one of several models.

    The total energy release (`Q_total`, in kJ) determines the duration of the steady-state phase
    such that the combined energy of all three phases matches the user input.

    Reference:
        Karlsson and Quintiere (2000), *Enclosure Fire Dynamics*, CRC Press LLC.

    Parameters
    ----------
    Q_max : float
        Maximum heat release rate [kW].
    Q_total : float
        Total desired energy release [kJ] over the fire curve.
    decay_model : str
        Decay model type.
        Options: "t_squared", "mirrored", "linear", "exponential".
    kwargs:  dict
        Additional parameters for growth and decay phases.
        Common options:

        - `alpha` (str or float): Growth rate (e.g., "slow", "medium", or numeric value in kW/s²).
        - `alpha_decay` (str or float): Decay rate for "t_squared" decay.
        - `t_end` (float): End time (in s) for linear or exponential decay.
        - `decay_constant` (float): Exponential time constant for "exponential" decay.
        - `num_points` (int) Number of points per phase (default: 20).

    Returns
    -------
    t_combined : np.ndarray
        Time array (in seconds) combining growth, steady, and decay phases.
    Q_combined : np.ndarray
        Corresponding heat release rate (in kW) at each time point.

    Raises
    ------
    ValueError
        If `Q_total` is too small to cover the energy of the growth and decay phases alone,
        leaving no energy for a steady phase.
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
