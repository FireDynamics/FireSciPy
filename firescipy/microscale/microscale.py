import numpy as np


def inst_hr(time_s, temp_C):
    """
    Compute instantaneous heating rate for DSC, TGA, STA, MCC.

    :param time_s: list of time steps [s]
    :param temp_C: list of temperatures [K] or [°C]
    :return: instantaneous heating rate as numpy.ndarray [K/s]
    """

    hr = np.gradient(temp_C, time_s)
    return hr


def H_baseline(m_0, dT_dt, c_T, m_t):
    """
    Compute sensible heat flow baseline, to later determine the specific heat capacities.

    From: Formula 11 in 'Measurement of kinetics and thermodynamics of the
    thermal degradation for non-charring polymers',
    Jing Li, Stanislav I.Stoliarov, Combustion and Flame 160 (2013) 1287–1297
    https://doi.org/10.1016/j.combustflame.2013.02.012
    https://doi.org/10.1016/j.polymdegradstab.2013.09.022

    :param m_0: initial sample mass [mg]
    :param dT_dt: instantaneous heating rate [K/s]
    :param c_T: list of c_j [J / (kg K)]
    :param m_t: list of m_j [mg]
    :return: baseline as numpy.ndarray
    """

    c_m = list()

    for i, c_i_T in enumerate(c_T):
        c_m_i = c_i_T * m_t[i]
        c_m.append(c_m_i)

    if len(c_m) > 1:
        sum_c_m = np.sum(c_m, axis=0)
    else:
        sum_c_m = c_m[0]

    baseline = 1/m_0 * dT_dt * sum_c_m

    return baseline
