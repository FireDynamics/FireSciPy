from numpy import gradient


def inst_hr(time_s, temp_C):
    """
    Compute instantaneous heating rate for DSC, TGA, STA, MCC.

    :param time_s: list of time steps [s]
    :param temp_C: list of temperatures [K] or [°C]
    :return: instantaneous heating rate as numpy.ndarray [K/s]
    """

    hr = gradient(temp_C, time_s)
    return hr
