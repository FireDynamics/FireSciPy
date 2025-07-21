Simple Kinetics Computation Mock-Up
===================================

Theoretical Background
----------------------

.. math::
    :label: conv_rate_short

    \frac{d \alpha}{dt} = k(T) ~f(\alpha)



More details are available in the recommendations provided by the International Confederation for Thermal Analysis and Calorimetry (ICTAC) Kinetics Committee:

- `ICTAC Kinetics Committee recommendations for performing kinetic computations on thermal analysis data <https://doi.org/10.1016/j.tca.2011.03.034>`_
- `ICTAC Kinetics Committee recommendations for collecting experimental thermal analysis data for kinetic computations <https://doi.org/10.1016/j.tca.2014.05.036>`_
- `ICTAC Kinetics Committee recommendations for analysis of multi-step kinetics <https://doi.org/10.1016/j.tca.2020.178597>`_
- `ICTAC Kinetics Committee recommendations for analysis of thermal decomposition kinetics <https://doi.org/10.1016/j.tca.2022.179384>`_


Isoconversional Computation with the Kissinger-Akaira-Sunose (KAS) Method
-------------------------------------------------------------------------

FireSciPy provides functionalities to conduct reaction kinetics computations. In this example, the activation energy :math:`E` is determined, using the Kissinger-Akaira-Sunose (KAS) method.



.. code-block:: python

    # Import necessary packages
    import matplotlib.pyplot as plt
    import firescipy as fsp

    # Reaction constants
    A = 10**10 / 60   # 1/s
    E = 125.4 * 1000  # J/mol
    alpha0 = 1e-12    # Avoid exactly zero to prevent numerical issues (e.g., division by zero)

The following parameters provide the basic settings for the temperature programs.

.. code-block:: python

    # Define temperature program (model recording frequency ΔT during TGA experiment)
    n_points = 2 * 450 + 1  # ΔT = 0.5 K

    # Temperatures in Kelvin
    start_temp = 300
    end_temp = 750

Multiple heating rates are defined to create the input for the isoconversional analysis. In this example heating rates :math:`\beta` of :math:`8~K/min`, :math:`12~K/min` and :math:`16~K/min` are used. For conveniance, the heating rate values are stored inside a dictionary.

.. code-block:: python

    # Define heating rates
    heating_rates = {
        "8Kmin": 8,
        "12Kmin": 12,
        "16Kmin": 16
    }

The different temperature-time series are stored inside their own dictionary for easy access later on.

.. code-block:: python

    # Initialise data collection
    reaction_rates = dict()



.. code-block:: python

    # Create decomposition example data
    for heating_rate in heating_rates:
        # Define model temperature program
        beta = heating_rates[heating_rate]
        temp_program = fsp.pyrolysis.modeling.create_linear_temp_program(
            start_temp=start_temp,
            end_temp=end_temp,
            beta=beta,
            beta_unit="K/min",
            steps=n_points)

        # Get time-temperature data series
        t_array = temp_program["Time"]
        T_array = temp_program["Temperature"]

        # Collect temperature data
        decomposition = fsp.utils.ensure_nested_dict(
            reaction_rates,
            [heating_rate, "FireSciPyExample"])

        # Store time-temperature data series
        decomposition["Time"] = t_array
        decomposition["Temperature_Avg"] = T_array

        # Compute reaction rate
        t_sol, alpha_sol = fsp.pyrolysis.modeling.solve_kinetics(
           t_array, T_array, alpha0, A, E,
           reaction_model='nth_order',
           model_params={'n': 1.0}
        )

        # Store conversion
        decomposition["Alpha"] = alpha_sol

        # Compute normalised mass, assuming no residue
        decomposition["Mass_Avg"] = 1 - alpha_sol
