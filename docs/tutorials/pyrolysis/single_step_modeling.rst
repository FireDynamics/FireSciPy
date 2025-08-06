1. Single-Step Pyrolysis Kinetics Modelling
===========================================

In this section, the basic pyrolysis modelling capabilities of FireSciPy are introduced.

This example is also available as Jupyter Notebook.

Theoretical Background
----------------------

Pyrolysis kinetics describe the rates of thermally induced decomposition processes and are commonly studied using thermal analysis techniques. Common methods are thermogravimetric analysis (TGA), Differential Scanning Calorimetry (DSC) or Microscale Combustion Calorimetry (MCC), to name a few. The process rates can be described in terms of the temperature :math:`T`, the extent of conversion :math:`\alpha` and the pressure :math:`P`, see formula :eq:`conversion_rate`:

.. math::
    :label: conversion_rate

    \frac{d \alpha}{dt} = k(T) ~f(\alpha) ~h(P)

Where :math:`\frac{d \alpha}{dt}` is the conversion rate, :math:`k(T)` is the reaction rate constant, :math:`f(\alpha)` is the reaction model and :math:`h(P)` is the pressure dependence. The pressure dependence plays a role when gases can react with the decomposing sample material. During TGA experiments this can be minimized by maintaining an inert atmosphere and ensuring a sufficiently high purge gas flow to swiftly remove evolved gases which might otherwise react with the remaining sample material. Under these conditions the pressure term can than be dropped.

Typically, the reaction rate constant :math:`k(T)` is expressed as an Arrhenius equation, see formula :eq:`rate_const`:

.. math::
    :label: rate_const

    k(T) = A ~exp \left( -\frac{E}{R ~T}\right)

Where :math:`A` is the pre-exponential factor, :math:`E` the activation energy and :math:`R` the gas constant.

Substituting equation :eq:`rate_const` into :eq:`conversion_rate` and neglecting :math:`h(P)` leads to equation :eq:`conv_rate_Arrhenius`:

.. math::
    :label: conv_rate_Arrhenius

    \frac{d \alpha}{dt} = A ~exp \left( -\frac{E}{R ~T}\right) ~f(\alpha)

Together, :math:`A`, :math:`E` and :math:`f(\alpha)` form the kinetic triplet.



More details are available in the recommendations provided by the International Confederation for Thermal Analysis and Calorimetry (ICTAC) Kinetics Committee:

- `ICTAC Kinetics Committee recommendations for performing kinetic computations on thermal analysis data, 2011 (https://doi.org/10.1016/j.tca.2011.03.034) <https://doi.org/10.1016/j.tca.2011.03.034>`_
- `ICTAC Kinetics Committee recommendations for collecting experimental thermal analysis data for kinetic computations, 2014 (https://doi.org/10.1016/j.tca.2014.05.036) <https://doi.org/10.1016/j.tca.2014.05.036>`_
- `ICTAC Kinetics Committee recommendations for analysis of multi-step kinetics, 2020 (https://doi.org/10.1016/j.tca.2020.178597) <https://doi.org/10.1016/j.tca.2020.178597>`_
- `ICTAC Kinetics Committee recommendations for analysis of thermal decomposition kinetics, 2023 (https://doi.org/10.1016/j.tca.2022.179384) <https://doi.org/10.1016/j.tca.2022.179384>`_


Conversion Modelling in FireSciPy
---------------------------------

In this introduction, the conversion of a single-step pyrolysis reaction is modelled. As an example, a linear heating rate of :math:`\beta = 5~K/min` is chosen for the temperature program. The parameters are taken from `Vyazovkin, Advanced Isoconversional Method, 1997 <https://doi.org/10.1007/BF01983708>`_.


.. code-block:: python

    # Import necessary packages
    import numpy as np
    
    import matplotlib.pyplot as plt
    import firescipy as fsp

    # Reaction constants
    A = 10**10 / 60   # 1/s
    E = 125.4 * 1000  # J/mol
    alpha0 = 1e-12    # Avoid exactly zero to prevent numerical issues (e.g., division by zero)


At first, the linear temperature program needs to be created. Let's assume the sample temperature at the start of the experiment is at 300 K and it will end at a temperature of 750 K. The temperature is to change linearly over time according to the heating rate of :math:`\beta = 5~K/min`. Heating rates can be provided in either :math:`K/min` or :math:`K/s`, just pass the unit as a string, and FireSciPy will internally handle the unit conversion. The recording frequency of the TGA device is assumed to be such that a spacing of :math:`\Delta T = 0.5~K` is achieved. This spacing is a model of the frequency with which a TGA or similar device records the data.


.. code-block:: python

    # Define temperature program (model recording frequency ΔT during TGA experiment)
    n_points = 2 * 450 + 1  # ΔT = 0.5 K

    # Temperatures in Kelvin
    start_temp = 300
    end_temp = 750

    # Define model temperature program
    beta = 5  # K/min

    # Create the temperature program
    temp_program = fsp.pyrolysis.modeling.create_linear_temp_program(
        start_temp=start_temp,
        end_temp=end_temp,
        beta=beta,
        beta_unit="K/min",
        steps=n_points)


Once the temperature program is set up, the conversion can be computed. The kinetics solver needs to be provided with a couple of parameters: the temperature program (`t_array`, `T_array`), the initial conversion level (`alpha0`), the Arrhenius parameters (`A`, `E`) and the reaction model.

FireSciPy comes with a small library of different reaction models like `'nth_order'` or `'Avrami_Erofeev'`. The values for their parameters can be provided as a dictionary.
In this example, we assume a first-order reaction model :math:`f(\alpha) = (1 - \alpha)^n` with :math:`n = 1`, which is commonly used in thermal decomposition modeling due to its simplicity.
Thus, the nth-order model is chosen (`'nth_order'`) with the order set to unity `{'n': 1.0}`. The result are data series of how the conversion (`alpha_sol`) changes over time (`t_sol`).

See also: :func:`fsp.pyrolysis.modeling.create_linear_temp_program`, :func:`fsp.pyrolysis.modeling.solve_kinetics`


.. code-block:: python

    # Get time-temperature data series
    time_model = temp_program["Time"]
    temp_model = temp_program["Temperature"]

    # Compute conversion
    t_sol, alpha_sol = fsp.pyrolysis.modeling.solve_kinetics(
        t_array=time_model,
        T_array=temp_model,
        A=A,
        E=E,
        alpha0=alpha0,
        R=fsp.constants.GAS_CONSTANT,
        reaction_model='nth_order',
        model_params={'n': 1.0})


Let's plot the conversion against the sample temperature to see the result. The code below will produce a plot showing the conversion as a function of temperature. Users are encouraged to run the example locally to see the result.


.. code-block:: python

    # Plot conversion
    plt.plot(T_array, alpha_sol,
             label=f"{beta} K/min")


    # Plot meta data
    plt.title("Conversion of a Single-Step Pyrolyis Reaction")
    plt.xlabel("Sample Temperature / K")
    plt.ylabel("Conversion ($\\alpha$) / -")

    plt.tight_layout()
    plt.legend()
    plt.grid()

This example is also available as Jupyter Notebook.
