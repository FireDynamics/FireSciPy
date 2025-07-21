Simple Design Fire
==================


This usage example shows how to compute a simple t-squared design fire using the
:func:`firescypy.handcalculation.design_fires.simple_design_fire` function.

.. code-block:: python

    from firescypy.handcalculation.design_fires import simple_design_fire

    # Define design fire parameters
    Q_max = 1000       # Maximum HRR [kW]
    Q_total = 300000   # Total energy [kJ]
    decay_model = "exponential"

    # Optional keyword arguments
    kwargs = {
        "alpha": "fast",         # Growth rate
        "decay_constant": 0.03,  # Decay rate [1/s]
        "t_end": 300,            # Duration [s]
        "num_points": 100        # Resolution
    }

    # Compute time and HRR arrays
    t, Q = simple_design_fire(Q_max, Q_total, decay_model, **kwargs)
