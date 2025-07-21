Design Fire Example
===================

This example demonstrates how to generate a simple design fire using a t-squared growth curve
and an exponential decay phase. The function :func:`simple_design_fire` combines a growth, steady-state,
and decay phase based on user-specified input parameters.

As a result, data time series are provided to be used further, for example in a plot.

It is part of the hand calculation tools available in FireSciPy.

.. code-block:: python

    import matplotlib.pyplot as plt
    from firescypy.handcalculation.design_fires import simple_design_fire

    # Define parameters
    Q_max = 1000  # Maximum heat release rate [kW]
    Q_total = 300000  # Total energy [kJ]
    decay_model = "exponential"

    # Optional parameters for exponential decay
    kwargs = {
        "alpha": "fast",         # Growth rate classification
        "decay_constant": 0.03,  # Exponential decay rate [1/s]
        "t_end": 300,            # Duration of decay phase [s]
        "num_points": 100        # Resolution
    }

    # Compute the design fire
    time, HRR = simple_design_fire(Q_max, Q_total, decay_model, **kwargs)

    # Plot the result
    plt.plot(time, HRR)
    plt.xlabel("Time / s")
    plt.ylabel("Heat Release Rate (HRR) / kW")
    plt.title("Simple Design Fire Curve")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
