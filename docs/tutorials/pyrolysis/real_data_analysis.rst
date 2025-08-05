Example Processing of Data from TGA Experiments
===============================================

In this example, the conversion data is derived from experimental thermogravimetric analysis (TGA) data.

This example is also available as Jupyter Notebook.

Theoretical Background
----------------------

The apparent activation energy :math:`E_a` is estimated using the well-established isoconversional method known as Kissinger–Akahira–Sunose (KAS). For background on the theory and mathematical formulation, see the corresponding section in the :doc:`mock_kinetics_computation`..

We make the distinction here, between the true activation energy :math:`E` and the apparent activation energy :math:`E_a`.

In micro-scale experiments such as TGA, it is generally not possible to deduce the fundamental reaction mechanisms or individual reaction steps involved in the decomposition. What appears as a single peak in the mass loss rate may in fact result from multiple overlapping reactions, with one step acting as rate-limiting and dominating the overall shape.

As such, the estimated activation energy from these experiments — :math:`E_a` — is considered apparent, reflecting the net behavior rather than a single elementary step.

The true activation energy is only definitively known in modeling scenarios, where the decomposition process is explicitly prescribed and controlled by the user (e.g., synthetic data or known reaction schemes).

See the ICTAC recommendations below for a detailed discussion of this distinction.

To apply the KAS method meaningfully, experimental data must be recorded under multiple temperature programs, either with linear heating rates or under isothermal conditions. The ICTAC Kinetics Committee recommends using at least three, preferably five, distinct datasets. If linear heating is used, the lowest and highest heating rates should differ by at least a factor of 10.

To ensure high-quality experimental data, a few key aspects must be considered:

- **Thermophysical effects of the sample material**: These influence the heat transfer within the sample and can distort the measured reaction rate. Their impact can be minimized by reducing the sample mass and repeating experiments with decreasing amounts until the results (e.g., mass loss rate curves) can be superimposed. This is especially important in micro-scale TGA experiments.

- **Secondary reactions involving evolved gases**: In some materials, volatile products may react with the remaining solid. This effect can be mitigated by increasing the purge gas flow rate to carry away evolved gases more effectively. Again, comparing mass loss rate curves for different flow rates can help confirm when the flow is sufficient (i.e., when the curves overlap).

In general, **apparent kinetic parameters should not be estimated from a single temperature program**. Multiple kinetic triplets can describe the same conversion curve equally well, making the solution **ambiguous**. See also:

- `Single heating rate methods are a faulty approach to pyrolysis kinetics (https://doi.org/10.1007/s13399-022-03735-z) <https://doi.org/10.1007/s13399-022-03735-z>`_
- `Computational aspects of kinetic analysis.: Part B: The ICTAC Kinetics Project — the decomposition kinetics of calcium carbonate revisited, or some tips on survival in the kinetic minefield (https://doi.org/10.1016/S0040-6031(00)00444-5) <https://doi.org/10.1016/S0040-6031(00)00444-5>`_


More detailed guidance is available from the recommendations provided by the International Confederation for Thermal Analysis and Calorimetry (ICTAC) Kinetics Committee:

- `ICTAC Kinetics Committee recommendations for performing kinetic computations on thermal analysis data <https://doi.org/10.1016/j.tca.2011.03.034>`_
- `ICTAC Kinetics Committee recommendations for collecting experimental thermal analysis data for kinetic computations <https://doi.org/10.1016/j.tca.2014.05.036>`_
- `ICTAC Kinetics Committee recommendations for analysis of multi-step kinetics <https://doi.org/10.1016/j.tca.2020.178597>`_
- `ICTAC Kinetics Committee recommendations for analysis of thermal decomposition kinetics <https://doi.org/10.1016/j.tca.2022.179384>`_



Activation Energy Analysis (KAS) from TGA Experiments
-----------------------------------------------------

In this example, the apparent activation energy :math:`E_a` is estimated using the KAS method based on experimental TGA data. The data was recorded for a **sample mass of 1 mg**, using multiple linear heating rates.

Other datasets from the same study — including those with different sample masses and complementary microscale calorimeter data — are available at:

- [Thermogravimetric And Microscale Calorimeter Data on Cast PMMA (https://doi.org/10.24355/dbbs.084-202504170956-0)](https://doi.org/10.24355/dbbs.084-202504170956-0)


At first, the necessary modules are imported.


.. code-block:: python

    # Import necessary packages
    import os

    import numpy as np
    import matplotlib.pyplot as plt
    import firescipy as fsp

    from scipy.ndimage import uniform_filter1d


In this example we analyse TGA data recorded at different linear heating rates. For each heating rate, three repetitions were conducted. The data is provided in files that follow the CSV format, one for each repetition and experimental condition. These are plain text files with a `.txt` extension, but they follow a typical tab-delimited CSV structure.

As in the synthetic data example, a Python dictionary is used to store all data related to this investigation. This structure is initialised using the function :func:`initialize_investigation_skeleton`.

Each CSV file is read into a Pandas DataFrame. If the files have meaningful names they can be directly used to label the data. For example, a file named `tga_dynamic_n2_dyn10_powder_1mg_r1.txt` corresponds to a mass of 1mg for a powdered sample, subjected to a 10 K/min heating rate and repetition 1.

The temperatures in the files are reported in degrees Celsius and must be converted to Kelvin before further processing, as FireSciPy's kinetic computations assume Kelvin.

Once converted, each DataFrame is added to the data structure using :func:`add_constant_heating_rate_tga`. This function stores the input data under the appropriate labels for **heating rate** and **repetition**.

Below is the code to read the files, adjust the temperatures, and store the results:


.. code-block:: python

    # Path to the experimental data csv files.
    fsp_data_path = os.path.join("C:\\", "path", "to", "the", "CSV_files")
    exp_root = os.path.join(fsp_data_path, "docs", "tutorials", "pyrolysis", "data")


    # Initialise data structure for the kinetics assessment
    PMMA_data_1mg = fsp.pyrolysis.kinetics.initialize_investigation_skeleton(
        material=f"PMMA",
        investigator="John Doe, Miskatonic University",
        instrument="TGA/DSC 3+, Mettler Toledo",
        date="Stardate: 42.69",
        notes="Constant heating rates, sample mass 1mg",
        signal={"name": "Mass", "unit": "mg"})


    for file_name in os.listdir(exp_root):
        if "powder_1mg_" in file_name:
    #         print(file_name)

            # Parse metadata from file name: heating rate and repetition
            name_parts = file_name.split("_")
            hr_value = int(name_parts[3][3:])  # e.g., extracts 10 from 'dyn10'
            hr_label = f"{hr_value}_Kmin"
            rep_label = f"Rep_{name_parts[-1][1]}"  # e.g., 'Rep_1' from '..._r1.txt'

            # Read CSV file as Pandas DataFrame
            exp_path = os.path.join(exp_root, file_name)
            exp_df = pd.read_csv(exp_path, header=0, skiprows=[1],
                                 delimiter=('\t'), encoding="cp858")

            # Adjust temperature to Kelvin
            exp_df["ts"] = exp_df["ts"] + 273.15
            exp_df["tr"] = exp_df["tr"] + 273.15

            # Add DataFrame to database
            fsp.pyrolysis.kinetics.add_constant_heating_rate_tga(
                database=PMMA_data_1mg,
                condition=hr_label,
                repetition=rep_label,
                raw_data=exp_df,
                data_type="integral",
                set_value=[hr_value, "K/min"])


    # Adjust column mapping for later functions
    column_mapping = {
            'time': 't',
            'temp': 'ts',
            'signal': 'weight'}
    for hr_label in PMMA_data_1mg["experiments"]["TGA"]["constant_heating_rate"]:
        # Compute averages and standard deviations per heating rate
        fsp.pyrolysis.kinetics.combine_repetitions(
            database=PMMA_data_1mg,
            condition=hr_label,
            column_mapping=column_mapping)


Next, the conversion curves are computed using the function :func:`compute_conversion`. This function operates on the averaged data produced by :func:`combine_repetitions` in the previous step.

After this step, the computed conversion data will be available in the data structure and can be visualized or used in the isoconversional analysis.

To compute conversions for all available heating rates, simply run:


.. code-block:: python

    fsp.pyrolysis.kinetics.compute_conversion(
        database=PMMA_data_1mg,
        condition="all",
        setup="constant_heating_rate")


Now, the desired conversion levels must be specified to indicate where the apparent activation energy :math:`E_a` should be evaluated.

Commonly, these levels lie in the range :math:`0.10 < \alpha < 0.90` or :math:`0.05 < \alpha < 0.95`, depending on the quality of the experimental data. In the early and late stages of the reaction, the signal changes more slowly, and even small fluctuations or noise can lead to significant artifacts in the computed activation energy. This is typically visible in the :math:`E_a` versus conversion plots (see below).

However, these regions should not be discarded blindly. It's important to visually inspect the data to decide whether and to what extent the tails of the conversion curve can be included in the analysis.

The desired conversion levels can be provided as a NumPy array. The function :func:`compute_conversion_levels` performs a linear interpolation of the available data to these levels in preparation for estimating the apparent activation energy.


.. code-block:: python

    # Define conversion fractions where to evaluate the activation energy
    # Note: commonly, they range between (0.05 < α < 0.95) or (0.1 < α < 0.9)
    # conversion_levels = np.linspace(0.05, 0.95, 37)  # Δα = 2.5
    conversion_levels = np.linspace(0.01, 0.99, 99)  # Δα = 1.0


    fsp.pyrolysis.kinetics.compute_conversion_levels(
        database=PMMA_data_1mg,
        desired_levels=conversion_levels,
        setup="constant_heating_rate",
        condition="all")


The function :func:`compute_Ea_KAS` performs the final step of the isoconversional method. Internally, it first sorts the available data by heating rate (from lowest to highest) for each conversion level. Then, for each level, it carries out a linear regression according to the KAS method (with the user-defined parameters :math:`B` and :math:`C`).

The estimated activation energy values :math:`E_a` are stored in the database, alongside the intermediate fit data and statistics.

This step concludes the KAS-based estimation of the apparent activation energy.


.. code-block:: python

    # Compute the activation energy using the KAS method
    fsp.pyrolysis.kinetics.compute_Ea_KAS(
        database=PMMA_data_1mg,
        B=1.92,
        C=1.0008)


Below the result is plotted: the apparent activation energy against the conversion. In gray, the :math:`5 %` range at the ends is indicated.

The :math:`E_a` is computed in J/mol and converted here to kJ/mol, as it is a common way to use it.

Below, the apparent activation energy $E_a$ is plotted against the conversion :math:`\alpha`.

The KAS method estimates $E_a$ in units of J/mol, but it is commonly reported in kJ/mol — so the values are converted accordingly before plotting.

The shaded gray regions indicate the first and last :math:`5 %` of the conversion range. These are typically excluded from analysis due to higher sensitivity to noise and lower data reliability. In the first :math:`5 %`, artifacts are visible due to increased noise.


.. code-block:: python

    fig, ax = plt.subplots()


    # Get the Ea and convert to kJ/mol.
    Ea_results_KAS = PMMA_data_1mg["experiments"]["TGA"]["Ea_results_KAS"]
    Ea = Ea_results_KAS["Ea"]/1000
    Ea_avg = np.average(Ea)

    # Plot the Ea against conversion.
    conv = Ea_results_KAS["Conversion"]
    plt.scatter(conv,
                Ea,
                marker=".", s=42,
                facecolors="none",
                edgecolors="tab:blue",
                label=f"KAS, ΔT = {ΔT} K")


    # Shaded areas to indicate first/last 5 % (typically excluded)
    x_min = -0.025
    x_max = 1.025
    ax.axvspan(x_min, 0.05, color='gray', alpha=0.3, label="5%")
    ax.axvspan(0.95, x_max, color='gray', alpha=0.3)


    # Plot meta data.
    plt.title(f"Activation Energy (KAS), E$_a$={Ea_avg:.2f} kJ/mol (avg.)")
    plt.xlabel("Conversion ($\\alpha$) / -")
    plt.ylabel("Activation Energy (E$_a$) / kJ/mol")

    plt.xlim(left=x_min, right=x_max)
    plt.ylim(bottom=68, top=257)

    plt.tight_layout()
    plt.legend(loc="lower center")
    plt.grid()


    # # Save image.
    # plot_label = f"Ea_Estimate_KAS.png"
    # plot_path = os.path.join(plot_label)
    # plt.savefig(plot_path, dpi=320, bbox_inches='tight', facecolor='w')


FireSciPy also evaluates the quality of each linear fit used in the KAS computation. Specifically, the **root mean square error (RMSE)** and the **coefficient of determination (R²)** are calculated.

- An **RMSE** of 0 indicates a perfect fit.
- An **R²** of 1 means that the fit perfectly explains the variance in the data.

In the plot below, fluctuations are more pronounced at low levels of conversion. This further supports the common practice of excluding the edges of the conversion range (typically the first and last :math:`5 %`) in isoconversional analyses.


.. code-block:: python

    # Pre-select data sets for convenience
    Ea_KAS = PMMA_data_1mg["experiments"]['TGA']["Ea_results_KAS"]

    # Optional: Reduce number of data points by using every n-th
    nth = 1

    markersize = 42

    fig, ax = plt.subplots()

    # Plot R² statistics
    x_data = np.asarray(Ea_KAS["Conversion"])[::nth]
    y_data = np.asarray(Ea_KAS["R_squared"])[::nth]
    plt.scatter(x_data,
                y_data,
                marker='.', s=markersize,
                facecolors='none',
                edgecolors='tab:blue',
                label=f"R² avrg.: {np.average(y_data):.6f}")


    # Plot RMSE statistics
    y_data = np.asarray(Ea_KAS["RMSE"])[::nth]
    plt.scatter(x_data,
                y_data,
                marker='.', s=markersize,
                facecolors='none',
                edgecolors='tab:orange',
                label=f"RMSE avrg.: {np.average(y_data):.6f}")


    # Shaded areas to indicate first/last 5 % (typically excluded)
    x_min = -0.025
    x_max = 1.025
    ax.axvspan(x_min, 0.05, color='gray', alpha=0.3, label="5%")
    ax.axvspan(0.95, x_max, color='gray', alpha=0.3)


    # Plot meta data.
    plt.title("Statistics of the KAS Fits")
    plt.xlabel("Conversion, $\\alpha$ / -")
    plt.ylabel('Arbitrary Units / -')

    plt.xlim(left=-0.025, right=1.025)
    plt.ylim(bottom=-0.05, top=1.05)

    plt.tight_layout()
    plt.legend()
    plt.grid()

    # # Save image.
    # plot_label = "Ea_Estimate_KAS_Statistics.png"
    # plot_path = os.path.join(plot_label)
    # plt.savefig(plot_path, dpi=320, bbox_inches='tight', facecolor='w')


Data produced during intermediate steps can also be accessed. For example, the averaged TG curves from the combined data sets can be used to plot normalised mass loss rates (MLR). This is shown below for heating rates of 30 K/min and 60 K/min.

The curves are aligned such that the final mass approaches zero, which is common when no significant residue remains. This zeroing helps compare curves consistently even when initial/final absolute values vary slightly. The MLR is then normalised by the initial mass to allow comparison between different heating rates.


.. code-block:: python

    hr_labels = ["30_Kmin", "60_Kmin"]

    for hr_label in hr_labels:
        # Access combined data
        PMMA_exp = PMMA_data_1mg["experiments"]["TGA"]["constant_heating_rate"][hr_label]["combined"]

        # Get average mass data
        mass = np.asarray(PMMA_exp["Mass_Avg"])

        # Estimate final mass (no residue was left in the experiment)
        mass_min = np.average(mass[-10:])
        mass_adj = mass - mass_min

        # Estimate initial mass
        mass_max = np.average(mass_adj[:10])

        # Compute normalised mass loss rate
        mlr = -np.gradient(mass_adj / mass_max, edge_order=1)
        mlr_smooth = -uniform_filter1d(mlr, size=25)

        # Plot the mass loss rate
        plt.plot(PMMA_exp["Temperature_Avg"],
                 mlr,
                 label=f"{hr_label.split('_')[0]} K/min (1mg)")


    # Plot meta data.
    plt.xlabel("Sample Temperature / K")
    plt.ylabel("Normalised Mass Loss Rate / 1/s")

    plt.xlim(left=380,right=770)

    plt.legend()
    plt.grid()


    # Save image.
    plot_label = f"NormalisedMLR.png"
    plot_path = os.path.join(plot_label)
    plt.savefig(plot_path, dpi=320, bbox_inches='tight', facecolor='w')


For more details, see the
`FireSciPy documentation <https://yourdocsurl>`_.

This example is also available as Jupyter Notebook.
