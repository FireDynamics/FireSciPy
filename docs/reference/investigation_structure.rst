Investigation Database Structure
================================

This page describes the nested dictionary format used to store raw thermal
analysis data, metadata, and results associated with the pyrolysis kinetics
computations. This structure is initialized using
:func:`initialize_investigation_skeleton()` and modified by other functions like
:func:`add_isothermal_tga()`, :func:`combine_repetitions()` or
:func:`compute_Ea_KAS()`. This data structure provides a standardised ordering
of the data and allows the various functions to easily find the data they need,
as well as provides a location where they can store their results.

Overview over the Structure
---------------------------

The dictionary is expected to collect data from a single set of related
experiments, such as integral thermogravimetric analysis (TGA) runs using
either isothermal or constant heating rate temperature programs.

Below is an example showing both temperature programs for illustration:

.. code-block:: python

    {
        "general_information": {
            "material": "Unobtainium",
            "investigator": "John Doe, Miskatonic University",
            "instrument": "ACME TGA 9000",
            "date": "Stardate: 42.69",
            "notes": "It has a colour out of space",
            "signal": {"name": "Mass", "unit": "mg"}
        },
        "experiments": {
            "TGA": {
                "isothermal": {
                    "300_C": {
                        "raw": {
                            "Rep_1": pd.DataFrame,
                            "Rep_2": pd.DataFrame
                        },
                        "set_value": {"value": 300, "unit": "C"},
                        "data_type": "integral",
                        "combined": pd.DataFrame,
                        "conversion": pd.DataFrame,
                        "conversion_levels": pd.DataFrame
                    },
                    "350_C": { ... }
                },
                "constant_heating_rate": {
                    "10_Kmin": {
                        "raw": {
                            "Rep_1": pd.DataFrame,
                            "Rep_2": pd.DataFrame
                        },
                        "set_value": {"value": 10, "unit": "K/min"},
                        "data_type": "integral",
                        "combined": pd.DataFrame,
                        "conversion": pd.DataFrame,
                        "conversion_levels": pd.DataFrame
                    },
                    "20_Kmin": { ... }
                },
                "Ea_results_KAS": pd.DataFrame
            }
        }
    }

How to Populate this Structure
------------------------------
The structure can be initialized using
:func:`initialize_investigation_skeleton()`. At this stage, optional metadata
such as material name, investigator, and signal description can already be
provided.

Next, raw data from the experimental files must be read into Pandas DataFrames.
These are added to the structure using :func:`add_isothermal_tga()` or
:func:`add_constant_heating_rate_tga()`, depending on the temperature
program used. The data will be stored under the appropriate program, set value
(e.g., `"300_C"`), and repetition (e.g., `"Rep_1"`).

Use :func:`combine_repetitions()` to merge repetitions for a given condition.
This function computes the mean and standard deviation of the signal
(e.g., mass) and sample temperature. It creates a new DataFrame that includes:
- Time,
- Raw values per repetition,
- Averages,
- Standard deviations.

This provides users with insight into intermediate processing steps and
not just final results.

Conversion is calculated with :func:`compute_conversion()`, which uses the
combined data as input.

To enable isoconversional analysis, conversion levels must be interpolated
across temperature programs. This is done via
:func:`compute_conversion_levels()`.

Finally, apparent activation energies :math:`E_a` can be determined—for example,
using the Kissinger-Akahira-Sunose method with the Starink improvement by
calling :func:`compute_Ea_KAS()`. This function generates a new DataFrame with:
- Conversion levels,
- Apparent activation energies :math:`E_a`,
- Linear fit coefficients,
- Fit statistics such as RMSE and :math:`R^2`,
- The data points used for each regression.

The specific columns may vary depending on the method used.
