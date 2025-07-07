import os
import sys
fsp_path = os.path.join("..", "..")
sys.path.insert(0, os.path.abspath(fsp_path))  # So FireSciPy is importable

project = "FireSciPy"
author = "Tristan Hehnen, Lukas Arnold"
release = "0.1.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",  # For NumPy-style docstrings
    "sphinx.ext.viewcode"
]

templates_path = ["_templates"]
exclude_patterns = []

html_theme = "alabaster"  # You can switch to 'sphinx_rtd_theme' or others later
html_static_path = ["_static"]
