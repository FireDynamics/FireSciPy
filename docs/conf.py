import os
import sys
fsp_path = os.path.join("..", "src")
sys.path.insert(0, os.path.abspath(fsp_path))  # So firescipy is importable

project = "FireSciPy"
author = "Tristan Hehnen, Lukas Arnold"
release = "0.1.0"

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",  # For NumPy-style docstrings
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax"  # or "sphinx.ext.imgmath" for image-based output
]

templates_path = ["_templates"]
exclude_patterns = []

html_theme = "classic"  # You can switch to 'sphinx_rtd_theme' or others later
html_static_path = ["_static"]
