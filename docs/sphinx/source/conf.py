# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import plotly.io as pio
# This is the "secret" scraper utility built into plotly
from plotly.io._sg_scraper import plotly_sg_scraper

project = 'Expectation Propagation'
copyright = '2026, Joaquin Rapela'
author = 'Joaquin Rapela'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx_gallery.gen_gallery']
# extensions = []

sphinx_gallery_conf = {
    'image_scrapers': (plotly_sg_scraper),
    'examples_dirs': '../../../examples/sphinx-gallery',   # path to your examples scripts
    'gallery_dirs': 'auto_examples',  # path to where the gallery should be placed
}

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']
