# Configuration file for the Sphinx documentation builder.

# -- Project information

project = 'HAET'
copyright = '2025, Pedro M. P. Curvo, Mohammadmahdi Rahimi, Salvador Torpes'
author = 'Pedro M. P. Curvo, Mohammadmahdi Rahimi, Salvador Torpes'

release = '0.1'
version = '0.1.0'

# -- General configuration

# Add project root directory to sys.path for proper imports
import os
import sys
sys.path.insert(0, os.path.abspath('../../'))

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx.ext.viewcode',  # Add links to source code
    'sphinx.ext.napoleon',  # Support for NumPy and Google style docstrings
]

# Settings for autodoc
autodoc_mock_imports = ['torch', 'torch_cluster', 'timm', 'numpy', 'einops', 'flash_attn', 'balltree', 'torch_scatter']
autodoc_typehints = 'description'
autoclass_content = 'both'
autodoc_member_order = 'bysource'

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
    'torch': ('https://pytorch.org/docs/stable/', None),
}
intersphinx_disabled_domains = ['std']

templates_path = ['_templates']

# -- Options for HTML output

html_theme = 'sphinx_rtd_theme'

# -- Options for EPUB output
epub_show_urls = 'footnote'
