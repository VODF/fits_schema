"""
Configuration file for the Sphinx documentation builder.

This file only contains a selection of the most common options. For a full
list see the documentation:
https://www.sphinx-doc.org/en/master/usage/configuration.html
"""

# -- Project information -----------------------------------------------------
# import your own package here
import fits_schema

project = "fits_schema"
# copyright = "VODF"
author = "Maximilian Linhoff"
version = fits_schema.__version__
# The full version, including alpha/beta/rc tags.
release = version


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.githubpages",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "numpydoc",
    "sphinx_changelog",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = []

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["changes"]

# have all links automatically associated with the right domain.
default_role = "py:obj"


# intersphinx allows referencing other packages sphinx docs
intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "astropy": ("https://docs.astropy.org/en/stable", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}

# needed for now... don't know why not correctly identified by intersphinx
nitpick_ignore = [
    ("py:class", "numpy.uint8"),
    ("py:class", "numpy.complex128"),
    ("py:class", "numpy.complex64"),
    ("py:class", "numpy.float64"),
    ("py:class", "numpy.float32"),
    ("py:class", "numpy.int16"),
    ("py:class", "numpy.int32"),
    ("py:class", "numpy.int64"),
]

# -- Options for HTML output -------------------------------------------------

html_theme = "pydata_sphinx_theme"
html_theme_options = dict(
    navigation_with_keys=False,
    # setup for displaying multiple versions, also see setup in .gitlab-ci.yml
    ## Disable version switcher for now until we have a place to host the pages
    # switcher=dict(
    #     json_url="<URL to fits_template_documentation>/versions.json",  # noqa: E501
    #     version_match="latest" if ".dev" in version else f"v{version}",
    # ),
    # navbar_center=["version-switcher", "navbar-nav"],
)

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = []

numpydoc_show_class_members = False

add_module_names = False
