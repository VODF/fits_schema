"""FITS Schema."""

from .version import __version__
from . import binary_table
from . import header

__all__ = ["__version__", "binary_table", "header"]
