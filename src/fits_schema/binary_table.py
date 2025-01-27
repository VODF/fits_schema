"""
Schema definitions for FITS binary table extensions.

See section 7.3 of the FITS standard:
https://fits.gsfc.nasa.gov/standard40/fits_standard40aa-le.pdf
"""

from dataclasses import dataclass
import logging
from abc import ABCMeta, abstractmethod
from typing import Tuple, Type

import astropy.units as u
from astropy.units.decorators import NoneType
import numpy as np
from astropy.io import fits
from astropy.table import Table

from .exceptions import (
    RequiredMissing,
    WrongDims,
    WrongShape,
    WrongType,
    WrongUnit,
)
from .header import HeaderCard, HeaderSchema
from .utils import log_or_raise
from .schema_element import SchemaElement

__all__ = [
    "BinaryTableHeader",
    "Column",
    "BinaryTable",
    "Bool",
    "Byte",
    "Int16",
    "Int32",
    "Int64",
    "Char",
    "Float",
    "Double",
    "ComplexFloat",
    "ComplexDouble",
]

log = logging.getLogger(__name__)


class BinaryTableHeader(HeaderSchema):
    """Default binary table header schema."""

    XTENSION = HeaderCard(allowed_values="BINTABLE", position=0)
    BITPIX = HeaderCard(allowed_values=8, position=1)
    NAXIS = HeaderCard(allowed_values=2, position=2)
    NAXIS1 = HeaderCard(type_=int, position=3)
    NAXIS2 = HeaderCard(type_=int, position=4)
    PCOUNT = HeaderCard(type_=int, position=5)
    GCOUNT = HeaderCard(allowed_values=1, position=6)
    TFIELDS = HeaderCard(type_=int, position=7)
    EXTNAME = HeaderCard(required=False, type_=str)

@dataclass
class Column(SchemaElement, metaclass=ABCMeta):
    """A binary table column descriptor.

    Attributes
    ----------
    unit: astropy.units.Unit
        unit of the column
    strict_unit: bool
        If True, the unit must match exactly, not only be convertible.
    required: bool
        If this column is required (True) or optional (False)
    name: str
        Use to specify a different column name than the class attribute name.
    ndim: int
        Dimensionality of a single row, numbers have ndim=0.
        The resulting data column has ``ndim_col = ndim + 1``
    shape: Tuple[int]
        Shape of a single row.
    """

    #: allow compatible units if False
    strict_unit: bool = False

    #: dimensionality of column
    ndim: int | None = None

    #: specify exact shape (length of each dimension)
    shape: Tuple[int] | None = None

    # the data type of the column
    dtype: Type = None

    def __post_init__(self):
        """Check the schema."""
        super().__post_init__()

        if self.shape is not None:
            self.shape = tuple(self.shape)
            # Dimensionality of the table is one more than that of a single row
            if self.ndim is None:
                self.ndim = len(self.shape)
            elif self.ndim != len(self.shape):
                raise ValueError(f"Shape={shape} and ndim={ndim} do not match")
        else:
            # simple column by default
            if self.ndim is None:
                self.ndim = 0

    def _check_name(self):
        """Ensure column name follows FITS conventions."""
        # allow anything?
        pass


    def __get__(self, instance, owner=None):
        """Get data."""
        # class attribute access
        if instance is None:
            return self

        return instance.__data__.get(self.name)

    def __set__(self, instance, value):
        """Set data."""
        instance.__data__[self.name] = value

    def __set_name__(self, owner, name):
        """Rename variable (protocol)."""
        # respect user override for names that are not valid identifiers
        if self.name is None:
            self.name = name

    def __delete__(self, instance):
        """Clear data of this column."""
        if self.name in instance.__data__:
            del instance.__data__[self.name]

    @property
    @abstractmethod
    def dtype():
        """Equivalent numpy dtype."""

    def validate_data(self, data, onerror="raise"):
        """Validate the data of this column in table."""
        if data is None:
            if self.required:
                log_or_raise(
                    f"Column {self.name} is required but missing",
                    RequiredMissing,
                    log=log,
                    onerror=onerror,
                )
            else:
                return

        # let's test first for the datatype
        try:
            # casting = 'safe' makes sure we don't change values
            # e.g. casting doubles to integers will no longer work
            data = np.asanyarray(data).astype(self.dtype, casting="safe")
        except TypeError as e:
            log_or_raise(
                f"dtype not convertible to column dtype: {e}",
                WrongType,
                log=log,
                onerror=onerror,
            )

        if self.strict_unit and hasattr(data, "unit") and data.unit != self.unit:
            log_or_raise(
                f"Unit {data.unit} of data does not match specified unit {self.unit}",
                WrongUnit,
                log=log,
                onerror=onerror,
            )

        # a table has one dimension more than it's rows,
        # we also allow a single scalar value for scalar rows
        if data.ndim != self.ndim + 1 and not (data.ndim == 0 and self.ndim == 0):
            log_or_raise(
                f"Dimensionality of rows is {data.ndim - 1}, should be {self.ndim}",
                WrongDims,
                log=log,
                onerror=onerror,
            )

        # the rest of the tests is done on a quantity object with correct dtype
        try:
            q = u.Quantity(
                data, self.unit, copy=False, ndmin=self.ndim + 1, dtype=self.dtype
            )
        except u.UnitConversionError as e:
            log_or_raise(str(e), WrongUnit, log=log, onerror=onerror)

        shape = q.shape[1:]
        if self.shape is not None and self.shape != shape:
            log_or_raise(
                f"Shape {shape} does not match required shape {self.shape}",
                WrongShape,
                log=log,
                onerror=onerror,
            )

        return q


class BinaryTableMeta(type):
    """Metaclass for the BinaryTable class."""

    def __new__(cls, name, bases, dct):
        """Create class."""
        dct["__columns__"] = {}
        dct["__slots__"] = ("__data__", "header")

        header_schema = dct.get("__header__", None)
        if header_schema is not None and not issubclass(
            header_schema, BinaryTableHeader
        ):
            raise TypeError(
                "`__header__` must be a class inheriting from `BinaryTableHeader`"
            )

        # ensure we have a __header__ if not specified
        if not header_schema:
            dct["__header__"] = BinaryTableHeader()

        # inherit header schema and  from bases
        for base in reversed(bases):
            if hasattr(base, "__header__"):
                dct["__header__"].update(base.__header__)

            if issubclass(base, BinaryTable):
                dct["__columns__"].update(base.__columns__)

        # collect columns of this new schema
        for k, v in dct.items():
            if isinstance(v, Column):
                k = v.name or k
                dct["__columns__"][k] = v

        if header_schema is not None:
            # add user defined header last
            dct["__header__"].update(header_schema)

        new_cls = super().__new__(cls, name, bases, dct)
        return new_cls


class BinaryTable(metaclass=BinaryTableMeta):
    """Schema definition class for a binary table.

    Examples
    --------
    >>> from fits_schema.binary_table import BinaryTable, BinaryTableHeader
    >>> from fits_schema.binary_table import Float, Int32
    >>> from fits_schema.header import HeaderCard
    >>>
    >>> class Events(BinaryTable):
    ...    EVENT_ID = Int32()
    ...    ENERGY   = Float(unit="TeV")
    ...
    ...    class __header__(BinaryTableHeader):
    ...        HDUCLASS = HeaderCard(required=True, allowed_values="Events")
    """

    def __init__(self, **column_data):
        self.__data__ = {}
        self.header = fits.Header()

        for k, v in column_data.items():
            setattr(self, k, v)

    def validate_data(self):
        """Check that data matches schema."""
        for k, col in self.__columns__.items():
            validated = col.validate_data(self.__data__.get(k))
            if validated is not None:
                setattr(self, k, validated)

    @classmethod
    def validate_hdu(cls, hdu: fits.BinTableHDU, onerror="raise"):
        """Check that HDU matches schema."""
        if not isinstance(hdu, fits.BinTableHDU):
            raise TypeError("hdu is not a BinTableHDU")

        cls.__header__.validate_header(hdu.header, onerror=onerror)
        required = set(c.name for c in cls.__columns__.values() if c.required)
        missing = required - set(c.name for c in hdu.columns)
        if missing:
            log_or_raise(
                f"The following required columns are missing {missing}",
                RequiredMissing,
                log=log,
                onerror=onerror,
            )

        table = Table.read(hdu)
        for k, col in cls.__columns__.items():
            if k in table.columns:
                col.validate_data(table[k], onerror=onerror)

@dataclass
class Bool(Column):
    """A Boolean binary table column."""

    tform_code = "L"
    dtype : Type = bool

@dataclass
class BitField(Column):
    """Bitfield binary table column."""

    tform_code = "X"
    dtype : Type = bool


@dataclass
class Byte(Column):
    """Byte binary table column."""

    tform_code : str = "B"
    dtype : Type = np.uint8

@dataclass
class Int16(Column):
    """16 Bit signed integer binary table column."""

    tform_code : str = "I"
    dtype : Type = np.int16

@dataclass
class Int32(Column):
    """32 Bit signed integer binary table column."""

    tform_code : str = "J"
    dtype : Type = np.int32

@dataclass
class Int64(Column):
    """64 Bit signed integer binary table column."""

    tform_code : str = "K"
    dtype : Type = np.int64

@dataclass
class Char(Column):
    """Single byte character binary table column."""

    tform_code : str = "A"
    dtype : Type = np.dtype("S1")

@dataclass
class Float(Column):
    """Single precision floating point binary table column."""

    tform_code : str = "E"
    dtype : Type = np.float32

@dataclass
class Double(Column):
    """Single precision floating point binary table column."""

    tform_code : str = "D"
    dtype : Type = np.float64

@dataclass
class ComplexFloat(Column):
    """Single precision complex binary table column."""

    tform_code : str = "C"
    dtype : Type = np.csingle

@dataclass
class ComplexDouble(Column):
    """Single precision complex binary table column."""

    tform_code : str = "M"
    dtype : Type = np.cdouble
