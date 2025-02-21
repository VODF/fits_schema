"""
Schema definitions for FITS binary table extensions.

See section 7.3 of the FITS standard:
https://fits.gsfc.nasa.gov/standard40/fits_standard40aa-le.pdf
"""

import logging
import re
from abc import ABCMeta, abstractmethod

import astropy.units as u
import numpy as np
from astropy.io import fits
from astropy.table import Table

from .exceptions import (
    RequiredMissing,
    SchemaError,
    WrongDims,
    WrongShape,
    WrongType,
    WrongUnit,
)
from .header import Header, HeaderCard
from .schema_element import SchemaElement
from .utils import log_or_raise

__all__ = [
    "BinaryTableHeader",
    "Column",
    "BinaryTable",
    "Bool",
    "Byte",
    "Int16",
    "Int32",
    "Int64",
    "String",
    "Float",
    "Double",
    "ComplexFloat",
    "ComplexDouble",
]

log = logging.getLogger(__name__)
_string_length = np.vectorize(len)


FORTRAN_FORMAT_REGEX = re.compile(
    r"""
    ^(
        A\d+ |                                  # Character: Aw (A followed by digits)
        [IBOZ]\d+(\.\d+)? |                     # Integer: Iw[.m], Binary/Octal/Hex: Bw[.m], Ow[.m], Zw[.m]
        F\d+\.\d+ |                             # Fixed-point: Fw.d
        (E|D|G)\d+\.\d+(E\d+)? |                # Exponential/General: Ew.d[Ee], Dw.d[Ee], Gw.d[Ee]
        EN\d+\.\d+ |                            # Engineering: ENw.d
        ES\d+\.\d+                              # Scientific: ESw.d
    )$
""",
    re.VERBOSE,
)


class BinaryTableHeader(Header):
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


class Column(SchemaElement, metaclass=ABCMeta):
    """A binary table column descriptor.

    Attributes
    ----------
    unit: astropy.units.Unit | None
        unit of the column
    strict_unit: bool
        If True, the unit must match exactly, not only be convertible.
    required: bool
        If this column is required (True) or optional (False)
    name: str | None
        Use to specify a different column name than the class attribute name.
    ndim: int | None
        Dimensionality of a single row, numbers have ndim=0.
        The resulting data column has ``ndim_col = ndim + 1``
    shape: tuple[int] | None
        Shape of a single row.
    display_format: str | None
        Fortran-style text format string (TDISP#) for converting values to
        strings. E.g. "F5.2" means display as floating points of width 5 with 2
        digits after decimal point
    """

    def __init__(
        self,
        *,
        strict_unit=False,
        name=None,
        ndim: int | None = None,
        shape: tuple[int] | None = None,
        display_format: str | None = None,
        description: str = "",
        required: bool = True,
        unit: u.Unit | None = None,
        examples: list[str] | None = None,
        reference: str | None = None,
        ucd: str | None = None,
        ivoa_name: str | None = None,
    ):
        super().__init__(
            description=description,
            required=required,
            unit=unit,
            examples=examples,
            reference=reference,
            ucd=ucd,
            ivoa_name=ivoa_name,
        )

        self.strict_unit = strict_unit
        self.name = name
        self.shape = shape
        self.ndim = ndim
        self.display_format = display_format

        if self.display_format and not FORTRAN_FORMAT_REGEX.fullmatch(
            self.display_format
        ):
            raise SchemaError(
                f"Column {self.name}: display format '{self.display_format}'"
                "is not valid."
            )

        if self.shape is not None:
            self.shape = tuple(shape)
            # Dimensionality of the table is one more than that of a single row
            if self.ndim is None:
                self.ndim = len(self.shape)
            elif self.ndim != len(self.shape):
                raise SchemaError(f"Shape={shape} and ndim={ndim} do not match")
        else:
            # simple column by default
            if self.ndim is None:
                self.ndim = 0

    def __get__(self, instance, owner=None):
        """Get data."""
        # class attribute access
        if instance is None:
            return self

        try:
            return instance.__data__[self.name]
        except KeyError:
            if self.required is False:
                return None
            raise

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
        del instance.__data__[self.name]

    def __repr__(self):
        """Representation of the class."""
        unit = f"'{self.unit.to_string('fits')}'" if self.unit is not None else None
        return (
            f"{self.__class__.__name__}("
            f"name={self.name!r}, required={self.required}, unit={unit}"
            ")"
        )

    @property
    @abstractmethod
    def dtype(self) -> type:
        """Equivalent numpy dtype."""

    @property
    @abstractmethod
    def tform_code(self) -> str:
        """FITS Format code."""

    def validate_data(self, data, onerror="raise"):
        """Validate the data of this column in table."""
        if data is None:
            if self.required:
                log_or_raise(
                    f"Column '{self.name}' is required but missing",
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

            if np.dtype(data.dtype).char in ["S", "U"]:
                # special case is if this is a string column, i.e. a Char field
                # with shape=None. There we should ignore the length of the
                # numpy dtype (e.g. 'S12' should be checked as 'S'). Note that
                # inside a FITS file, the strings are stored as dtype('S')
                # always, however if you call hdu.data[column], they are
                # magically converted to 'U' by astropy, but not for astropy
                # Tables or np.recarrays (why is not obvious).
                try:
                    # string data must be ascii only, so ensure it by casting:
                    data = np.asanyarray(data).astype("S")
                except UnicodeError as err:
                    log_or_raise(
                        f"Column '{self.name}': non-ascii data is not allowed ({data=}): {err}",
                        WrongType,
                        log=log,
                        onerror=onerror,
                    )
            else:
                data = np.asanyarray(data).astype(self.dtype, casting="safe")
        except TypeError as e:
            log_or_raise(
                f"Column '{self.name}': dtype not convertible to column dtype: {e}",
                WrongType,
                log=log,
                onerror=onerror,
            )

        if self.strict_unit and hasattr(data, "unit") and data.unit != self.unit:
            log_or_raise(
                f"Column '{self.name}': unit '{data.unit}' of data does not match specified unit '{self.unit}'",
                WrongUnit,
                log=log,
                onerror=onerror,
            )

        # a table has one dimension more than it's rows,
        # we also allow a single scalar value for scalar rows
        if data.ndim != self.ndim + 1 and not (data.ndim == 0 and self.ndim == 0):
            log_or_raise(
                f"Column '{self.name}': dimensionality of rows is {data.ndim - 1}, should be {self.ndim}",
                WrongDims,
                log=log,
                onerror=onerror,
            )

        # the rest of the tests is done on a quantity object with correct dtype
        if self.unit:
            try:
                q = u.Quantity(
                    data, self.unit, copy=False, ndmin=self.ndim + 1, dtype=self.dtype
                )
                data = q
            except u.UnitConversionError as e:
                log_or_raise(
                    "Column '{self.name}': " + str(e),
                    WrongUnit,
                    log=log,
                    onerror=onerror,
                )

        shape = data.shape[1:]
        if self.shape is not None and self.shape != shape:
            log_or_raise(
                f"Column '{self.name}': Shape {shape} does not match required shape {self.shape}",
                WrongShape,
                log=log,
                onerror=onerror,
            )

        return data


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
            dct["__header__"] = BinaryTableHeader

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

    def __init__(self, hdu):
        if isinstance(hdu, fits.BinTableHDU):
            self.validate_hdu(hdu)
            self.__data__ = Table.read(hdu)
        elif isinstance(hdu, Table):
            # make sure we have a table and not a subclass
            # FIXME: there might be a better way than creating a hdu from the table
            # i.e. implement validate_table
            table = hdu
            hdu = fits.BinTableHDU(hdu)
            self.validate_hdu(hdu)
            self.__data__ = Table(table, copy=False)
        else:
            raise TypeError(
                f"hdu must be a BinTableHDU or Table, got {hdu.__class__!r}"
            )

        self.header = self.__header__(hdu.header)

    def validate_data(self):
        """Check that data matches schema."""
        for k, col in self.__columns__.items():
            validated = col.validate_data(getattr(self, k))
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


class Bool(Column):
    """A Boolean binary table column."""

    tform_code = "L"
    dtype = bool


class BitField(Column):
    """Bitfield binary table column."""

    tform_code = "X"
    dtype = bool


class Byte(Column):
    """Byte binary table column."""

    tform_code = "B"
    dtype = np.uint8


class Int16(Column):
    """16 Bit signed integer binary table column."""

    tform_code = "I"
    dtype = np.int16


class Int32(Column):
    """32 Bit signed integer binary table column."""

    tform_code = "J"
    dtype = np.int32


class Int64(Column):
    """64 Bit signed integer binary table column."""

    tform_code = "K"
    dtype = np.int64


class String(Column):
    """Character string binary table column."""

    tform_code = "A"
    dtype = np.dtype("S")

    def __init__(
        self,
        *,
        string_size: int = None,
        max_string_length: int = None,
        min_string_length: int = None,
        **kwargs,
    ):
        """
        Construct a String column.

        Parameters
        ----------
        string_size: int | None
           required column element storage size in characters. Note that strings
           may be smaller than this size if termination characters are used.
        max_string_length: int | None
           require strings to be less than or equal to this number of characters,
           even if storage size is larger.
        min_string_length: int | None
           require strings to be longer than or equal to this number of characters
        """
        super().__init__(**kwargs)
        self.string_size = string_size
        self.max_string_length = max_string_length
        self.min_string_length = min_string_length

        if (
            (string_size and max_string_length)
            and (max_string_length > string_size)
            or (string_size and min_string_length)
            and (min_string_length > string_size)
        ):
            raise ValueError(
                "Specified a max or min string length that is "
                "incompatible with the required string size"
            )

    def validate_data(self, data, onerror="raise"):
        """Validate the data of this column in table."""
        super().validate_data(data, onerror=onerror)

        # check fixed-size, note that if inside an HDU, even byte-strings get
        # returned as unicode, so have to multiply size by 4.
        if self.string_size and (
            (data.dtype.char == "U" and data.dtype.itemsize != self.string_size * 4)
            or (data.dtype.char == "S" and data.dtype.itemsize != self.string_size)
        ):
            log_or_raise(
                f"Column '{self.name}': storage size should be {self.string_size} bytes, not {data.dtype.itemsize}",
                WrongShape,
                log=log,
                onerror=onerror,
            )

        if self.max_string_length and np.any(
            _string_length(data) > self.max_string_length
        ):
            log_or_raise(
                f"Column '{self.name}': strings must not be longer than {self.max_string_length} characters.",
                WrongShape,
                log=log,
                onerror=onerror,
            )

        if self.min_string_length and np.any(
            _string_length(data) < self.min_string_length
        ):
            log_or_raise(
                f"Column '{self.name}': strings must not be shorter than {self.min_string_length} characters.",
                WrongShape,
                log=log,
                onerror=onerror,
            )


class Float(Column):
    """Single precision floating point binary table column."""

    tform_code = "E"
    dtype = np.float32


class Double(Column):
    """Single precision floating point binary table column."""

    tform_code = "D"
    dtype = np.float64


class ComplexFloat(Column):
    """Single precision complex binary table column."""

    tform_code = "C"
    dtype = np.csingle


class ComplexDouble(Column):
    """Single precision complex binary table column."""

    tform_code = "M"
    dtype = np.cdouble
