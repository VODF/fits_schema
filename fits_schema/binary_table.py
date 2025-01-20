'''
Schema definitions for FITS binary table extensions

See section 7.3 of the FITS standard:
https://fits.gsfc.nasa.gov/standard40/fits_standard40aa-le.pdf
'''
from abc import ABCMeta, abstractmethod
import numpy as np
import astropy.units as u
from astropy.io import fits
from astropy.table import Table
import logging

from .header import HeaderSchema, HeaderCard, HeaderSchemaMeta
from .exceptions import (
    WrongUnit, WrongDims, WrongType, RequiredMissing, WrongShape,
)
from .utils import log_or_raise


log = logging.getLogger(__name__)


class BinaryTableHeader(HeaderSchema):
    '''default binary table header schema'''
    XTENSION = HeaderCard(allowed_values='BINTABLE', position=0)
    BITPIX = HeaderCard(allowed_values=8, position=1)
    NAXIS = HeaderCard(allowed_values=2, position=2)
    NAXIS1 = HeaderCard(type_=int, position=3)
    NAXIS2 = HeaderCard(type_=int, position=4)
    PCOUNT = HeaderCard(type_=int, position=5)
    GCOUNT = HeaderCard(allowed_values=1, position=6)
    TFIELDS = HeaderCard(type_=int, position=7)
    EXTNAME = HeaderCard(required=False, type_=str)


class Column(metaclass=ABCMeta):
    '''
    A column descriptor for columns consisting of a primitive data type
    or fixed shape array.

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
        The resulting data column has `ndim_col = ndim + 1`
    shape: Tuple[int]
        Shape of a single row.
    '''
    def __init__(
        self, *,
        unit=None,
        strict_unit=False,
        required=True,
        name=None,
        ndim=None,
        shape=None,
    ):
        self.required = required
        self.unit = unit
        self.strict_unit = strict_unit
        self.name = name
        self.shape = shape
        self.ndim = ndim

        if self.shape is not None:
            self.shape = tuple(shape)
            # Dimensionality of the table is one more than that of a single row
            if self.ndim is None:
                self.ndim = len(self.shape)
            elif self.ndim != len(self.shape):
                raise ValueError(f'Shape={shape} and ndim={ndim} do not match')
        else:
            # simple column by default
            if self.ndim is None:
                self.ndim = 0

    def __get__(self, instance, owner=None):
        # class attribute access
        if instance is None:
            return self

        return instance.__data__.get(self.name)

    def __set__(self, instance, value):
        instance.__data__[self.name] = value

    def __set_name__(self, owner, name):
        # respect user override for names that are not valid identifiers
        if self.name is None:
            self.name = name

    def __delete__(self, instance):
        '''clear data of this column'''
        if self.name in instance.__data__:
            del instance.__data__[self.name]

    def __repr__(self):
        unit = f'\'{self.unit.to_string("fits")}\'' if self.unit is not None else None
        return (
            f'{self.__class__.__name__}('
            f"name={self.name!r}, required={self.required}, unit={unit}"
            ')'
        )

    @property
    @abstractmethod
    def dtype():
        '''Equivalent numpy dtype'''

    def validate_data(self, data, onerror='raise'):
        ''' Validate the data of this column in table '''
        if data is None:
            if self.required:
                log_or_raise(
                    f'Column {self.name} is required but missing',
                    RequiredMissing, log=log, onerror=onerror
                )
            else:
                return

        # let's test first for the datatype
        try:
            # casting = 'safe' makes sure we don't change values
            # e.g. casting doubles to integers will no longer work
            data = np.asanyarray(data).astype(self.dtype, casting='safe')
        except TypeError as e:
            log_or_raise(
                f'dtype not convertible to column dtype: {e}',
                WrongType, log=log, onerror=onerror
            )

        if self.strict_unit and hasattr(data, 'unit') and data.unit != self.unit:
            log_or_raise(
                f'Unit {data.unit} of data does not match specified unit {self.unit}',
                WrongUnit, log=log, onerror=onerror,
            )

        # a table as one dimension more than it's rows,
        # we also allow a single scalar value for scalar rows
        if data.ndim != self.ndim + 1 and not (data.ndim == 0 and self.ndim == 0):
            log_or_raise(
                f'Dimensionality of rows is {data.ndim - 1}, should be {self.ndim}',
                WrongDims, log=log, onerror=onerror,
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
                f'Shape {shape} does not match required shape {self.shape}',
                WrongShape, log=log, onerror=onerror,
            )

        return q


class BinaryTableMeta(type):
    '''Metaclass for the BinaryTable class'''
    def __new__(cls, name, bases, dct):
        dct['__columns__'] = {}
        dct['__slots__'] = ('__data__', 'header')

        header_schema = dct.get('__header__', None)
        if header_schema is not None and not issubclass(header_schema, BinaryTableHeader):
            raise TypeError(
                '`__header__` must be a class inheriting from `BinaryTableHeader`'
            )

        # collect columns of this new schema
        for k, v in dct.items():
            if isinstance(v, Column):
                k = v.name or k
                dct['__columns__'][k] = v

        new_cls = super().__new__(cls, name, bases, dct)
        return new_cls


class BinaryTable(metaclass=BinaryTableMeta):
    '''
    Schema definition class for a binary table
    '''

    def __init__(self, **column_data):
        self.__data__ = {}
        self.header = fits.Header()

        for k, v in column_data.items():
            setattr(self, k, v)

    def validate_data(self):
        for k, col in self.__columns__.items():
            validated = col.validate_data(self.__data__.get(k))
            if validated is not None:
                setattr(self, k, validated)

    @classmethod
    def validate_hdu(cls, hdu: fits.BinTableHDU, onerror='raise'):
        if not isinstance(hdu, fits.BinTableHDU):
            raise TypeError('hdu is not a BinTableHDU')

        cls.__header__.validate_header(hdu.header, onerror=onerror)
        required = set(c.name for c in cls.__columns__.values() if c.required)
        missing = required - set(c.name for c in hdu.columns)
        if missing:
            log_or_raise(
                f'The following required columns are missing {missing}',
                RequiredMissing,
                log=log,
                onerror=onerror
            )

        table = Table.read(hdu)
        for k, col in cls.__columns__.items():
            if k in table.columns:
                col.validate_data(table[k], onerror=onerror)


class Bool(Column):
    '''A Boolean binary table column'''
    tform_code = 'L'
    dtype = bool


class BitField(Column):
    '''Bitfield binary table column'''
    tform_code = 'X'
    dtype = bool


class Byte(Column):
    '''Byte binary table column'''
    tform_code = 'B'
    dtype = np.uint8


class Int16(Column):
    '''16 Bit signed integer binary table column'''
    tform_code = 'I'
    dtype = np.int16


class Int32(Column):
    '''32 Bit signed integer binary table column'''
    tform_code = 'J'
    dtype = np.int32


class Int64(Column):
    '''64 Bit signed integer binary table column'''
    tform_code = 'K'
    dtype = np.int64


class Char(Column):
    '''Single byte character binary table column'''
    tform_code = 'A'
    dtype = np.dtype('S1')


class Float(Column):
    '''Single precision floating point binary table column'''
    tform_code = 'E'
    dtype = np.float32


class Double(Column):
    '''Single precision floating point binary table column'''
    tform_code = 'D'
    dtype = np.float64


class ComplexFloat(Column):
    '''Single precision complex binary table column'''
    tform_code = 'C'
    dtype = np.csingle


class ComplexDouble(Column):
    '''Single precision complex binary table column'''
    tform_code = 'M'
    dtype = np.cdouble
