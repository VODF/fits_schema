'''
Schema definitions for FITS Headers.

See section 4 of the FITS Standard:
https://fits.gsfc.nasa.gov/standard40/fits_standard40aa-le.pdf
'''
from datetime import date, datetime
from typing import Self
from .exceptions import (
    RequiredMissing, WrongType, WrongPosition, AdditionalHeaderCard,
    WrongValue,
)
import re
from collections.abc import Iterable
from astropy.io import fits
import logging

from .utils import log_or_raise


log = logging.getLogger(__name__)


HEADER_ALLOWED_TYPES = (str, bool, int, float, complex, date, datetime)
TABLE_KEYWORDS = {'TTYPE', 'TUNIT', 'TFORM', 'TSCAL', 'TZERO', 'TDISP', 'TDIM'}
IGNORE = TABLE_KEYWORDS


class HeaderCard:
    '''
    Schema for the entry of a FITS header

    Attributes
    ----------
    keyword: str
        override the keyword given as the class member name,
        useful to define keywords containing hyphens or starting with numbers
    required: bool
        If this card is required
    allowed_values: instance of any in ``HEADER_ALLOWED_TYPES`` or iterable of that
        If specified, card must have on of these values
    position: int or None
        if not None, the card must be at this position in the header,
        starting with the first card at 0
    type: one or a tuple of the types in ``HEADER_ALLOWED_TYPES``
    empty: True, False or None
        If True, value must be empty, if False must not be empty,
        if None, no check if a value is present is performed
    case_insensitive: True
        match str values case insensitively
    '''
    def __init__(
        self, keyword=None, description: str = "", *,required=True,
        allowed_values=None, position=None, type_=None,
        empty=None, case_insensitive=True,
    ):
        self.keyword = None
        if keyword is not None:
            self.__set_name__(None, keyword)

        self.required = required
        self.position = position
        self.empty = empty
        self.case_insensitive = case_insensitive
        self.description = description

        vals = allowed_values
        if vals is not None:
            if not isinstance(vals, Iterable) or isinstance(vals, str):
                vals = {vals}
            else:
                vals = set(vals)

            if self.case_insensitive:
                vals = set(v.upper() if isinstance(v, str) else v for v in vals)

            if not all(isinstance(v, HEADER_ALLOWED_TYPES) for v in vals):
                raise ValueError(f'Values must be instances of {HEADER_ALLOWED_TYPES}')

        self.type = type_
        if type_ is not None:
            if isinstance(type_, Iterable):
                self.type = tuple(set(type_))

            # check that value and type match if both supplied
            if vals is not None:
                if any(not isinstance(v, type_) for v in vals):
                    raise TypeError(f'`values` must be of type `type_`({type_}) or None')
        else:
            # if only value is supplied, deduce type from value
            if vals is not None:
                self.type = tuple(set(type(v) for v in vals))

        self.allowed_values = vals

    def __set_name__(self, owner, name):
        if self.keyword is None:
            if len(name) > 8:
                raise ValueError('FITS header keywords must be 8 characters or shorter')

            if not re.match(r'^[A-Z0-9\-_]{1,8}$', name):
                raise ValueError(
                    'FITS header keywords must only contain'
                    ' ascii uppercase, digit, _ or -'
                )
            self.keyword = name

    def validate(self, card, pos, onerror='raise'):
        '''Validate an astropy.io.fits.card.Card'''
        valid = True
        k = self.keyword

        if self.position is not None and self.position != pos:
            valid = False
            msg = (
                f'Expected card {k} at position {self.position}'
                f' but found at {pos}'
            )
            log_or_raise(msg, WrongPosition, log, onerror=onerror)

        if self.type is not None and not isinstance(card.value, self.type):
            valid = False
            msg = (
                f'Header keyword {k} has wrong type {type(card.value)}'
                f', expected one of {self.type}'
            )
            log_or_raise(msg, WrongType, log, onerror=onerror)

        if self.allowed_values is not None:
            if self.case_insensitive and isinstance(card.value, str):
                val = card.value.upper()
            else:
                val = card.value
            if val not in self.allowed_values:
                log_or_raise(
                    f'Possible values for {k!r} are {self.allowed_values}'
                    f', found {card.value!r}',
                    WrongValue,
                    log,
                    onerror=onerror
                )

        has_value = not (card.value is None or isinstance(card.value, fits.Undefined))
        if self.empty is True and has_value:
            log_or_raise(
                f'Card {k} is required to be empty but has value {card.value}',
                WrongValue, log, onerror,
            )

        if self.empty is False and not has_value:
            log_or_raise(f'Card {k} exists but has no value', WrongValue, log, onerror)

        return valid


class HeaderSchemaMeta(type):
    def __new__(cls, name, bases, dct):
        dct['__cards__'] = {}
        dct['__slots__'] = tuple()

        for base in reversed(bases):
            if issubclass(base, HeaderSchema):
                dct['__cards__'].update(base.__cards__)

        for k, v in dct.items():
            if isinstance(v, HeaderCard):
                k = v.keyword or k  # use user override for keyword if there
                dct['__cards__'][k] = v

        new_cls = super().__new__(cls, name, bases, dct)
        return new_cls


class HeaderSchema(metaclass=HeaderSchemaMeta):
    '''
    Schema definition for the header of a FITS HDU

    To be added as `class __header_schema__(HeaderSchema)` to HDU schema classes.

    Add `Card` class members to define the schema.


    Example
    -------
    >>> from fits_schema.binary_table import BinaryTable, Int32
    >>> from fits_schema.header import HeaderSchema, HeaderCard
    >>>
    >>> class Events(BinaryTable):
    ...    EVENT_ID = Int32()
    ...
    ...    class __header_schema__(HeaderSchema):
    ...        HDUCLASS = HeaderCard(required=True, allowed_values="Events")
    '''

    @classmethod
    def _header_schemas(cls) -> list[Self]:
        """returns a list of HeaderSchema parents"""
        return list(reversed([cls,] + [base for base in cls.__bases__ if issubclass(base, HeaderSchema)]))

    @classmethod
    def grouped_cards(self) -> dict[Self,HeaderCard]:
        """Return a list of cards grouped by parent HeaderSchema class"""
        seen = set()
        group = {}

        for schema in self._header_schemas():
            group[schema] = {k: c for k,c in schema.__cards__.items() if k not in seen}
            seen.update(schema.__cards__.keys())

        return group



    @classmethod
    def validate_header(cls, header: fits.Header, onerror='raise'):

        required = {k for k, c in cls.__cards__.items() if c.required}
        missing = required - set(header.keys())

        # first let's test for any missing required keys
        if missing:
            log_or_raise(
                f"Header is missing the following required keywords: {missing}",
                RequiredMissing, log=log, onerror=onerror,
            )

        # no go through each of the header items and validate them with the schema
        for pos, card in enumerate(header.cards):
            kw = card.keyword
            if kw not in cls.__cards__:
                if kw.rstrip('0123456789') not in IGNORE:
                    log_or_raise(
                        f'Unexpected header card "{str(card).strip()}"',
                        AdditionalHeaderCard,
                        log=log,
                        onerror=onerror
                    )
                continue

            cls.__cards__[card.keyword].validate(card, pos, onerror)

    @classmethod
    def update(cls, other_schema):
        cls.__cards__.update(other_schema.__cards__)
