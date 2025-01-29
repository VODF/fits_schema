"""
Schema definitions for FITS Headers.

See section 4 of the FITS Standard:
https://fits.gsfc.nasa.gov/standard40/fits_standard40aa-le.pdf
"""

import logging
import re
from collections.abc import Iterable
from datetime import date, datetime
from typing import Self

from astropy.io import fits
from astropy.units import Unit

from .exceptions import (
    AdditionalHeaderCard,
    RequiredMissing,
    SchemaError,
    WrongPosition,
    WrongType,
    WrongValue,
)
from .schema_element import SchemaElement
from .utils import log_or_raise

__all__ = ["Header", "HeaderCard"]

log = logging.getLogger(__name__)


HEADER_ALLOWED_TYPES = (str, bool, int, float, complex, date, datetime)
TABLE_KEYWORDS = {"TTYPE", "TUNIT", "TFORM", "TSCAL", "TZERO", "TDISP", "TDIM"}
IGNORE = TABLE_KEYWORDS


class HeaderCard(SchemaElement):
    """
    Schema for the entry of a FITS header.

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
    """

    def __init__(
        self,
        keyword=None,
        *,
        required: bool = True,
        allowed_values=None,
        position=None,
        type_=None,
        empty=None,
        case_insensitive=True,
        description: str = "",
        unit: Unit | None = None,
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

        self.keyword = None
        if keyword is not None:
            self.__set_name__(None, keyword)

        self.allowed_values = allowed_values
        self.position = position
        self.empty = empty
        self.case_insensitive = case_insensitive

        vals = allowed_values
        if vals is not None:
            if not isinstance(vals, Iterable) or isinstance(vals, str):
                vals = {vals}
            else:
                vals = set(vals)

            if self.case_insensitive:
                vals = set(v.upper() if isinstance(v, str) else v for v in vals)

            if not all(isinstance(v, HEADER_ALLOWED_TYPES) for v in vals):
                raise SchemaError(f"Values must be instances of {HEADER_ALLOWED_TYPES}")

        self.type = type_
        if type_ is not None:
            if isinstance(type_, Iterable):
                self.type = tuple(set(type_))

            # check that value and type match if both supplied
            if vals is not None:
                if any(not isinstance(v, type_) for v in vals):
                    raise SchemaError(
                        f"The type of `allowed_values` ({self.allowed_values}) "
                        f"and `type` ({self.type}) do not agree."
                    )
        else:
            # if only value is supplied, deduce type from value
            if vals is not None:
                self.type = tuple(set(type(v) for v in vals))

        self.allowed_values = vals

    def __set_name__(self, owner, name):
        """Rename to keyword if existing and check name style."""
        if self.keyword is None:
            if len(name) > 8:
                raise SchemaError(
                    "FITS header keywords must be 8 characters or shorter"
                )

            if not re.match(r"^[A-Z0-9\-_]{1,8}$", name):
                raise SchemaError(
                    "FITS header keywords must only contain"
                    " ascii uppercase, digit, _ or -"
                )
            self.keyword = name

    def __get__(self, instance: "None | Header", owner: "None | HeaderMeta" = None):
        """Delegate to the actual fits header for getting value."""
        # accessed as class attribute
        if instance is None:
            return self
        return instance._header[self.keyword]

    def __set__(self, instance: "Header", value):
        """Delegate to the actual fits header for setting value."""
        # TODO: need to validate here before setting
        # but current validate only works after value has been set
        instance._header[self.keyword] = value

    def validate(self, card, pos, onerror="raise"):
        """Validate an astropy.io.fits.card.Card."""
        valid = True
        k = self.keyword

        if self.position is not None and self.position != pos:
            valid = False
            msg = f"Expected card {k} at position {self.position} but found at {pos}"
            log_or_raise(msg, WrongPosition, log, onerror=onerror)

        if self.type is not None and not isinstance(card.value, self.type):
            valid = False
            msg = (
                f"Header keyword {k} has wrong type {type(card.value)}"
                f", expected one of {self.type}"
            )
            log_or_raise(msg, WrongType, log, onerror=onerror)

        if self.allowed_values is not None:
            if self.case_insensitive and isinstance(card.value, str):
                val = card.value.upper()
            else:
                val = card.value
            if val not in self.allowed_values:
                log_or_raise(
                    f"Possible values for {k!r} are {self.allowed_values}"
                    f", found {card.value!r}",
                    WrongValue,
                    log,
                    onerror=onerror,
                )

        has_value = not (card.value is None or isinstance(card.value, fits.Undefined))
        if self.empty is True and has_value:
            log_or_raise(
                f"Card {k} is required to be empty but has value {card.value}",
                WrongValue,
                log,
                onerror,
            )

        if self.empty is False and not has_value:
            log_or_raise(f"Card {k} exists but has no value", WrongValue, log, onerror)

        return valid


class HeaderMeta(type):
    """Metaclass for Header."""

    def __new__(cls, name, bases, dct):
        """Instantiate and check a Header."""
        dct["__cards__"] = {}
        dct["__slots__"] = ("_header",)

        for base in reversed(bases):
            if issubclass(base, Header):
                dct["__cards__"].update(base.__cards__)

        for k, v in dct.items():
            if isinstance(v, HeaderCard):
                k = v.keyword or k  # use user override for keyword if there
                dct["__cards__"][k] = v

        new_cls = super().__new__(cls, name, bases, dct)
        return new_cls


class Header(metaclass=HeaderMeta):
    """
    Schema definition for the header of a FITS HDU.

    To be added as ``class __header__(Header)`` to HDU schema classes.

    Add `HeaderCard` class members to define the schema.


    Examples
    --------
    >>> from fits_schema.header import Header, HeaderCard
    >>>
    >>> class MyHeader(Header):
    ...     FOO = HeaderCard(required=True, type_=int)
    ...     BAR = HeaderCard(required=True, type_=str)  # doctest: +SKIP
    """

    def __init__(self, header):
        self._header = header

    @classmethod
    def _header_schemas(cls) -> list[Self]:
        """Return a list of Header parents."""
        return list(
            reversed(
                [
                    cls,
                ]
                + [base for base in cls.__bases__ if issubclass(base, Header)]
            )
        )

    @classmethod
    def grouped_cards(cls) -> dict[Self, list[HeaderCard]]:
        """Return a list of cards grouped by parent Header class."""
        seen = set()
        group = {}

        for schema in cls._header_schemas():
            group[schema] = {k: c for k, c in schema.__cards__.items() if k not in seen}
            seen.update(schema.__cards__.keys())

        return group

    @classmethod
    def validate_header(cls, header: fits.Header, onerror="raise"):
        """Check a header against the schema."""
        required = {k for k, c in cls.__cards__.items() if c.required}
        missing = required - set(header.keys())

        # first let's test for any missing required keys
        if missing:
            log_or_raise(
                f"Header is missing the following required keywords: {missing}",
                RequiredMissing,
                log=log,
                onerror=onerror,
            )

        # no go through each of the header items and validate them with the schema
        for pos, card in enumerate(header.cards):
            kw = card.keyword
            if kw not in cls.__cards__:
                if kw.rstrip("0123456789") not in IGNORE:
                    log_or_raise(
                        f'Unexpected header card "{str(card).strip()}"',
                        AdditionalHeaderCard,
                        log=log,
                        onerror=onerror,
                    )
                continue

            cls.__cards__[card.keyword].validate(card, pos, onerror)

    @classmethod
    def update(cls, other_schema):
        """Update cards."""
        cls.__cards__.update(other_schema.__cards__)
