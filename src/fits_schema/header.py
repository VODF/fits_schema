"""
Schema definitions for FITS Headers.

See section 4 of the FITS Standard:
https://fits.gsfc.nasa.gov/standard40/fits_standard40aa-le.pdf
"""

import logging
import re
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date, datetime
from typing import Self

from astropy.io import fits

from .exceptions import (
    AdditionalHeaderCard,
    RequiredMissing,
    WrongPosition,
    WrongType,
    WrongValue,
)
from .schema_element import SchemaElement
from .utils import log_or_raise

__all__ = ["HeaderSchema", "HeaderCard"]

log = logging.getLogger(__name__)


HEADER_ALLOWED_TYPES = (str, bool, int, float, complex, date, datetime)
TABLE_KEYWORDS = {"TTYPE", "TUNIT", "TFORM", "TSCAL", "TZERO", "TDISP", "TDIM"}
IGNORE = TABLE_KEYWORDS


@dataclass
class HeaderCard(SchemaElement):
    """Schema for the entry of a FITS header."""

    allowed_values: Iterable | None = None
    position: int | None = None
    type_: type | tuple[type] | None = None
    empty: bool | None = None
    case_insensitive: bool = True

    def __post_init__(self):
        """Validate the header schema."""
        # super().__post_init__()

        vals = self.allowed_values
        if vals is not None:
            if not isinstance(vals, Iterable) or isinstance(vals, str):
                vals = {vals}
            else:
                vals = set(vals)

            if self.case_insensitive:
                vals = set(v.upper() if isinstance(v, str) else v for v in vals)

            if not all(isinstance(v, HEADER_ALLOWED_TYPES) for v in vals):
                raise ValueError(f"Values must be instances of {HEADER_ALLOWED_TYPES}")

        if self.type_ is not None:
            if isinstance(self.type_, Iterable):
                self.type_ = tuple(set(self.type_))

            # check that value and type match if both supplied
            if vals is not None:
                if any(not isinstance(v, self.type_) for v in vals):
                    raise TypeError(
                        f"The type of `allowed_values` ({self.allowed_values}) "
                        f"and `type_` ({self.type_}) do not agree."
                    )
        else:
            # if only value is supplied, deduce type from value
            if vals is not None:
                self.type_ = tuple(set(type(v) for v in vals))

        self.allowed_values = vals

        if self.name:
            self._check_name()

    def _check_name(self):
        """Ensure card name follows FITS conventions."""
        if self.name is not None:
            if len(self.name) > 8:
                raise ValueError("FITS header keywords must be 8 characters or shorter")

            if not re.match(r"^[A-Z0-9\-_]{1,8}$", self.name):
                raise ValueError(
                    "FITS header keywords must only contain"
                    " ascii uppercase, digit, _ or -"
                )

    def validate(self, card, pos, onerror="raise"):
        """Validate an astropy.io.fits.card.Card."""
        valid = True
        k = self.name

        if self.position is not None and self.position != pos:
            valid = False
            msg = f"Expected card {k} at position {self.position} but found at {pos}"
            log_or_raise(msg, WrongPosition, log, onerror=onerror)

        if self.type_ is not None and not isinstance(card.value, self.type_):
            valid = False
            msg = (
                f"Header keyword {k} has wrong type {type(card.value)}"
                f", expected one of {self.type_}"
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


class HeaderSchemaMeta(type):
    """Metaclass for HeaderSchema."""

    def __new__(cls, name, bases, dct):
        """Instantiate and check a HeaderSchema."""
        dct["__cards__"] = {}
        dct["__slots__"] = tuple()

        for base in reversed(bases):
            if issubclass(base, HeaderSchema):
                dct["__cards__"].update(base.__cards__)

        for k, v in dct.items():
            if isinstance(v, HeaderCard):
                k = v.name or k  # use user override for keyword if there
                dct["__cards__"][k] = v

        new_cls = super().__new__(cls, name, bases, dct)
        return new_cls


class HeaderSchema(metaclass=HeaderSchemaMeta):
    """
    Schema definition for the header of a FITS HDU.

    To be added as ``class __header__(HeaderSchema)`` to HDU schema classes.

    Add `HeaderCard` class members to define the schema.


    Examples
    --------
    >>> from fits_schema.header import HeaderSchema, HeaderCard
    >>>
    >>> class MyHeaderSchema(HeaderSchema):
    ...     FOO = HeaderCard(required=True, type_=int)
    ...     BAR = HeaderCard(required=True, type_=str)  # doctest: +SKIP
    """

    @classmethod
    def _header_schemas(cls) -> list[Self]:
        """Return a list of HeaderSchema parents."""
        return list(
            reversed(
                [
                    cls,
                ]
                + [base for base in cls.__bases__ if issubclass(base, HeaderSchema)]
            )
        )

    @classmethod
    def grouped_cards(cls) -> dict[Self, list[HeaderCard]]:
        """Return a list of cards grouped by parent HeaderSchema class."""
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
