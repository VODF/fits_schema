#!/usr/bin/env python3

"""Defines common schema metadata inherited by Column and HeaderCard."""

from abc import ABCMeta, abstractmethod
from dataclasses import dataclass

from astropy.io.votable.ucd import check_ucd
from astropy.units import Unit

# collects all references used, useful for generating a citation list.
_REFERENCE_SET = set()


@dataclass
class SchemaElement(metaclass=ABCMeta):
    """Base for Schema elements defining common metadata keys."""

    #: name to use if not the attribute name, useful in case the column or
    #: header has a non-python-allowed name, i.e. with a ``-`` or starting with a
    #: number
    name: str | None = None

    #: human-readable description
    description: str = ""

    #: if this item is optional, set to false
    required: bool = True

    #: The unit associated with the element. For Columns, this will be verified if set,
    #: for HeaderCards, it is only for documentation purposes.
    unit: Unit | None = None

    #: list of examples, for documentation
    examples: list[str] | None = None

    #: Citation for the origin of this attribute.
    reference: str | None = None

    #: IVOA uniform content descriptor string
    ucd: str | None = None

    def __post_init__(self):
        """Check that the metadata keywords are as expected"""
        if not isinstance(self.description, str):
            raise ValueError("description should be a string")

        if self.unit:
            self.unit = Unit(self.unit)

        if self.examples and not isinstance(self.examples, list):
            raise ValueError("examples should be a list of strings")

        if self.examples is None:
            self.examples = []

        if self.ucd:
            if check_ucd(self.ucd, check_controlled_vocabulary=True) is False:
                raise ValueError(f"UCD '{self.ucd}' is not valid")

        if self.reference:
            if not isinstance(self.reference, str):
                raise ValueError("Reference should be a string")
            else:
                _REFERENCE_SET.add(self.reference)

    def __set_name__(self, owner, name):
        """Rename self.name to the class variable if it is not specified."""
        if self.name is None:
            self.name = name
        self._check_name()

    @abstractmethod
    def _check_name(self):
        """Check if the name is valid."""
        pass
