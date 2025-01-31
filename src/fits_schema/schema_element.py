#!/usr/bin/env python3

"""Defines common schema metadata inherited by Column and HeaderCard."""

from abc import ABCMeta

from astropy.io.votable.ucd import check_ucd
from astropy.units import Unit

from .exceptions import SchemaError

# collects all references used, useful for generating a citation list.
_REFERENCE_SET = set()


class SchemaElement(metaclass=ABCMeta):
    """Base for Schema elements defining common metadata keys."""

    def __init__(
        self,
        description: str = "",
        required: bool = True,
        unit: Unit | None = None,
        examples: list[str] | None = None,
        reference: str | None = None,
        ucd: str | None = None,
        ivoa_name: str | None = None,
    ):
        """Initialize HeaderSchema.

        Properties
        ----------
        description:
            human-readable description
        required:
            if this item is optional, set to false
        unit:
            The unit associated with the element. For Columns, this will be verified if set,
            for HeaderCards, it is only for documentation purposes.

        Examples
        --------
            list of examples, for documentation
        reference:
            Citation for the origin of this attribute.
        ucd:
            IVOA uniform content descriptor string
        ivoa_name:
            If this element is associated with an IVOA data model, provide the
            ``ModelName.keyword``, for example ``ObsCore.obs_publisher_did``
        """
        self.description = description
        self.required = required
        self.unit = unit
        self.examples = examples
        self.reference = reference
        self.ucd = ucd
        self.ivoa_name = ivoa_name

        if not isinstance(self.description, str):
            raise SchemaError("description should be a string")

        if self.unit:
            self.unit = Unit(self.unit)

        if self.examples and not isinstance(self.examples, list):
            raise SchemaError("examples should be a list of strings")

        if self.examples is None:
            self.examples = []

        if self.ucd:
            if check_ucd(self.ucd, check_controlled_vocabulary=True) is False:
                raise SchemaError(f"UCD '{self.ucd}' is not valid")

        if self.reference:
            if not isinstance(self.reference, str):
                raise SchemaError("Reference should be a string")
            else:
                _REFERENCE_SET.add(self.reference)
