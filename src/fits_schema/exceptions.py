"""Exceptions raised during validation."""


class ValidationError(ValueError):
    """Base class for all exceptions raised by ``fits_schema``."""


class RequiredMissing(ValidationError):
    """Raised when a required element is missing."""


class WrongUnit(ValidationError):
    """Raised when an element does not have the correct unit."""


class WrongDims(ValidationError):
    """Raised when an element does not have the correct dimensions."""


class WrongShape(ValidationError):
    """Raised when an element does not have the correct shape."""


class WrongType(ValidationError):
    """Raised when an element does not have the correct data type."""


class WrongPosition(ValidationError):
    """Raised when an element is not at the correct position."""


class WrongKeyword(ValidationError):
    """Raised when a HeaderCards is validated and keywords don't match."""


class WrongValue(ValidationError):
    """Raised when a HeaderCards has not the right value."""


class AdditionalHeaderCard(UserWarning):
    """Issued when a header has a card not mentioned in the schema."""

class SchemaError(ValueError):
    """Errors in Schema Definition, not data validation."""
