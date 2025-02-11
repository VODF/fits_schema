#!/usr/bin/env python3

"""Code to convert fits schemas into FITS TPL files.

This is mainly a test of introspection and documentation, but can be helpful for
use in fits tools that can read the TPL format.
"""

from collections.abc import Generator, Iterable
from textwrap import wrap

from astropy import units as u

from .binary_table import BinaryTable, Column
from .header import Header, HeaderCard
from .schema_element import SchemaElement

SINGLE_LINE = "".join(["/ ", "-" * 87])
DOUBLE_LINE = "".join(["/ ", "=" * 87])


def unit_str(element: SchemaElement):
    """Convert unit to comment string."""
    if not element.unit:
        return ""
    return f"[{u.Unit(element.unit.to_string('fits'))}] "


def type_str(element: SchemaElement):
    """Convert type to comment string."""
    # Note this will be much simpler when we have a proper type class hierarchy
    if not element.type:
        return ""

    # WHy not just make all type values to be tuples? Having to support both
    # scalar and tuple is annoying.
    if isinstance(element.type, Iterable):
        return f"({','.join(x.__name__ for x in element.type)}) "
    else:
        return f"({element.type.__name__}) "


def header_card_template(card: HeaderCard) -> Generator[str]:
    """Output FITS template lines for a HeaderCard."""
    optional = " optional " if card.required is False else ""
    value = card.allowed_values.copy().pop() if card.allowed_values else ""

    yield f"{card.keyword.upper():8s} = {value:<10} / {unit_str(card)}{optional}{type_str(card)}{card.description}"


def header_template(
    header: Header, exclude: set[Header] | None = None
) -> Generator[str]:
    """Output FITS template lines for a Header.

    Parameters
    ----------
    exclude : set[Header] | None
        header groups to exclude from the template
    """
    if exclude:
        exclude = set(exclude)
    else:
        exclude = set()

    for group, cards in header.grouped_cards().items():
        if group in exclude or len(cards) == 0:
            continue

        group_name = group.__name__
        if group_name == "__header__":
            group_name = "Headers specific to this HDU"

        if group.__doc__:
            yield from wrap(
                f"{group_name}: {group.__doc__}",
                initial_indent="/    ",
                subsequent_indent="/    ",
                drop_whitespace=True,
            )
        else:
            yield f"/    {group_name}:"

        for card in cards.values():
            yield from header_card_template(card)

        yield ""


def column_template(column: Column) -> Generator[str]:
    """Output FITS template lines for a Column."""
    yield f"TTYPE# = {column.name:20s} / {column.description:.70s}"

    if not hasattr(column, "tform_code"):
        raise AttributeError(f"Missing tform_code for column {column}")

    yield f"TFORM# = {column.tform_code:20s} / {column.dtype.__name__}"

    if column.unit:
        yield (
            f"TUNIT# = {u.Unit(column.unit).to_string('fits'):20s}"
            f" / or convertible to '{u.Unit(column.unit).physical_type}'"
        )
    if column.ucd:
        yield f"TUCD#  = {column.ucd:20s}"
    if column.ndim:
        yield f"TDIM#  = {column.ndim:<20d} / value is a {column.ndim}-dimensional array"
    if column.display_format:
        yield f"TDISP#  = {column.display_format:20s} / display format (FORTRAN-style)"
    yield ""  # spacer


def bintable_template(hdu: BinaryTable) -> Generator[str]:
    """Output FITS template lines for a full HDU."""
    yield DOUBLE_LINE
    yield f"/ HDU: {hdu.__name__}"  # should put in the EXTNAME here...
    yield "/ DESCRIPTION: "
    yield from wrap(
        hdu.__doc__,
        initial_indent="/    ",
        subsequent_indent="/    ",
        drop_whitespace=True,
    )

    yield SINGLE_LINE
    yield "/ HEADERS: "
    yield SINGLE_LINE
    yield ""

    # output the headers, but no the BinaryTableHeaders ones, which we want to
    # treat specially
    yield from header_template(hdu.__header__)  # exclude=[BinaryTableHeader])

    yield SINGLE_LINE
    yield "/ COLUMNS: "
    yield SINGLE_LINE
    yield ""

    for column in hdu.__columns__.values():
        yield from column_template(column)

    yield "END"
