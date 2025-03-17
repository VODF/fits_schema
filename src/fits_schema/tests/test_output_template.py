#!/usr/bin/env python3

"""Check output to FITS templates."""


def test_output_hdu():
    """Checks BinTable, Header, HeaderCard, and Column outputs."""

    from fits_schema import (
        BinaryTable,
        BinaryTableHeader,
        Double,
        HeaderCard,
        Int64,
        String,
    )
    from fits_schema.output_template import bintable_template

    class TestTable(BinaryTable):
        """An example binary table."""

        class __header__(BinaryTableHeader):
            EXTNAME = HeaderCard(allowed_values="TEST_EXT")
            EXTRA = HeaderCard(description="Extra info, not in the base class")

        ENERGY = Double(
            unit="TeV",
            description="Estimated Energy",
            ucd="em.energy",
            display_format="F5.2",
        )
        EVENT_ID = Int64(description="Event identifier, unique within an observation")
        STRING_COL = String()

    template_lines = list(bintable_template(TestTable))

    assert len(template_lines) > 10
