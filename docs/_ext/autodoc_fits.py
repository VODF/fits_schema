"""Sphinx Directives for documenting FITS Schema elements."""

from __future__ import annotations

from typing import Any, override

from sphinx.application import Sphinx
from sphinx.ext.autodoc import ClassDocumenter
from sphinx.util.typing import ExtensionMetadata

from fits_schema import Header


class FITSHeaderDocumenter(ClassDocumenter):
    """Document a fits Header class."""

    objtype = "fitsheader"
    directivetype = ClassDocumenter.objtype
    priority = 10 + ClassDocumenter.priority
    option_spec = dict(ClassDocumenter.option_spec)

    @classmethod
    def can_document_member(
        cls, member: Any, membername: str, isattr: bool, parent: Any
    ) -> bool:
        """Check if we can document this."""
        try:
            return issubclass(member, Header)
        except TypeError:
            return False

    @override
    def add_content(self, more_content: list[str] | None) -> None:
        super().add_content(more_content)
        source_name = self.get_sourcename()
        header: Header = self.object

        self.add_line("", source_name)

        for name, card in header.__cards__.items():
            self.add_line(
                f"* **{name}**: {card.description} {card.ucd} {card.unit}", source_name
            )


def setup(app: Sphinx) -> ExtensionMetadata:
    """Register extensions."""
    app.setup_extension("sphinx.ext.autodoc")  # Require autodoc extension
    app.add_autodocumenter(FITSHeaderDocumenter)
    return {
        "version": "1",
        "parallel_read_safe": True,
    }
