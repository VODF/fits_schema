"""Primary Header Schema."""

from .header import HeaderCard, HeaderSchema


class PrimaryHeader(HeaderSchema):
    """Primary Header Schema."""

    SIMPLE = HeaderCard(allowed_values=[True], position=0)
    BITPIX = HeaderCard(allowed_values=[8, 16, 32, 64, -32, -64], position=1)
    NAXIS = HeaderCard(position=2, allowed_values=range(999))
    EXTEND = HeaderCard(type_=bool, required=False)
    TELESCOP = HeaderCard(type_=str, required=False)
    INSTRUME = HeaderCard(type_=str, required=False)
    OBSERVER = HeaderCard(type_=str, required=False)
    OBJECT = HeaderCard(type_=str, required=False)
    DATE_OBS = HeaderCard(name="DATE-OBS", type_=str, required=False)
