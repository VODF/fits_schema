import pytest
from fits_schema.exceptions import (
    RequiredMissing, WrongValue, WrongPosition, WrongType,
)
from astropy.io import fits


def test_length():
    from fits_schema.header import HeaderCard, HeaderSchema

    with pytest.raises((ValueError, RuntimeError)):
        class LengthHeader(HeaderSchema):
            MORE_THAN_8 = HeaderCard()

    with pytest.raises((ValueError,RuntimeError)):
        class LowerHeader(HeaderSchema):
            lowercas = HeaderCard()

    class DateHeader(HeaderSchema):
        DATE_OBS = HeaderCard(keyword='DATE-OBS')

    assert 'DATE-OBS' in DateHeader.__cards__


def test_primary():
    from fits_schema.primary import PrimaryHeader

    hdu = fits.PrimaryHDU()
    PrimaryHeader.validate_header(hdu.header)


def test_position():
    from fits_schema.primary import PrimaryHeader
    h = fits.Header()
    h['BITPIX'] = 16
    h['SIMPLE'] = True
    h['NAXIS'] = 0

    with pytest.raises(WrongPosition):
        PrimaryHeader.validate_header(h)


def test_required():
    from fits_schema.primary import PrimaryHeader
    h = fits.Header()
    h['SIMPLE'] = True
    h['BITPIX'] = 16

    # NAXIS is required but missing
    with pytest.raises(RequiredMissing):
        PrimaryHeader.validate_header(h)


def test_wrong_value():
    from fits_schema.primary import PrimaryHeader
    h = fits.Header()
    h['SIMPLE'] = False
    h['BITPIX'] = 16
    h['NAXIS'] = 0

    # SIMPLE must be True
    with pytest.raises(WrongValue):
        PrimaryHeader.validate_header(h)


def test_type():
    from fits_schema.header import HeaderSchema, HeaderCard
    h = fits.Header()

    class Header(HeaderSchema):
        TEST = HeaderCard(type_=str)

    h['TEST'] = 5
    with pytest.raises(WrongType):
        Header.validate_header(h)

    h['TEST'] = 'hello'
    Header.validate_header(h)

    class Header(HeaderSchema):
        TEST = HeaderCard(type_=[str, int])

    h['TEST'] = 'hello'
    Header.validate_header(h)
    h['TEST'] = 5
    Header.validate_header(h)

    h['TEST'] = 5.5
    with pytest.raises(WrongType):
        Header.validate_header(h)


def test_empty():
    from fits_schema.header import HeaderSchema, HeaderCard
    h = fits.Header()

    class Header(HeaderSchema):
        TEST = HeaderCard(empty=True)

    h['TEST'] = None
    Header.validate_header(h)

    h['TEST'] = fits.Undefined()
    Header.validate_header(h)

    h['TEST'] = 'something'
    with pytest.raises(WrongValue):
        Header.validate_header(h)

    class Header(HeaderSchema):
        TEST = HeaderCard(empty=False)

    h['TEST'] = None
    with pytest.raises(WrongValue):
        Header.validate_header(h)

    h['TEST'] = 'foo'
    Header.validate_header(h)


def test_inheritance():
    from fits_schema.header import HeaderSchema, HeaderCard

    class BaseHeader(HeaderSchema):
        FOO = HeaderCard()
        BAR = HeaderCard(type_=str)

    class Header(BaseHeader):
        BAR = HeaderCard(type_=int)

    assert set(Header.__cards__) == set(BaseHeader.__cards__)
    assert BaseHeader.BAR.type == str
    assert Header.BAR.type == int


def test_invalid_arguments():
    from fits_schema.header import HeaderCard

    with pytest.raises(ValueError):
        HeaderCard(allowed_values=[(1, 2, 3)])

    with pytest.raises(TypeError):
        # allowed values does not match allowed type
        HeaderCard(type_=str, allowed_values=1)


def test_additional():
    from fits_schema.header import HeaderSchema, HeaderCard
    from fits_schema.exceptions import AdditionalHeaderCard

    class Header(HeaderSchema):
        TEST = HeaderCard()

    h = fits.Header()
    h['TEST'] = 'foo'
    h['FOO'] = 'bar'

    with pytest.warns(AdditionalHeaderCard):
        Header.validate_header(h)


def test_case():
    from fits_schema.header import HeaderSchema, HeaderCard

    class Header(HeaderSchema):
        TEST = HeaderCard(allowed_values={'foo'})

    h = fits.Header()
    h['TEST'] = 'foo'
    Header.validate_header(h)

    h['TEST'] = 'Foo'
    Header.validate_header(h)

    h['TEST'] = 'FOO'
    Header.validate_header(h)

    class Header(HeaderSchema):
        TEST = HeaderCard(allowed_values={'foo'}, case_insensitive=False)

    h['TEST'] = 'Foo'
    with pytest.raises(WrongValue):
        Header.validate_header(h)
