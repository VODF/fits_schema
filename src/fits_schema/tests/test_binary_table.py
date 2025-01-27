import astropy.units as u
import numpy as np
import pytest
from astropy.io import fits
from astropy.table import Table

from fits_schema.exceptions import (
    RequiredMissing,
    WrongDims,
    WrongShape,
    WrongType,
    WrongUnit,
)


def test_unit():
    from fits_schema.binary_table import BinaryTable, Double

    class TestTable(BinaryTable):
        test = Double(unit=u.m)

    # allow no units
    table = TestTable(test=[1, 2, 3])
    table.validate_data()
    assert (table.test == u.Quantity([1, 2, 3], u.m)).all()

    # convertible unit
    table = TestTable(test=[1, 2, 3] * u.cm)
    table.validate_data()

    table = TestTable(test=5 * u.deg)
    with pytest.raises(WrongUnit):
        table.validate_data()

    # validate no unit is enforced:
    class TestTable(BinaryTable):
        test = Double(unit=u.dimensionless_unscaled)

    table = TestTable(test=[1, 2, 3])
    table.validate_data()

    table = TestTable(test=[1, 2, 3] * u.deg)
    with pytest.raises(WrongUnit):
        table.validate_data()

    # test strict_unit
    class TestTable(BinaryTable):
        test = Double(unit=u.m, strict_unit=True)

    table = TestTable(test=[1, 2, 3] * u.cm)
    with pytest.raises(WrongUnit):
        table.validate_data()


def test_repr():
    from fits_schema.binary_table import BinaryTable, Double

    class TestTable(BinaryTable):
        test = Double()

    assert (
        repr(TestTable.test)
        == "Double(name='test', description='', required=True, unit=None, examples=[], reference=None, ucd=None, ivoa_name=None, strict_unit=False, ndim=0, shape=None, dtype=<class 'numpy.float64'>, tform_code='D')"
    )

    TestTable.test.unit = u.m
    assert (
        repr(TestTable.test)
        == "Double(name='test', description='', required=True, unit=Unit(\"m\"), examples=[], reference=None, ucd=None, ivoa_name=None, strict_unit=False, ndim=0, shape=None, dtype=<class 'numpy.float64'>, tform_code='D')"
    )

    TestTable.test.unit = u.m**-2
    assert (
        repr(TestTable.test)
        == "Double(name='test', description='', required=True, unit=Unit(\"1 / m2\"), examples=[], reference=None, ucd=None, ivoa_name=None, strict_unit=False, ndim=0, shape=None, dtype=<class 'numpy.float64'>, tform_code='D')"
    )


def test_access():
    from fits_schema.binary_table import BinaryTable, Double

    class TestTable(BinaryTable):
        test = Double(required=False)

    t = TestTable()
    assert t.test is None
    t.test = [5.0]
    assert t.test[0] == 5.0

    # assignment does not validate
    t.test == ["foo"]

    del t.test
    assert t.test is None

    # test that we can only assign to columns
    with pytest.raises(AttributeError):
        t.foo = "bar"


def test_shape():
    from fits_schema.binary_table import BinaryTable, Double

    class TestTable(BinaryTable):
        test = Double(shape=(10,))

    # single number, wrong number of dimensions
    table = TestTable(test=3.14)
    with pytest.raises(WrongDims):
        table.validate_data()

    # three numbers per row, should be ten
    table = TestTable(test=[[1, 2, 3]])
    with pytest.raises(WrongShape):
        table.validate_data()

    # this should work
    rng = np.random.default_rng(1337)
    table = TestTable(test=[np.arange(10), rng.normal(size=10)])
    table.validate_data()


def test_ndim():
    from fits_schema.binary_table import BinaryTable, Double

    class TestTable(BinaryTable):
        test = Double(ndim=2)

    # single numbers
    table = TestTable(test=[1, 2, 3])
    with pytest.raises(WrongDims):
        table.validate_data()

    # 1d
    table = TestTable(
        test=[
            [1, 2, 3],
            [4, 5, 6],
        ]
    )
    with pytest.raises(WrongDims):
        table.validate_data()

    # each row is 2d, fits
    table = TestTable(
        test=[
            np.random.default_rng().normal(size=(5, 3)),
            np.random.default_rng().normal(size=(5, 3)),
        ]
    )
    table.validate_data()

    # 3d not
    table = TestTable(test=[np.zeros((2, 2, 2)), np.ones((2, 2, 2))])
    with pytest.raises(WrongDims):
        table.validate_data()

    class TestTable(BinaryTable):
        test = Double()

    # check a single number is allowed for normal columns
    table = TestTable(test=5)
    table.validate_data()


def test_required():
    from fits_schema.binary_table import BinaryTable, Bool

    class TestTable(BinaryTable):
        test = Bool(required=True)

    table = TestTable()

    with pytest.raises(RequiredMissing):
        table.validate_data()

    table.test = [True, False]
    table.validate_data()


def test_data_types():
    from fits_schema.binary_table import BinaryTable, Int16

    class TestTable(BinaryTable):
        test = Int16(required=False)

    table = TestTable()

    # check no data is ok, as column is optional
    assert table.validate_data() is None

    # integer is ok
    table.test = np.array([1, 2, 3], dtype=np.int16)
    table.validate_data()

    # double would loose information
    table.test = np.array([1.0, 2.0, 3.0])
    with pytest.raises(WrongType):
        table.validate_data()

    # too large for int16
    table.test = np.array([1**15, 2**15, 3**15])
    with pytest.raises(WrongType):
        table.validate_data()


def test_inheritance():
    from fits_schema.binary_table import BinaryTable, Bool

    class BaseTable(BinaryTable):
        foo = Bool()
        bar = Bool()

    class TestTable(BaseTable):
        # make sure it's different from base definition
        bar = Bool(required=not BaseTable.foo.required)
        baz = Bool()

    assert list(TestTable.__columns__) == ["foo", "bar", "baz"]
    assert TestTable.bar.required != BaseTable.bar.required


def test_validate_hdu():
    from fits_schema.binary_table import BinaryTable, Double

    class TestTable(BinaryTable):
        energy = Double(unit=u.TeV)
        ra = Double(unit=u.deg)
        dec = Double(unit=u.deg)

    data = {
        "energy": 10 ** np.random.default_rng().uniform(-1, 2, 100) * u.TeV,
        "ra": np.random.default_rng().uniform(0, 360, 100) * u.deg,
    }

    # make sure a correct table passes validation
    t = Table(data)
    hdu = fits.BinTableHDU(t)
    with pytest.raises(RequiredMissing):
        TestTable.validate_hdu(hdu)

    t["dec"] = np.random.default_rng().uniform(0, 360, 100) * u.TeV
    hdu = fits.BinTableHDU(t)

    with pytest.raises(WrongUnit):
        TestTable.validate_hdu(hdu)

    t["dec"] = np.random.default_rng().uniform(0, 360, 100) * u.deg
    hdu = fits.BinTableHDU(t)
    TestTable.validate_hdu(hdu)


def test_optional_columns():
    from fits_schema.binary_table import BinaryTable, Double

    class TestTable(BinaryTable):
        energy = Double(unit=u.TeV)
        ra = Double(unit=u.deg)
        dec = Double(unit=u.deg, required=False)

    data = {
        "energy": 10 ** np.random.default_rng().uniform(-1, 2, 100) * u.TeV,
        "ra": np.random.default_rng().uniform(0, 360, 100) * u.deg,
    }

    # make sure a correct table passes validation
    t = Table(data)
    hdu = fits.BinTableHDU(t)
    TestTable.validate_hdu(hdu)


def test_header_not_schema():
    from fits_schema.binary_table import BinaryTable

    with pytest.raises(TypeError):

        class Table(BinaryTable):
            # should inherit from HeaderSchema
            class __header__:
                pass


def test_header():
    from fits_schema.binary_table import BinaryTable, BinaryTableHeader, Double
    from fits_schema.header import HeaderCard

    class TestTable(BinaryTable):
        energy = Double(unit=u.TeV)

        class __header__(BinaryTableHeader):
            TEST = HeaderCard(type_=str)

    t = Table({"energy": [1, 2, 3] * u.TeV})
    hdu = fits.BinTableHDU(t)

    with pytest.raises(RequiredMissing):
        TestTable.validate_hdu(hdu)

    t.meta["TEST"] = 5
    hdu = fits.BinTableHDU(t)
    with pytest.raises(WrongType):
        TestTable.validate_hdu(hdu)

    t.meta["TEST"] = "hello"
    hdu = fits.BinTableHDU(t)
    TestTable.validate_hdu(hdu)


def test_column_name():
    from fits_schema import Int32

    # ensure that upper, lower, underscore, and numbers match
    Int32(name="UPPERLower_Thing0123456")

    # check that adding other characters fails:

    with pytest.warns(UserWarning):
        Int32(name="this-should-fail")

    with pytest.warns(UserWarning):
        Int32(name="Has Spaces")
