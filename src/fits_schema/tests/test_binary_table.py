import astropy.units as u
import numpy as np
import pytest
from astropy.io import fits
from astropy.table import Column, Table

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
    table = TestTable(Table({"test": [1, 2, 3]}))
    table.validate_data()
    assert (table.test == u.Quantity([1, 2, 3], u.m)).all()

    # convertible unit
    table = TestTable(Table(dict(test=[1, 2, 3] * u.cm)))
    table.validate_data()

    with pytest.raises(WrongUnit):
        table = TestTable(Table(dict(test=[5] * u.deg)))

    # validate no unit is enforced:
    class TestTable(BinaryTable):
        test = Double(unit=u.dimensionless_unscaled)

    table = TestTable(Table(dict(test=[1, 2, 3])))
    table.validate_data()

    with pytest.raises(WrongUnit):
        table = TestTable(Table(dict(test=[1, 2, 3] * u.deg)))

    # test strict_unit
    class TestTable(BinaryTable):
        test = Double(unit=u.m, strict_unit=True)

    with pytest.raises(WrongUnit):
        table = TestTable(Table(dict(test=[1, 2, 3] * u.cm)))


def test_repr():
    from fits_schema.binary_table import BinaryTable, Double

    class TestTable(BinaryTable):
        test = Double()

    assert repr(TestTable.test) == "Double(name='test', required=True, unit=None)"

    TestTable.test.unit = u.m
    assert repr(TestTable.test) == "Double(name='test', required=True, unit='m')"

    TestTable.test.unit = u.m**-2
    assert repr(TestTable.test) == "Double(name='test', required=True, unit='m-2')"


def test_access():
    from fits_schema.binary_table import BinaryTable, Double

    class TestTable(BinaryTable):
        test = Double(required=False)

    t = TestTable(Table({}))
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
    with pytest.raises(WrongDims):
        TestTable(Table({"test": [3.14]}))

    # three numbers per row, should be ten
    with pytest.raises(WrongShape):
        TestTable(Table({"test": [[1, 2, 3]]}))

    # this should work
    rng = np.random.default_rng(1337)
    TestTable(Table({"test": [np.arange(10), rng.normal(size=10)]}))


def test_ndim():
    from fits_schema.binary_table import BinaryTable, Double

    class TestTable(BinaryTable):
        test = Double(ndim=2)

    # single numbers
    with pytest.raises(WrongDims):
        TestTable(Table({"test": [1, 2, 3]}))

    # 1d
    with pytest.raises(WrongDims):
        TestTable(
            Table(
                dict(
                    test=[
                        [1, 2, 3],
                        [4, 5, 6],
                    ]
                )
            )
        )

    # each row is 2d, fits
    TestTable(
        Table(
            dict(
                test=[
                    np.random.default_rng().normal(size=(5, 3)),
                    np.random.default_rng().normal(size=(5, 3)),
                ]
            )
        )
    )

    # 3d not
    with pytest.raises(WrongDims):
        TestTable(Table(dict(test=[np.zeros((2, 2, 2)), np.ones((2, 2, 2))])))

    class TestTable(BinaryTable):
        test = Double()

    # check a single number is allowed for normal columns
    TestTable(Table(dict(test=[5])))


def test_required():
    from fits_schema.binary_table import BinaryTable, Bool

    class TestTable(BinaryTable):
        test = Bool(required=True)

    with pytest.raises(RequiredMissing):
        TestTable(Table({}))

    TestTable(Table(dict(test=[True, False])))


def test_data_types():
    from fits_schema.binary_table import BinaryTable, Int16

    class TestTable(BinaryTable):
        test = Int16(required=False)

    table = TestTable(Table({}))

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


@pytest.fixture
def events_file(tmp_path):
    energy = [0.1, 0.2, 0.3] * u.TeV
    event_id = np.arange(len(energy))
    table = Table({"event_id": event_id, "energy": energy})

    hdu = fits.BinTableHDU(table, name="EVENTS")
    hdu.header["OBS_ID"] = 1

    hdul = fits.HDUList([fits.PrimaryHDU(), hdu])
    path = tmp_path / "events.fits.gz"
    hdul.writeto(path)
    return path


def test_data(events_file):
    from fits_schema.binary_table import BinaryTable, BinaryTableHeader, Double, Int64
    from fits_schema.header import HeaderCard

    class EventsTable(BinaryTable):
        event_id = Int64()
        energy = Double(unit=u.TeV)

        class __header__(BinaryTableHeader):
            OBS_ID = HeaderCard(type_=int)

    with fits.open(events_file) as hdul:
        events = EventsTable(hdul["EVENTS"])

    assert isinstance(events.event_id, Column)
    assert isinstance(events.energy, Column)
    assert len(events.event_id) == 3
    assert events.energy.unit == u.TeV
