import logging

import pytest

from fits_schema.exceptions import AdditionalHeaderCard


def test_log_or_raise(caplog):
    from fits_schema.utils import log_or_raise

    log = logging.getLogger("test_log_or_raise")

    with pytest.raises(ValueError):
        log_or_raise("Foo", ValueError, log=log, onerror="raise")

    with pytest.raises(TypeError):
        log_or_raise("Foo", TypeError, log=log, onerror="raise")

    with pytest.warns(AdditionalHeaderCard):
        log_or_raise("Foo", AdditionalHeaderCard, log=log, onerror="raise")

    caplog.clear()
    log_or_raise("Foo", ValueError, log=log, onerror="log")
    assert caplog.record_tuples[0] == ("test_log_or_raise", logging.ERROR, "Foo")

    caplog.clear()
    log_or_raise("Foo", AdditionalHeaderCard, log=log, onerror="log")
    assert caplog.record_tuples[0] == ("test_log_or_raise", logging.WARNING, "Foo")

    with pytest.raises(ValueError):
        log_or_raise("Foo", TypeError, log=log, onerror="invalid")
