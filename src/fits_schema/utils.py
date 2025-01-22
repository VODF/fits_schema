import logging
import warnings

from .exceptions import ValidationError

log = logging.getLogger(__name__)


def log_or_raise(msg, exc_type=ValidationError, log=log, onerror="raise"):
    """Utility for error handling.
    onerror decides if a validation error or warnign is raised (either as exception)
    or as warning, depending on `exception_type`
    or if it is just logged.
    """
    # raise if exception, warn if warning
    if onerror == "raise":
        if issubclass(exc_type, UserWarning):
            warnings.warn(msg, exc_type)
        else:
            raise exc_type(msg)
    # only log stuff
    elif onerror == "log":
        if issubclass(exc_type, UserWarning):
            log.warning(msg)
        else:
            log.error(msg)
    # invalid arg to `onerror`
    else:
        raise ValueError('`onerror` must be either "raise" or "log"')
