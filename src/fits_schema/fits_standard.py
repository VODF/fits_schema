#!/usr/bin/env python3

"""Definitions from the FITS standard, for schema validation."""

from .header import HeaderCard, HeaderSchema

__all__ = ["FITSStandardHeaders"]


class FITSStandardHeaders(HeaderSchema):
    """FITS standard headers.

    Optional Headers defined by the `FITS Standard v4.0
    <https://fits.gsfc.nasa.gov/standard40/fits_standard40aa-le.pdf>`.
    """

    EXTEND = HeaderCard(type_=bool, required=False)
    ORIGIN = HeaderCard(type_=str, required=False)
    TELESCOP = HeaderCard(type_=str, required=False)
    INSTRUME = HeaderCard(type_=str, required=False)
    OBSERVER = HeaderCard(type_=str, required=False)
    OBJECT = HeaderCard(type_=str, required=False)
    AUTHOR = HeaderCard(type_=str, required=False)
    REFERENC = HeaderCard(type_=str, required=False)
    COMMENT = HeaderCard(type_=str, required=False)
    HISTORY = HeaderCard(type_=str, required=False)
    CREATOR = HeaderCard(type_=str, required=False)
    PROGRAM = HeaderCard(type_=str, required=False)

    # see table 22 of https://fits.gsfc.nasa.gov/standard40/fits_standard40aa-le.pdf
    DATE = HeaderCard(type_=str, required=False)
    DATE_OBS = HeaderCard(name="DATE-OBS", type_=str, required=False)
    DATE_BEG = HeaderCard(name="DATE-BEG", type_=str, required=False)
    DATE_AVG = HeaderCard(name="DATE-AVG", type_=str, required=False)
    DATE_END = HeaderCard(name="DATE-END", type_=str, required=False)

    MJD_OBS = HeaderCard(name="MJD-OBS", type_=float, required=False)
    MJD_BEG = HeaderCard(name="MJD-BEG", type_=float, required=False)
    MJD_AVG = HeaderCard(name="MJD-AVG", type_=float, required=False)
    MJD_END = HeaderCard(name="MJD-END", type_=float, required=False)

    # time definition
    MJDREF = HeaderCard(type_=float, required=False)
    MJDREFI = HeaderCard(type_=int, required=False)
    MJDREFF = HeaderCard(type_=float, required=False)
    JDREF = HeaderCard(type_=float, required=False)
    JDREFI = HeaderCard(type_=int, required=False)
    JDREFF = HeaderCard(type_=float, required=False)
    DATEREF = HeaderCard(type_=float, required=False)

    # start / end time relative to reference time
    TSTART = HeaderCard(type_=float, required=False)
    TSTOP = HeaderCard(type_=float, required=False)
