#!/usr/bin/env python3


def test_headers_and_cols_with_metadata():
    from astropy import units as u

    from fits_schema import Header, HeaderCard

    class ReferencePosition(Header):
        """Reference position of the observatory, for time and coordinates."""

        TREFPOS = HeaderCard(
            description="Code for the spatial location at which the observation time is valid",
            reference="FITS Standard, v4",
            type_=str,
            allowed_values=["TOPOCENTER"],
        )

        OBSGEO_B = HeaderCard(
            keyword="OBSGEO-B",
            description="the latitude of the observation, with North positive",
            type_=float,
            unit=u.deg,
            ucd="pos.earth.lat",
            reference="FITS Standard, v4",
        )
        OBSGEO_L = HeaderCard(
            keyword="OBSGEO-L",
            description="the longitide of the observation, with East positive",
            type_=float,
            unit=u.deg,
            ucd="pos.earth.lon",
            reference="FITS Standard, v4",
        )
        OBSGEO_H = HeaderCard(
            keyword="OBSGEO-H",
            description="the altitude of the observation",
            type_=float,
            unit=u.m,
            ucd="pos.earth.altitude",
            reference="FITS Standard, v4",
        )
