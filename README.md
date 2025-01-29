# fits-schema
[![CI](https://github.com/VODF/fits_schema/actions/workflows/ci.yml/badge.svg)](https://github.com/VODF/fits_schema/actions/workflows/ci.yml)
[![Quality Gate Status](https://sonar-cta-dpps.zeuthen.desy.de/api/project_badges/measure?project=VODF_fits_schema_AZSNmvSdAfB1AuE28SGr&metric=alert_status&token=sqb_fa08e2cf40384be0a05c433de526b8d744e2ae30)](https://sonar-cta-dpps.zeuthen.desy.de/dashboard?id=VODF_fits_schema_AZSNmvSdAfB1AuE28SGr)
[![Coverage](https://sonar-cta-dpps.zeuthen.desy.de/api/project_badges/measure?project=VODF_fits_schema_AZSNmvSdAfB1AuE28SGr&metric=coverage&token=sqb_fa08e2cf40384be0a05c433de526b8d744e2ae30)](https://sonar-cta-dpps.zeuthen.desy.de/dashboard?id=VODF_fits_schema_AZSNmvSdAfB1AuE28SGr)
[![PyPI version](https://badge.fury.io/py/fits-schema.svg)](https://badge.fury.io/py/fits-schema)



A python package to define and validate schemata for FITS files.


```python
from fits_schema.binary_table import BinaryTable, Double
from fits_schema.header import Header, HeaderCard
import astropy.units as u
from astropy.io import fits


class Events(BinaryTable):
    '''A Binary Table of Events'''
    energy = Double(unit=u.TeV)
    ra     = Double(unit=u.deg)
    dec    = Double(unit=u.deg)

    class __header__(Header):
        EXTNAME = HeaderCard(allowed_values='events')


hdulist = fits.open('events.fits')
Events.validate_hdu(hdulist['events'])
```
