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
## Installation for developpers
It is highly recommanded to use your own fork in order to follow the individual Pull Requests. Otherwise, the installation process is the following:

```bash
git clone https://github.com/VODF/vodf_schema
cd vodf_schema

conda create -n vodf python=3.13 astropy sphinx 
conda activate vodf
pip install git+https://github.com/VODF/fits_schema
pip install -e .[all]   # installs vodf_schema in edit mode
```

A `pre-commit` system has been setup. To initialise it, use this command
```bash
pip install pre-commit
pre-commit install
pre-commit run
```

## Licence
VODF is licensed under a 3-clause BSD style license - see the `LICENSE <https://github.com/VODF/fits_schema/blob/main/LICENSE>`_  file.

## Supporting the project
The VODF initiative is not sponsored and the development is made by the staff of the institutes supporting the project over their research time. Any contribution is then encouraged, as punctual or regular contributor.
