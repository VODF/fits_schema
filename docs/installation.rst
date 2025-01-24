==============
 Installation
==============

User Installation
=================

As a user, install from pypi:

.. code-block:: shell

    $ pip install fits_schema


Developer Setup
===============

As a developer, clone the repository, create a virtual environment
and then install the package in development mode:

.. code-block:: shell

   $ git clone git@github.com/VODF/fits_schema
   $ cd fits_schema
   $ python -m venv venv
   $ source venv/bin/activate
   $ pip install -e .[test,doc,dev]

The same also works with conda, create a conda env instead of a venv above.
