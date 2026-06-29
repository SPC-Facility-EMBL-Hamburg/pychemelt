Installation
============

Requirements
------------

PyChemelt requires Python 3.12 or later and the following packages:

* numpy
* pandas
* scipy
* xlrd
* openpyxl

Install for Users
-----------------

The recommended way to install PyChemelt is with ``uv``:

.. code-block:: bash

    uv add pychemelt

If you are not using ``uv``, install PyChemelt with ``pip``:

.. code-block:: bash

    pip install pychemelt


Install from Source - Development
---------------------------------
Clone the repository and install in development mode with ``uv``:

.. code-block:: bash

    git clone https://github.com/osvalB/pychemelt.git
    cd pychemelt
    uv sync --extra dev

Verify Installation
-------------------

By running the tests:

.. code-block:: bash

    uv run pytest

By creating the documentation:

.. code-block:: bash

    uv run build_docs.py
