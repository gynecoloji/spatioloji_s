Contributing
============

Contributions are welcome! Here's how to get started.

Development setup
-----------------

.. code-block:: bash

   git clone https://github.com/gynecoloji/spatioloji_s.git
   cd spatioloji_s
   pip install -e ".[test]"

Running tests
-------------

.. code-block:: bash

   pytest tests/ -v

Code style
----------

- **Linter/formatter**: ruff (line length 120)
- **Docstrings**: Google-style with Args, Returns, Raises, Example sections
- **Type hints**: Required on all public functions
- **Imports**: stdlib, third-party, internal (ruff isort order)
- **Optional deps**: Guard with ``try/except ImportError``

.. code-block:: bash

   ruff check src/ tests/ --fix
   ruff format src/ tests/

Git workflow
------------

- Branch naming: ``feature/<desc>``, ``fix/<desc>``, ``docs/<desc>``
- Commit messages: conventional commits format
- Never commit directly to main
- All tests must pass before merging

Adding a new analysis function
------------------------------

1. Add the function to the appropriate module under ``src/spatioloji_s/``
2. Add Google-style docstring with Args, Returns, Raises, Example
3. Add type hints on all parameters and return type
4. Write pytest tests in ``tests/unit/``
5. Export from the module's ``__init__.py``
6. Update visualization if the function produces plottable results
