"""The script PyInstaller freezes (spec 19.3 step 5).

PyInstaller's command line takes a script path and has no equivalent of ``python -m spells``,
so the packaged build needs a file to point at. This is that file and nothing else: the two
lines of ``src/spells/__main__.py``, kept in build/ rather than in the package so the frozen
executable and ``python -m spells`` stay one entry point apart.
"""

from spells.app import main

raise SystemExit(main())
