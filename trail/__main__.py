"""Enable ``python -m trail`` as an alias for the ``trail`` console script."""
from trail.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
