"""
src package
============
This file makes the "src" folder a proper Python package, so we can write
things like:

    from src.encoding import rate_encode
    from src.reservoir import LSMReservoir

instead of messy relative imports. It is intentionally left mostly empty --
its only job is to exist.
"""
