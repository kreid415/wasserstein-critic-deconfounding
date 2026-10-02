"""Put tests/prereg (synthetic helpers) and scripts/ (rule code) on sys.path."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
for p in (HERE, os.path.join(os.path.dirname(os.path.dirname(HERE)), "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)
