"""Makes the repository root importable when pytest runs the suite in-place."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
