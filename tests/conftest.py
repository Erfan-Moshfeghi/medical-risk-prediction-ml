"""Configure the test environment.

This file ensures that the `src` package is discoverable when running tests
from the repository root.  Pytest automatically loads this file before
collecting tests.
"""

import sys
from pathlib import Path

# Append the src directory to sys.path so that `import src` works during tests
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Add the project root to sys.path so that `import src` resolves correctly.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))