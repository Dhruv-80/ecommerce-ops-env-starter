import sys
from pathlib import Path


# Ensure repository root is importable in CI so `server.*` imports resolve.
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
