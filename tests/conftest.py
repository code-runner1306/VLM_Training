import sys
from pathlib import Path

# Insert root directory to sys.path for pytest discovery
root_dir = str(Path(__file__).resolve().parent.parent)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
