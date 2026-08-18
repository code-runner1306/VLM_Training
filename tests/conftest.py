import sys
from pathlib import Path

# Insert root directory to sys.path for pytest discovery
root_dir = str(Path(__file__).resolve().parent.parent)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

# Import transformers.Trainer before any module that imports `peft`.
# On Windows, importing `peft` before `transformers.Trainer` can crash with a
# native DLL access violation (bitsandbytes/cuBLAS load-order conflict).
try:
    from transformers import Trainer, TrainingArguments  # noqa: F401
except ImportError:
    pass
