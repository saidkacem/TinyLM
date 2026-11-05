"""Toddler LLM: a tiny language model built from scratch, one step at a time."""
from importlib.util import find_spec

_missing = [name for name in ("numpy", "pandas", "matplotlib") if find_spec(name) is None]
if _missing:
    raise ModuleNotFoundError(
        f"{', '.join(_missing)} missing: this notebook runs on the wrong Python. At the top right of the notebook, "
        "click the kernel name, then Select Another Kernel > Python Environments > .venv, and run the cells again."
    )
