"""Toddler LLM: a tiny language model built from scratch, one step at a time."""

try:
    import numpy  # noqa: F401
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        "numpy is missing: this notebook runs on the wrong Python. At the top right of the notebook, "
        "click the kernel name, then Select Another Kernel > Python Environments > .venv, and run the cells again."
    ) from None
