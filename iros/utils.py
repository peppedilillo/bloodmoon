from contextlib import contextmanager
from time import perf_counter
from typing import Callable


@contextmanager
def catchtime(label: str, write: Callable = print) -> Callable[[], float]:
    """A context manager for measuring processing times."""
    t1 = t2 = perf_counter()
    yield lambda: t2 - t1
    t2 = perf_counter()
    write(f"{label} took {t2 - t1:.7}s")
