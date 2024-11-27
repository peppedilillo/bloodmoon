from contextlib import (
    contextmanager,
)


@contextmanager
def catchtime(
    label: str
) -> Callable[[], float]:
    """A context manager for measuring processing times."""
    t1 = t2 = perf_counter()
    yield lambda: t2 - t1
    t2 = perf_counter()
    print(f"{label} took {t2 - t1:.7}s")
