"""
General utility functions for the bloodmoon package.
"""

from typing import Generator, Any
from contextlib import contextmanager
import time
from datetime import datetime

__all__ = [
    "timer",
]


@contextmanager
def timer(name: str) -> Generator[None, Any, None]:
    """
    Timer context manager.

    Args:
        name (str): Name of the task the timer is handling.
    
    Raises:
        Exception: Encoutered exception within the algorithm (if any).
    """
    def handle_time(start: float, end: float) -> str:
        """Handles output time format."""
        time_interval = end - start
        h, rem = divmod(time_interval, 3600)
        m, s = divmod(rem, 60)
        return f"{int(h):02}h:{int(m):02}m:{s:.3f}s"

    start_time = time.perf_counter()
    error = False
    print(f"    # Starting '{name}' at {datetime.now():%H:%M:%S}.")
    try:
        yield
    except Exception as e:
        error = type(e).__name__
        raise
    finally:
        end_time = time.perf_counter()
        elapsed_time = handle_time(start_time, end_time)
        mess = f"    # Finished '{name}' in {elapsed_time}"
        print(mess + ".\n" if not error else mess + f" with {error}.\n")


# end