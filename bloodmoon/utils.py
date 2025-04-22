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
        Exception: The timer raises the encoutered exception
                   within the algorithm and prints its name.
    """
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
        mess = f"    # Finished '{name}' in {end_time - start_time:.3f}s"
        print(mess + ".\n" if not error else mess + f" with {error}.\n")


# end