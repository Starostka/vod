import functools
import time
import typing as typ

from loguru import logger


class Chrono:
    """A simple chronometer."""

    _laps: list[tuple[float, float]]
    _start_time: None | float

    def __init__(self, buffer_size: int = 100) -> None:
        self._laps = []
        self._start_time = None
        self.buffer_size = buffer_size

    def reset(self) -> "Chrono":
        """Reset the chrono."""
        self._laps = []
        self._start_time = None
        return self

    def start(self) -> "Chrono":
        """Start the chrono."""
        self._start_time = time.perf_counter()
        return self

    def stop(self) -> "Chrono":
        """Stop the chrono."""
        if self._start_time is None:
            raise RuntimeError("Chrono is not running")
        curr_time = time.perf_counter()
        self._laps.append((self._start_time, curr_time))
        if len(self._laps) > self.buffer_size:
            self._laps.pop(0)
        self._start_time = None
        return self

    def get_total_time(self) -> float:
        """Return the total time elapsed since the chrono was started."""
        return sum(end - start for start, end in self._laps)

    def get_avg_time(self) -> float:
        """Return the average time elapsed since the chrono was started."""
        return self.get_total_time() / len(self._laps)

    def get_avg_laps_per_second(self) -> float:
        """Return the average number of laps per second."""
        return len(self._laps) / self.get_total_time()


T = typ.TypeVar("T")
P = typ.ParamSpec("P")


def log_exec_time_decorator(fn: typ.Callable[P, T]) -> typ.Callable[P, T]:
    """A wrapper to log the elapsed time of a function using loguru."""

    @functools.wraps(fn)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        chrono = Chrono()
        chrono.start()
        output = fn(*args, **kwargs)
        chrono.stop()
        logger.debug(f"{fn.__name__} - took {chrono.get_avg_time():.3f}s")
        return output

    return wrapper
