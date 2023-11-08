import pathlib
import shutil
import typing as typ

import loguru
from lightning_utilities.core.rank_zero import rank_zero_only


class CacheManager:
    """A context manager that creates a temporary directory and deletes it when exiting the context."""

    path: pathlib.Path
    delete_existing: bool
    persist: bool
    __slots__ = ("path", "delete_existing", "persist")

    def __init__(self, path: str | pathlib.Path, *, delete_existing: bool = False, persist: bool = False):
        self.delete_existing = delete_existing
        self.persist = persist
        self.path = pathlib.Path(path)

    def __enter__(self) -> pathlib.Path:
        """Create the temporary directory and return its path."""
        self._create(self.path)
        return self.path

    def __exit__(self, *args: typ.Any, **kwargs: typ.Any) -> None:
        """Cleanup the temporary directory."""
        if not self.persist:
            self._cleanup(self.path)

    @rank_zero_only
    def _create(self, path: pathlib.Path) -> None:
        if self.delete_existing and path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)

    @rank_zero_only
    def _cleanup(self, path: pathlib.Path) -> None:
        for f in path.iterdir():
            loguru.logger.info(f"Deleting file ({f.stat().st_size / 1e6:.2f} MB): `{f}`")
            if f.is_dir():
                shutil.rmtree(f)
            else:
                f.unlink()
        shutil.rmtree(path)
