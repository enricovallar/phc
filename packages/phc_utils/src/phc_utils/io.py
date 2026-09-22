from pathlib import Path
from typing import Any


def export_gds(
    component: Any,
    filepath: str | Path,
    overwrite: bool = True,
) -> Path:
    """Safely exports a GDSII component to a binary file on disk.

    Handles parent directory creation, atomic or safe overwriting of existing files,
    and returns the resolved file path.

    Args:
        component: A gdsfactory Component (or any object with a write_gds method).
        filepath: Destination path for the .gds file.
        overwrite: If True, replaces an existing file at the destination path.
            If False and the file exists, raises FileExistsError.

    Returns:
        Resolved Path object pointing to the written GDSII file.

    Raises:
        FileExistsError: If destination file exists and overwrite is False.
        TypeError: If component does not have a write_gds method.
        RuntimeError: If the underlying layout engine fails to write the file.
    """
    path = Path(filepath).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        if not overwrite:
            raise FileExistsError(
                f"GDS file already exists at '{path}' and overwrite=False."
            )
        try:
            path.unlink()
        except OSError as e:
            raise RuntimeError(
                f"Failed to remove existing GDS file '{path}': {e}"
            ) from e

    if not hasattr(component, "write_gds"):
        raise TypeError(
            f"Expected a layout Component with write_gds method, got {type(component).__name__}."
        )

    try:
        component.write_gds(str(path))
    except Exception as e:
        raise RuntimeError(f"Error exporting GDSII to '{path}': {e}") from e

    if not path.is_file():
        raise RuntimeError(
            f"write_gds completed without error, but '{path}' was not found on disk."
        )

    return path


import os
import sys
from collections.abc import Generator
from contextlib import contextmanager


@contextmanager
def silence_c_stdout() -> Generator[None, None, None]:
    """Redirects low-level C file descriptor 1 (stdout) and 2 (stderr) to os.devnull.

    Silences underlying compiled C/Fortran extension libraries (such as MPB and Meep)
    that write directly to STDOUT_FILENO, allowing real-time Python terminal progress bars
    (like tqdm) to render cleanly without line break disruptions.
    """
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    saved_stdout = os.dup(1)
    saved_stderr = os.dup(2)
    try:
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(devnull_fd, 1)
        os.dup2(devnull_fd, 2)
        yield
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(saved_stdout, 1)
        os.dup2(saved_stderr, 2)
        os.close(saved_stdout)
        os.close(saved_stderr)
        os.close(devnull_fd)
