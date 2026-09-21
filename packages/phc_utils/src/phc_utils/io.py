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
