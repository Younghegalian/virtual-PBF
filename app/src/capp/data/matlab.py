from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path


class MatFileKind(StrEnum):
    MATLAB_73_HDF5 = "matlab_7_3_hdf5"
    MATLAB_50 = "matlab_5_0"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class MatFileInfo:
    path: Path
    kind: MatFileKind
    header: str
    size_bytes: int


def inspect_mat_file(path: str | Path) -> MatFileInfo:
    mat_path = Path(path)
    with mat_path.open("rb") as handle:
        header_bytes = handle.read(128)

    header = header_bytes.decode("ascii", errors="ignore").strip("\x00 ")
    if header.startswith("MATLAB 7.3 MAT-file"):
        kind = MatFileKind.MATLAB_73_HDF5
    elif header.startswith("MATLAB 5.0 MAT-file"):
        kind = MatFileKind.MATLAB_50
    else:
        kind = MatFileKind.UNKNOWN

    return MatFileInfo(
        path=mat_path,
        kind=kind,
        header=header,
        size_bytes=mat_path.stat().st_size,
    )

