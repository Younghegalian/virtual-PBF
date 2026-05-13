from pathlib import Path

from capp.data.matlab import MatFileKind, inspect_mat_file


def test_inspect_mat_file_identifies_matlab_73(tmp_path: Path):
    path = tmp_path / "sample.mat"
    path.write_bytes(b"MATLAB 7.3 MAT-file, Platform: PCWIN64".ljust(128, b" "))

    info = inspect_mat_file(path)

    assert info.kind == MatFileKind.MATLAB_73_HDF5

