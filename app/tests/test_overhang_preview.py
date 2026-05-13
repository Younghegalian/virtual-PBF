import numpy as np

from capp.workbench.preview import compute_overhang_angles


def test_overhang_angles_match_matlab_definition():
    normals = np.array(
        [
            [0.0, 0.0, -1.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    angles = compute_overhang_angles(normals)

    np.testing.assert_allclose(angles, [0.0, 90.0, 180.0], atol=1e-5)


def test_overhang_angles_are_scale_invariant():
    normals = np.array([[0.0, 0.0, -4.0], [3.0, 0.0, 0.0]], dtype=np.float32)

    angles = compute_overhang_angles(normals)

    np.testing.assert_allclose(angles, [0.0, 90.0], atol=1e-5)
