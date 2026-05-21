from capp.config import load_simulation_config


def test_load_simulation_config_parses_generated_support(tmp_path):
    geometry = tmp_path / "part.stl"
    support = tmp_path / "support.stl"
    geometry.write_text("solid part\nendsolid part\n", encoding="utf-8")
    support.write_text("solid support\nendsolid support\n", encoding="utf-8")
    config_path = tmp_path / "simulation.yaml"
    config_path.write_text(
        "\n".join(
            [
                "geometry_path: part.stl",
                "support_geometry_path: support.stl",
                "support_type: Line support",
                "voxel_spacing: 0.5",
                "support_generation:",
                "  support_type: X surface support",
                "  overhang_angle: 55",
                "  pitch: 2.5",
                "  thickness: 0.75",
                "  footprint_offset: 0.25",
                "  build_plate_z: 0",
            ]
        ),
        encoding="utf-8",
    )

    config = load_simulation_config(config_path)

    assert config.support_geometry_path == support.resolve()
    assert config.support_type == "Line support"
    assert config.support_generation is not None
    assert config.support_generation.support_type == "X surface support"
    assert config.support_generation.overhang_angle == 55.0
    assert config.support_generation.pitch == 2.5
    assert config.support_generation.thickness == 0.75
    assert config.support_generation.footprint_offset == 0.25
    assert config.support_generation.build_plate_z == 0.0
