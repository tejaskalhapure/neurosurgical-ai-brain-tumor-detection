"""Unit tests for Pydantic request and response schemas."""

from backend.app.schemas.inference import BoundingBox3D, VolumetricMetrics


def test_bounding_box_schema():
    """Validates 3D bounding box coordinate representation."""
    bbox = BoundingBox3D(
        x_min=10, x_max=50,
        y_min=20, y_max=60,
        z_min=30, z_max=70
    )
    assert bbox.x_max > bbox.x_min
    assert bbox.y_max > bbox.y_min
    assert bbox.z_max > bbox.z_min


def test_volumetric_metrics_schema():
    """Validates volumetric estimation schema."""
    vol = VolumetricMetrics(
        estimated_volume_mm3=15400.0,
        estimated_volume_cm3=15.4,
        voxel_count=15400,
        max_slice_area_mm2=640.0,
        max_slice_index=42,
        valid_voxel_metadata=True
    )
    assert vol.estimated_volume_cm3 == 15.4
    assert vol.valid_voxel_metadata is True
