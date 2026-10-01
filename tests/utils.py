"""Test utilities for creating authentic NIfTI-1 test files and directory fixtures

without generating synthetic patient data.
"""

from pathlib import Path
import struct
import gzip
from typing import Tuple, Optional


def create_minimal_nifti1_bytes(
    dimensions: Tuple[int, int, int] = (64, 64, 32),
    voxel_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    data_bytes: Optional[bytes] = None
) -> bytes:
    """Creates a strictly valid 348-byte NIfTI-1 header + binary payload.

    Conforms to the NIfTI-1 binary data format specification.
    """
    nx, ny, nz = dimensions
    dx, dy, dz = voxel_spacing

    # 348 bytes header buffer
    hdr = bytearray(348)

    # 1. sizeof_hdr (offset 0, 4 bytes int32 = 348)
    struct.pack_into("<i", hdr, 0, 348)

    # 2. dim (offset 40, 8 short ints = [3, nx, ny, nz, 1, 1, 1, 1])
    struct.pack_into("<8h", hdr, 40, 3, nx, ny, nz, 1, 1, 1, 1)

    # 3. datatype & bitpix (offset 70, short int = 16 (FLOAT32), offset 72 = 32)
    struct.pack_into("<h", hdr, 70, 16)
    struct.pack_into("<h", hdr, 72, 32)

    # 4. pixdim (offset 76, 8 floats: [0.0, dx, dy, dz, 0.0, 0.0, 0.0, 0.0])
    struct.pack_into("<8f", hdr, 76, 0.0, dx, dy, dz, 0.0, 0.0, 0.0, 0.0)

    # 5. vox_offset (offset 108, float32 = 352.0)
    struct.pack_into("<f", hdr, 108, 352.0)

    # 6. sform_code (offset 254, short = 1)
    struct.pack_into("<h", hdr, 254, 1)

    # 7. srow_x, srow_y, srow_z (offset 280, 12 floats)
    struct.pack_into(
        "<12f", hdr, 280,
        dx, 0.0, 0.0, 0.0,
        0.0, dy, 0.0, 0.0,
        0.0, 0.0, dz, 0.0
    )

    # 8. magic (offset 344, 4 bytes = b"n+1\0")
    hdr[344:348] = b"n+1\0"

    # 4 bytes padding to reach offset 352
    padding = b"\x00\x00\x00\x00"

    # Payload
    num_voxels = nx * ny * nz
    if data_bytes is None:
        payload = b"\x00" * (num_voxels * 4)  # zero floats
    else:
        payload = data_bytes

    return bytes(hdr) + padding + payload


def write_test_nifti(
    file_path: Path,
    dimensions: Tuple[int, int, int] = (64, 64, 32),
    voxel_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    data_bytes: Optional[bytes] = None
) -> Path:
    """Writes a valid NIfTI file (.nii or .nii.gz) to disk."""
    raw = create_minimal_nifti1_bytes(dimensions, voxel_spacing, data_bytes)
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.name.endswith(".gz"):
        with gzip.open(path, "wb") as f:
            f.write(raw)
    else:
        with open(path, "wb") as f:
            f.write(raw)

    return path
