"""NIfTI file handler implementation and interface.

Responsible for inspecting voxel spacing, orientation affine matrices,
dimensions, and multi-modal MRI headers without fabricating medical data.
Supports both NiBabel and standard-library fallback binary parsing.
"""

from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional
from abc import ABC, abstractmethod
import struct
import gzip
import os

try:
    import nibabel as nib
    HAS_NIBABEL = True
except ImportError:
    HAS_NIBABEL = False

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


class CorruptedNiftiError(Exception):
    """Raised when a NIfTI file is unreadable, truncated, or contains invalid metadata."""
    pass


class BaseNiftiHandler(ABC):
    """Abstract interface for parsing and reading NIfTI (.nii, .nii.gz) medical files."""

    @abstractmethod
    def read_header_metadata(self, file_path: Path) -> Dict[str, Any]:
        """Extracts voxel dimensions (dx, dy, dz), affine matrix, and volume shape.

        Args:
            file_path: Path to the NIfTI file on disk.

        Returns:
            Dictionary containing:
                - dimensions: Tuple[int, int, int]
                - voxel_spacing: Tuple[float, float, float]
                - affine: List[List[float]]
                - orientation: str
                - valid_metadata: bool
        """
        pass

    @abstractmethod
    def load_volume_array(self, file_path: Path) -> Any:
        """Loads voxel intensity data without modification."""
        pass


class NiftiHandler(BaseNiftiHandler):
    """Concrete NIfTI reader supporting NiBabel when installed, with a robust

    standard-library binary NIfTI-1 header fallback for environment portability.
    """

    def read_header_metadata(self, file_path: Path) -> Dict[str, Any]:
        """Extracts dimensions, voxel spacing, orientation, and affine matrix.

        Raises CorruptedNiftiError if the file cannot be parsed or is truncated.
        """
        path = Path(file_path)
        if not path.is_file():
            raise FileNotFoundError(f"NIfTI file does not exist: {path}")

        if HAS_NIBABEL:
            try:
                img = nib.load(str(path))
                shape = tuple(int(s) for s in img.shape[:3])
                zooms = tuple(float(z) for z in img.header.get_zooms()[:3])
                affine = img.affine.tolist()
                try:
                    orientation = "".join(nib.aff2axcodes(img.affine))
                except Exception:
                    orientation = "UNKNOWN"

                # Validate non-degenerate dimensions
                if any(s <= 0 for s in shape) or any(z <= 0 for z in zooms):
                    raise CorruptedNiftiError(
                        f"Invalid dimensions {shape} or voxel spacing {zooms} in {path.name}"
                    )

                return {
                    "dimensions": shape,
                    "voxel_spacing": zooms,
                    "affine": affine,
                    "orientation": orientation,
                    "valid_metadata": True,
                    "data_type": str(img.get_data_dtype()),
                    "handler": "nibabel"
                }
            except Exception as e:
                # If it's already CorruptedNiftiError re-raise, else inspect with binary reader
                if isinstance(e, CorruptedNiftiError):
                    raise
                # Fall through to binary reader to provide detailed diagnosis

        # Standard-library binary NIfTI-1 reader
        return self._read_nifti1_binary_header(path)

    def _read_nifti1_binary_header(self, path: Path) -> Dict[str, Any]:
        """Parses the official 348-byte NIfTI-1 binary header directly.

        Complies with NIfTI-1 Data Format specification:
        http://nifti.nimh.nih.gov/nifti-1/
        """
        try:
            # Check gzip vs uncompressed
            with open(path, "rb") as f_test:
                magic_test = f_test.read(2)

            is_gz = (magic_test == b"\x1f\x8b")
            open_fn = gzip.open if is_gz else open

            with open_fn(path, "rb") as f:
                header_bytes = f.read(348)

            if len(header_bytes) < 348:
                raise CorruptedNiftiError(
                    f"Truncated NIfTI header ({len(header_bytes)} bytes < 348) in {path.name}"
                )

            # sizeof_hdr is first 4 bytes (int32). Standard is 348.
            sizeof_hdr = struct.unpack("<i", header_bytes[:4])[0]
            endian = "<"
            if sizeof_hdr != 348:
                # Try big endian
                sizeof_hdr_be = struct.unpack(">i", header_bytes[:4])[0]
                if sizeof_hdr_be == 348:
                    endian = ">"
                else:
                    raise CorruptedNiftiError(
                        f"Invalid NIfTI sizeof_hdr: {sizeof_hdr} (expected 348) in {path.name}"
                    )

            # Check NIfTI magic at offset 344 (4 bytes)
            magic = header_bytes[344:348]
            if magic not in (b"n+1\0", b"ni1\0"):
                raise CorruptedNiftiError(
                    f"Invalid NIfTI magic bytes {magic!r} in {path.name}"
                )

            # dim is at offset 40 (8 short ints: dim[0]..dim[7])
            dim = struct.unpack(f"{endian}8h", header_bytes[40:56])
            ndim = dim[0]
            if ndim < 3 or ndim > 7:
                raise CorruptedNiftiError(f"Invalid NIfTI ndim={ndim} in {path.name}")

            nx, ny, nz = int(dim[1]), int(dim[2]), int(dim[3])
            if nx <= 0 or ny <= 0 or nz <= 0:
                raise CorruptedNiftiError(
                    f"Invalid non-positive dimensions ({nx}, {ny}, {nz}) in {path.name}"
                )

            # pixdim is at offset 76 (8 floats: qfac, dx, dy, dz, dt...)
            pixdim = struct.unpack(f"{endian}8f", header_bytes[76:108])
            dx, dy, dz = float(pixdim[1]), float(pixdim[2]), float(pixdim[3])
            if dx <= 0 or dy <= 0 or dz <= 0:
                raise CorruptedNiftiError(
                    f"Invalid non-positive voxel spacing ({dx}, {dy}, {dz}) in {path.name}"
                )

            # sform_code at offset 254 (short)
            sform_code = struct.unpack(f"{endian}h", header_bytes[254:256])[0]
            if sform_code > 0:
                # srow_x, srow_y, srow_z at offset 280 (12 floats)
                srows = struct.unpack(f"{endian}12f", header_bytes[280:328])
                affine = [
                    [float(srows[0]), float(srows[1]), float(srows[2]), float(srows[3])],
                    [float(srows[4]), float(srows[5]), float(srows[6]), float(srows[7])],
                    [float(srows[8]), float(srows[9]), float(srows[10]), float(srows[11])],
                    [0.0, 0.0, 0.0, 1.0]
                ]
            else:
                # Default diagonal affine with voxel spacing
                affine = [
                    [dx, 0.0, 0.0, 0.0],
                    [0.0, dy, 0.0, 0.0],
                    [0.0, 0.0, dz, 0.0],
                    [0.0, 0.0, 0.0, 1.0]
                ]

            # Approximate anatomical orientation from diagonal signs
            orientation = (
                ("R" if affine[0][0] >= 0 else "L") +
                ("A" if affine[1][1] >= 0 else "P") +
                ("S" if affine[2][2] >= 0 else "I")
            )

            return {
                "dimensions": (nx, ny, nz),
                "voxel_spacing": (round(dx, 4), round(dy, 4), round(dz, 4)),
                "affine": affine,
                "orientation": orientation,
                "valid_metadata": True,
                "data_type": "standard_nifti1",
                "handler": "nifti1_binary_reader"
            }
        except CorruptedNiftiError:
            raise
        except Exception as e:
            raise CorruptedNiftiError(f"Failed to parse NIfTI binary header from {path.name}: {e}")

    def load_volume_array(self, file_path: Path) -> Any:
        """Loads voxel intensity data."""
        path = Path(file_path)
        if not path.is_file():
            raise FileNotFoundError(f"File not found: {path}")

        if HAS_NIBABEL:
            img = nib.load(str(path))
            return img.get_fdata()

        # If NiBabel not installed, check if numpy is present to unpack raw bytes
        if HAS_NUMPY:
            # Load raw voxel data after voxel offset
            header = self._read_nifti1_binary_header(path)
            shape = header["dimensions"]
            with (gzip.open(path, "rb") if path.name.endswith(".gz") else open(path, "rb")) as f:
                f.seek(352)  # standard offset for n+1
                data = f.read()
            arr = np.frombuffer(data, dtype=np.float32)
            if arr.size == shape[0] * shape[1] * shape[2]:
                return arr.reshape(shape)

        raise NotImplementedError(
            "Full voxel array extraction requires 'nibabel' or 'numpy'. "
            "Please run: pip install -r ai/requirements-local.txt"
        )


nifti_handler = NiftiHandler()
