"""NIfTI label and graph archive loader for Batch 1 data extraction."""

from __future__ import annotations

import gzip
import struct
from pathlib import Path
from typing import Any

import numpy as np

EPS = 1.0e-12
NIFTI_DTYPES = {
    2: "u1", 4: "i2", 8: "i4", 16: "f4", 64: "f8", 256: "i1",
    512: "u2", 768: "u4", 1024: "i8", 1280: "u8",
}


def read_nifti_header(path: Path) -> dict[str, Any]:
    """Read essential header fields from a NIfTI-1 file."""
    with gzip.open(path, "rb") if path.suffix == ".gz" else open(path, "rb") as handle:
        header = handle.read(352)
    if len(header) < 352:
        raise ValueError(f"truncated NIfTI header: {path.name}")
    if struct.unpack("<i", header[:4])[0] == 348:
        endian = "<"
    elif struct.unpack(">i", header[:4])[0] == 348:
        endian = ">"
    else:
        raise ValueError(f"not a NIfTI-1 file: {path.name}")

    dimensions = struct.unpack(endian + "8h", header[40:56])
    if dimensions[0] != 3:
        raise ValueError(f"expected a 3-D label volume; got {dimensions[0]} dimensions")
    shape = tuple(int(value) for value in dimensions[1:4])
    spacing = np.asarray(struct.unpack(endian + "8f", header[76:108])[1:4], dtype=float)
    datatype = int(struct.unpack(endian + "h", header[70:72])[0])
    if datatype not in NIFTI_DTYPES:
        raise ValueError(f"unsupported NIfTI datatype code {datatype}")
    voxel_offset = int(round(struct.unpack(endian + "f", header[108:112])[0]))
    sform_code = int(struct.unpack(endian + "h", header[254:256])[0])
    
    # Extract sform or qform matrix for physical RAS coordinates
    affine = np.eye(4, dtype=float)
    if sform_code > 0:
        affine[:3, :] = np.asarray(
            [struct.unpack(endian + "4f", header[offset:offset + 16]) for offset in (280, 296, 312)]
        )
    else:
        # Fallback to spacing scaling if sform is missing
        affine[0, 0] = spacing[0]
        affine[1, 1] = spacing[1]
        affine[2, 2] = spacing[2]

    if not np.all(np.isfinite(affine)) or abs(np.linalg.det(affine[:3, :3])) <= EPS:
        raise ValueError("NIfTI affine matrix is non-finite or singular")

    return {
        "endian": endian,
        "shape": shape,
        "spacing_mm": spacing,
        "datatype": datatype,
        "voxel_offset": voxel_offset,
        "affine": affine,
        "sform_code": sform_code,
    }


def load_binary_nifti(path: Path, header: dict[str, Any]) -> np.ndarray:
    """Load gzipped or uncompressed NIfTI image as 3D boolean mask array."""
    shape = header["shape"]
    dtype = np.dtype(header["endian"] + NIFTI_DTYPES[header["datatype"]])
    values_per_plane = shape[0] * shape[1]
    bytes_per_plane = values_per_plane * dtype.itemsize
    mask = np.empty(shape, dtype=bool)

    open_fn = gzip.open if path.suffix == ".gz" else open
    with open_fn(path, "rb") as handle:
        handle.seek(header["voxel_offset"])
        for z_index in range(shape[2]):
            raw = handle.read(bytes_per_plane)
            if len(raw) != bytes_per_plane:
                raise ValueError(f"truncated NIfTI image data at z={z_index}")
            plane = np.frombuffer(raw, dtype=dtype, count=values_per_plane)
            mask[:, :, z_index] = plane.reshape(shape[:2], order="F") > 0

    if not np.any(mask):
        raise ValueError("label volume is empty")
    return mask


def load_centerline_archive(archive_dir: Path) -> dict[str, Any]:
    """Load pre-extracted centerline graph archive (branches.npz + summary.csv)."""
    branches_path = archive_dir / "branches.npz"
    summary_path = archive_dir / "summary.csv"
    landmarks_path = archive_dir / "landmarks.npy"

    if not branches_path.is_file() or not summary_path.is_file():
        raise FileNotFoundError(f"missing centerline archive in {archive_dir}")

    branches_data = np.load(branches_path)
    branches_dict = {key: branches_data[key] for key in branches_data.files}
    
    import pandas as pd
    summary_df = pd.read_csv(summary_path)

    existing_landmarks = None
    if landmarks_path.is_file():
        try:
            existing_landmarks = np.load(landmarks_path, allow_pickle=True).item()
        except Exception:
            existing_landmarks = None

    return {
        "branches_dict": branches_dict,
        "summary_df": summary_df,
        "existing_landmarks": existing_landmarks,
    }
