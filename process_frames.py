"""VTK/animation frame loading and entropy extraction."""

import hashlib
import sys
from pathlib import Path

import numpy as np

ALPHA_FIELD_CANDIDATES = [
    "alpha.water", "alpha.phase1", "alpha_1", "alpha", "alpha1",
    "vof", "phi", "f", "C", "volume_fraction", "phase"
]


def find_vtk_files(vtk_dir: str) -> list[Path]:
    d = Path(vtk_dir)

    # Prefer one assembled file per cycle (handles multi-piece .pvtu automatically)
    files = sorted(d.glob("Cycle*/data.pvtu"))

    if not files:
        files = sorted(
            list(d.glob("*.pvtu"))
            + list(d.glob("*.vtu"))
            + list(d.glob("*.vtk"))
        )

    if not files:
        files = sorted(
            list(d.glob("Cycle*/*.vtu"))
            + list(d.glob("Cycle*/*.vtk"))
        )

    if not files:
        sys.exit(
            f"No VTK files found in {vtk_dir}\n"
            "  Checked: Cycle*/data.pvtu, *.pvtu, *.vtu, *.vtk"
        )

    print(f"[vtk]  Found {len(files)} timestep files in {vtk_dir}")
    return files


def load_vtk_snapshot(filepath: Path, field_name: str = None) -> tuple[np.ndarray, np.ndarray, str]:
    """
    Load mesh point coordinates and alpha field from VTK/VTU/PVTU.
    Returns (points[N,3], alpha[N], field_name_used).
    """
    try:
        import vtk
        from vtk.util.numpy_support import vtk_to_numpy
    except ImportError:
        sys.exit("vtk required: pip install vtk")

    path = str(filepath)
    if path.endswith(".pvtu"):
        reader = vtk.vtkXMLPUnstructuredGridReader()
    elif path.endswith(".vtu"):
        reader = vtk.vtkXMLUnstructuredGridReader()
    elif path.endswith(".vtk"):
        reader = vtk.vtkUnstructuredGridReader()
    else:
        raise ValueError(f"Unsupported file type: {filepath.name}")

    reader.SetFileName(path)
    reader.Update()
    mesh = reader.GetOutput()
    points = vtk_to_numpy(mesh.GetPoints().GetData()).astype(np.float64)
    point_data = mesh.GetPointData()

    if point_data.GetNumberOfArrays() == 0:
        raise ValueError(f"No point data arrays in {filepath.name}")

    if field_name:
        arr = point_data.GetArray(field_name)
        if arr is None:
            available = [point_data.GetArrayName(i) for i in range(point_data.GetNumberOfArrays())]
            raise ValueError(
                f"Field '{field_name}' not found in {filepath.name}. Available: {available}"
            )
        return points, vtk_to_numpy(arr).astype(np.float64), field_name

    for candidate in ALPHA_FIELD_CANDIDATES:
        arr = point_data.GetArray(candidate)
        if arr is not None:
            return points, vtk_to_numpy(arr).astype(np.float64), candidate

    name = point_data.GetArrayName(0)
    available = [point_data.GetArrayName(i) for i in range(point_data.GetNumberOfArrays())]
    print(f"[vtk]  Auto-selected field '{name}' (available: {available})")
    return points, vtk_to_numpy(point_data.GetArray(0)).astype(np.float64), name


def extract_interface_triplets(
    points: np.ndarray,
    alpha: np.ndarray,
    lo: float = 0.05,
    hi: float = 0.95,
) -> np.ndarray:
    """
    Extract (x, y, alpha) triplets in the interface band [lo, hi].

    Returns shape (M, 3). Points are sorted by (y, x) within each snapshot
    so the serialized state resembles a spatial image scan (lavarand-style).
    """
    mask = (alpha >= lo) & (alpha <= hi)
    triplets = np.column_stack([points[mask, 0], points[mask, 1], alpha[mask]])
    if len(triplets) == 0:
        return triplets
    order = np.lexsort((triplets[:, 0], triplets[:, 1]))
    return triplets[order]


def find_anim_frames(anim_dir: str) -> list[Path]:
    d = Path(anim_dir)
    files = sorted(set(d.glob("frame*.png")) | set(d.glob("*.png")))
    if not files:
        sys.exit(f"No PNG frames found in {anim_dir}")
    print(f"[anim] Found {len(files)} frames in {anim_dir}")
    return files


def load_frame_pixels(filepath: Path) -> bytes:
    """
    Decode a PNG frame to raw RGB pixel bytes (lavarand uses camera pixel data).
    """
    try:
        from PIL import Image
    except ImportError:
        sys.exit("pillow required for --anim-dir: pip install pillow")

    img = Image.open(filepath).convert("RGB")
    return np.ascontiguousarray(np.asarray(img), dtype=np.uint8).tobytes()


def frames_to_random_bytes(frames: list[Path]) -> tuple[bytes, int]:
    """
    Hash RGB pixel data from each animation frame individually.
    Returns the concatenated SHA-256 hashes and total pixel count.
    """
    hashes = []
    total_pixels = 0
    for fpath in frames:
        pixels = load_frame_pixels(fpath)
        hashes.append(hashlib.sha256(pixels).digest())
        total_pixels += len(pixels) // 3
    return b"".join(hashes), total_pixels


def interface_triplets_to_random_bytes(snapshots: list[np.ndarray]) -> bytes:
    """
    Extract random bytes by hashing each (x, y, alpha) interface snapshot.
    """
    hashes = []
    for triplets in snapshots:
        if len(triplets) == 0:
            continue
        # Round to ensure stability, then hash the raw coordinate/alpha data
        snapshot = np.round(triplets, decimals=12).astype(np.float64, copy=False)
        state = np.ascontiguousarray(snapshot).tobytes()
        hashes.append(hashlib.sha256(state).digest())
    return b"".join(hashes)


def bytes_to_uniform_floats(b: bytes) -> np.ndarray:
    """Convert byte stream to uniform float64 values in [0, 1)."""
    n = len(b) // 8
    vals = np.frombuffer(b[:n * 8], dtype=np.uint64)
    return vals.astype(np.float64) / np.float64(2**64)


def bytes_to_bits(b: bytes) -> np.ndarray:
    """Convert bytes to array of bits (0/1)."""
    return np.unpackbits(np.frombuffer(b, dtype=np.uint8))
