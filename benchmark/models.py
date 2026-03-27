"""
ONNX model management for the Splitz benchmark.

Handles downloading Open-Unmix models from the Splitz Research CDN,
SHA-256 verification, caching, and ONNX Runtime session creation.
"""

import hashlib
import json
import logging
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen, urlretrieve

import numpy as np
import onnxruntime as ort

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_CDN = (
    "https://thomasphillips3.github.io/splitz-research/models"
)
STEMS = ["vocals", "drums", "bass", "other"]
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "splitz-research" / "models"
MANIFEST_FILENAME = "manifest.json"


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


def _resolve_cache_dir(cache_dir: Path | None = None) -> Path:
    """Return the cache directory, creating it if needed."""
    d = cache_dir or DEFAULT_CACHE_DIR
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_manifest(cache_dir: Path | None = None) -> dict:
    """Fetch the model manifest from the CDN and cache locally.

    The manifest maps model filenames to their SHA-256 checksums and
    metadata.  It is re-fetched on every call to pick up new model
    releases, but the cached copy is used as fallback if the CDN is
    unreachable.

    Parameters
    ----------
    cache_dir : Path, optional
        Override the default cache directory.

    Returns
    -------
    dict
        Parsed manifest JSON.
    """
    d = _resolve_cache_dir(cache_dir)
    manifest_path = d / MANIFEST_FILENAME
    manifest_url = f"{MODEL_CDN}/{MANIFEST_FILENAME}"

    try:
        logger.debug(f"Fetching manifest from {manifest_url}")
        with urlopen(manifest_url, timeout=30) as resp:
            data = resp.read()
        manifest_path.write_bytes(data)
        logger.info("Model manifest updated from CDN")
    except (URLError, OSError) as exc:
        if manifest_path.exists():
            logger.warning(
                f"CDN unreachable ({exc}), using cached manifest"
            )
            data = manifest_path.read_bytes()
        else:
            raise RuntimeError(
                f"Cannot fetch manifest and no cache exists: {exc}"
            ) from exc

    return json.loads(data)


# ---------------------------------------------------------------------------
# Download + verify
# ---------------------------------------------------------------------------


def _sha256_file(path: Path) -> str:
    """Compute hex SHA-256 of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def download_model(
    stem: str,
    cache_dir: Path | None = None,
    quantization: str = "fp16",
) -> Path:
    """Download an ONNX model for a given stem, with SHA-256 verification.

    Models are cached; repeated calls return the cached path instantly
    unless the checksum has changed in the manifest.

    Parameters
    ----------
    stem : str
        One of ``STEMS`` (vocals, drums, bass, other).
    cache_dir : Path, optional
        Override the default cache directory.
    quantization : str
        Quantization variant: ``fp32``, ``fp16``, or ``int8``.

    Returns
    -------
    Path
        Local path to the verified ONNX model file.

    Raises
    ------
    RuntimeError
        If the download fails or checksum does not match.
    """
    if stem not in STEMS:
        raise ValueError(
            f"Unknown stem '{stem}', expected one of {STEMS}"
        )

    d = _resolve_cache_dir(cache_dir)
    manifest = load_manifest(d)

    filename = f"open-unmix-{stem}-{quantization}.onnx"
    model_path = d / filename

    # Look up expected checksum
    model_info = manifest.get("models", {}).get(filename)
    expected_sha = model_info.get("sha256") if model_info else None

    # Use cache if valid
    if model_path.exists() and expected_sha:
        actual_sha = _sha256_file(model_path)
        if actual_sha == expected_sha:
            logger.debug(f"Cache hit for {filename}")
            return model_path
        logger.warning(
            f"Checksum mismatch for cached {filename}, "
            f"re-downloading"
        )

    # Download
    url = f"{MODEL_CDN}/{filename}"
    logger.info(f"Downloading {filename} from {url}")
    try:
        urlretrieve(url, model_path)
    except (URLError, OSError) as exc:
        raise RuntimeError(
            f"Failed to download {filename}: {exc}"
        ) from exc

    # Verify
    if expected_sha:
        actual_sha = _sha256_file(model_path)
        if actual_sha != expected_sha:
            model_path.unlink(missing_ok=True)
            raise RuntimeError(
                f"SHA-256 mismatch for {filename}: "
                f"expected {expected_sha[:16]}..., "
                f"got {actual_sha[:16]}..."
            )
        logger.info(f"Verified {filename} (SHA-256 OK)")
    else:
        logger.warning(
            f"No checksum in manifest for {filename}, "
            f"skipping verification"
        )

    return model_path


# ---------------------------------------------------------------------------
# ONNX Runtime inference
# ---------------------------------------------------------------------------


def load_session(model_path: Path) -> ort.InferenceSession:
    """Create an ONNX Runtime inference session.

    Uses the CPU execution provider.  GPU providers (CUDA,
    CoreML) can be added by extending the provider list.

    Parameters
    ----------
    model_path : Path
        Path to the ``.onnx`` file.

    Returns
    -------
    ort.InferenceSession
        Ready-to-use session.
    """
    opts = ort.SessionOptions()
    opts.inter_op_num_threads = 1
    opts.intra_op_num_threads = 4
    opts.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    )

    session = ort.InferenceSession(
        str(model_path),
        sess_options=opts,
        providers=["CPUExecutionProvider"],
    )
    logger.info(f"Loaded ONNX session: {model_path.name}")
    return session


def infer_spectrogram(
    session: ort.InferenceSession,
    magnitude_flat: np.ndarray,
    shape: tuple[int, int, int, int],
) -> np.ndarray:
    """Run a single model inference on a magnitude spectrogram.

    Parameters
    ----------
    session : ort.InferenceSession
        Pre-loaded ONNX session.
    magnitude_flat : np.ndarray
        Flat float32 array of the input magnitude spectrogram.
    shape : tuple[int, int, int, int]
        Expected input tensor shape, e.g. ``(1, 2, 2049, 431)``.

    Returns
    -------
    np.ndarray
        Flat float32 output array (estimated magnitude).
    """
    input_tensor = magnitude_flat.reshape(shape).astype(np.float32)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    results = session.run([output_name], {input_name: input_tensor})
    return results[0].flatten().astype(np.float32)
