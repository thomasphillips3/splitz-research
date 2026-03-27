"""
Quality measurement for stem separation evaluation.

Provides BSS (Blind Source Separation) metrics via mir_eval, plus
lightweight energy, leakage, and reconstruction error metrics that
work without ground-truth references.
"""

import logging
import time
from typing import Any, Callable

import numpy as np
from mir_eval.separation import bss_eval_sources

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# BSS metrics (require ground-truth reference)
# ---------------------------------------------------------------------------


def compute_bss_metrics(
    reference: np.ndarray,
    estimated: np.ndarray,
    sr: int = 44100,
) -> dict[str, float]:
    """Compute SDR, SIR, and SAR using mir_eval.

    Both inputs must be mono or multi-channel with the same shape.
    For stereo signals the metrics are averaged across channels.

    Parameters
    ----------
    reference : np.ndarray
        Ground-truth stem, shape ``[channels, samples]`` or ``[samples]``.
    estimated : np.ndarray
        Estimated stem, same shape as *reference*.
    sr : int
        Sample rate (logged for context, not used in computation).

    Returns
    -------
    dict[str, float]
        Keys: ``sdr``, ``sir``, ``sar`` (in dB).
    """
    # Ensure 2-D [sources, samples] — mir_eval wants that shape
    if reference.ndim == 1:
        reference = reference[np.newaxis, :]
    if estimated.ndim == 1:
        estimated = estimated[np.newaxis, :]

    # Trim to equal length
    min_len = min(reference.shape[-1], estimated.shape[-1])
    reference = reference[..., :min_len]
    estimated = estimated[..., :min_len]

    # Average across channels for stereo
    if reference.shape[0] > 1:
        ref_mono = reference.mean(axis=0, keepdims=True)
        est_mono = estimated.mean(axis=0, keepdims=True)
    else:
        ref_mono = reference
        est_mono = estimated

    sdr, sir, sar, _ = bss_eval_sources(ref_mono, est_mono)

    result = {
        "sdr": float(sdr[0]),
        "sir": float(sir[0]),
        "sar": float(sar[0]),
    }
    logger.debug(
        f"BSS metrics (sr={sr}): "
        f"SDR={result['sdr']:.2f} dB, "
        f"SIR={result['sir']:.2f} dB, "
        f"SAR={result['sar']:.2f} dB"
    )
    return result


# ---------------------------------------------------------------------------
# Relative metrics (no ground truth needed)
# ---------------------------------------------------------------------------


def compute_energy_ratio(
    stem_audio: np.ndarray, mix_audio: np.ndarray
) -> float:
    """Energy ratio of a stem relative to the mix.

    ``ratio = sum(stem^2) / sum(mix^2)``

    Parameters
    ----------
    stem_audio : np.ndarray
        Separated stem waveform.
    mix_audio : np.ndarray
        Original mix waveform.

    Returns
    -------
    float
        Energy ratio (0.0 to ~1.0 for well-behaved separations).
    """
    mix_energy = np.sum(mix_audio.astype(np.float64) ** 2)
    if mix_energy < 1e-12:
        return 0.0
    stem_energy = np.sum(stem_audio.astype(np.float64) ** 2)
    return float(stem_energy / mix_energy)


def compute_cross_stem_leakage(
    stems: dict[str, np.ndarray],
) -> dict[str, float]:
    """Pairwise Pearson correlation between all separated stems.

    High correlation between stems indicates leakage (the same
    signal appearing in multiple outputs).

    Parameters
    ----------
    stems : dict[str, np.ndarray]
        Stem name -> waveform array (any shape; flattened internally).

    Returns
    -------
    dict[str, float]
        ``"stem_a/stem_b"`` -> correlation coefficient.
    """
    names = sorted(stems.keys())
    flat = {
        name: stems[name].flatten().astype(np.float64)
        for name in names
    }

    # Trim all to common length
    min_len = min(a.shape[0] for a in flat.values())
    flat = {k: v[:min_len] for k, v in flat.items()}

    result: dict[str, float] = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = flat[names[i]], flat[names[j]]
            a_std = np.std(a)
            b_std = np.std(b)
            if a_std < 1e-12 or b_std < 1e-12:
                corr = 0.0
            else:
                corr = float(
                    np.corrcoef(a, b)[0, 1]
                )
            pair_key = f"{names[i]}/{names[j]}"
            result[pair_key] = corr

    return result


def compute_reconstruction_error(
    mix_audio: np.ndarray,
    stems: dict[str, np.ndarray],
) -> float:
    """Normalized L2 reconstruction error.

    ``error = ||mix - sum(stems)|| / ||mix||``

    A perfect decomposition yields 0.0.  Wiener masking should give
    near-zero since masks sum to 1.

    Parameters
    ----------
    mix_audio : np.ndarray
        Original mix, shape ``[channels, samples]``.
    stems : dict[str, np.ndarray]
        Separated stems, each same shape as *mix_audio*.

    Returns
    -------
    float
        Normalized reconstruction error.
    """
    mix = mix_audio.astype(np.float64)
    min_len = mix.shape[-1]
    for s in stems.values():
        min_len = min(min_len, s.shape[-1])

    mix = mix[..., :min_len]
    reconstruction = np.zeros_like(mix)
    for s in stems.values():
        reconstruction += s[..., :min_len].astype(np.float64)

    residual_norm = np.linalg.norm(mix - reconstruction)
    mix_norm = np.linalg.norm(mix)

    if mix_norm < 1e-12:
        return 0.0
    return float(residual_norm / mix_norm)


# ---------------------------------------------------------------------------
# Timing utility
# ---------------------------------------------------------------------------


def time_function(
    func: Callable, *args: Any, **kwargs: Any
) -> tuple[Any, float]:
    """Time a function call and return (result, elapsed_ms).

    Parameters
    ----------
    func : Callable
        Function to time.
    *args, **kwargs
        Arguments forwarded to *func*.

    Returns
    -------
    tuple[Any, float]
        ``(return_value, elapsed_milliseconds)``.
    """
    t0 = time.perf_counter()
    result = func(*args, **kwargs)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return result, elapsed_ms
