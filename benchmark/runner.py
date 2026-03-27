"""
Main orchestrator for Splitz benchmark experiments.

Ties together the DSP pipeline, ONNX model inference, and quality
metrics into a single ``run_experiment`` function that produces
structured JSON results.
"""

import logging
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from .metrics import (
    compute_bss_metrics,
    compute_cross_stem_leakage,
    compute_energy_ratio,
    compute_reconstruction_error,
    time_function,
)
from .models import STEMS, download_model, infer_spectrogram, load_session
from .pipeline import (
    HOP_LENGTH,
    N_FFT,
    NB_BINS,
    NB_EXPECTED_FRAMES,
    SR,
    chunk_audio,
    istft,
    magnitude,
    pad_to_frames,
    ratio_mask_separate,
    reassemble_chunks,
    stft,
    wiener_separate,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _flatten_magnitude(
    mag: np.ndarray,
    channels: int,
    actual_frames: int,
) -> np.ndarray:
    """Pack a magnitude spectrogram into a flat float32 array.

    Matches ``StemSeparationEngine.flattenMagnitude`` in Kotlin:
    layout is ``[ch, bin, frame]`` in row-major order, zero-padded
    to ``NB_EXPECTED_FRAMES`` along the frame axis.
    """
    size = channels * NB_BINS * NB_EXPECTED_FRAMES
    flat = np.zeros(size, dtype=np.float32)

    for ch in range(channels):
        for k in range(NB_BINS):
            base = (
                ch * NB_BINS * NB_EXPECTED_FRAMES
                + k * NB_EXPECTED_FRAMES
            )
            frames_to_copy = min(actual_frames, NB_EXPECTED_FRAMES)
            flat[base : base + frames_to_copy] = mag[
                ch, k, :frames_to_copy
            ]

    return flat


def _unpack_estimate(
    flat: np.ndarray,
    channels: int,
    actual_frames: int,
) -> np.ndarray:
    """Unpack a flat model output into ``[channels, NB_BINS, actual_frames]``.

    Matches ``StemSeparationEngine.unpackEstimate`` in Kotlin.
    """
    result = np.zeros(
        (channels, NB_BINS, actual_frames), dtype=np.float32
    )

    for ch in range(channels):
        for k in range(NB_BINS):
            base = (
                ch * NB_BINS * NB_EXPECTED_FRAMES
                + k * NB_EXPECTED_FRAMES
            )
            frames_to_copy = min(actual_frames, NB_EXPECTED_FRAMES)
            result[ch, k, :frames_to_copy] = flat[
                base : base + frames_to_copy
            ]

    return result


def _load_audio(audio_path: str) -> tuple[np.ndarray, int]:
    """Load a WAV file and return ``([channels, samples], sr)``."""
    data, sr = sf.read(audio_path, dtype="float32", always_2d=True)
    # soundfile returns [samples, channels] — transpose
    audio = data.T  # [channels, samples]
    logger.info(
        f"Loaded audio: {audio.shape[1]} samples, "
        f"{audio.shape[0]} channels, sr={sr}"
    )
    return audio, sr


def _normalize_peak(audio: np.ndarray) -> np.ndarray:
    """Normalize audio peak to [-1, 1]."""
    peak = np.max(np.abs(audio))
    if peak > 1e-8:
        return audio / peak
    return audio


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_experiment(
    config: dict,
    audio_path: str | None = None,
    audio_array: np.ndarray | None = None,
    reference_stems: dict[str, np.ndarray] | None = None,
    cache_dir: Path | None = None,
) -> dict:
    """Run a single separation experiment.

    Parameters
    ----------
    config : dict
        Experiment configuration with keys:

        - ``name`` (str): experiment identifier
        - ``wiener_power`` (float): Wiener mask exponent (default 2.0)
        - ``use_wiener`` (bool): use Wiener masking (default True)
        - ``chunk_seconds`` (float): chunk duration (default 10.0)
        - ``overlap_seconds`` (float): chunk overlap (default 0.5)
        - ``quantization`` (str): model quantization (default "fp16")

    audio_path : str, optional
        Path to a WAV file.  Mutually exclusive with *audio_array*.
    audio_array : np.ndarray, optional
        Pre-loaded audio ``[channels, samples]``.
    reference_stems : dict, optional
        Ground-truth stems for SDR computation (synthetic signal
        only).  Maps stem name to ``[channels, samples]`` array.
    cache_dir : Path, optional
        Override model cache directory.

    Returns
    -------
    dict
        Structured results: timings, metrics, configuration.
    """
    exp_name = config.get("name", "unnamed")
    wiener_power = float(config.get("wiener_power", 2.0))
    use_wiener = bool(config.get("use_wiener", True))
    chunk_seconds = float(config.get("chunk_seconds", 10.0))
    overlap_seconds = float(config.get("overlap_seconds", 0.5))
    quantization = config.get("quantization", "fp16")

    logger.info(f"=== Experiment: {exp_name} ===")
    logger.info(
        f"  wiener={use_wiener}, power={wiener_power}, "
        f"chunks={chunk_seconds}s, overlap={overlap_seconds}s, "
        f"quant={quantization}"
    )

    total_start = time.perf_counter()
    timings: dict[str, float] = {}

    # ------------------------------------------------------------------
    # 1. Load audio
    # ------------------------------------------------------------------
    if audio_path is not None:
        (audio, sr), load_ms = time_function(_load_audio, audio_path)
    elif audio_array is not None:
        audio = audio_array
        sr = SR
        load_ms = 0.0
    else:
        raise ValueError(
            "Either audio_path or audio_array must be provided"
        )
    timings["load_audio_ms"] = load_ms

    # 2. Normalize
    audio = _normalize_peak(audio)
    total_samples = audio.shape[1]
    channels = audio.shape[0]
    logger.info(
        f"Audio: {total_samples} samples, {channels} channels, "
        f"sr={sr}"
    )

    # ------------------------------------------------------------------
    # 3. Download / load ONNX models
    # ------------------------------------------------------------------
    sessions = {}
    model_load_start = time.perf_counter()
    try:
        for stem in STEMS:
            model_path = download_model(
                stem, cache_dir=cache_dir, quantization=quantization
            )
            sessions[stem] = load_session(model_path)
        timings["model_load_ms"] = (
            (time.perf_counter() - model_load_start) * 1000.0
        )
        logger.info(f"All {len(sessions)} ONNX sessions loaded")

        # --------------------------------------------------------------
        # 4. Chunk audio
        # --------------------------------------------------------------
        chunks = chunk_audio(
            audio,
            chunk_seconds=chunk_seconds,
            overlap_seconds=overlap_seconds,
            sr=sr,
        )
        logger.info(f"Split into {len(chunks)} chunks")

        overlap_samples = int(overlap_seconds * sr)

        # --------------------------------------------------------------
        # 5. Process each chunk
        # --------------------------------------------------------------
        stem_chunks: dict[str, list[dict]] = {
            s: [] for s in STEMS
        }
        inference_ms_total = 0.0

        for chunk_idx, chunk in enumerate(chunks):
            chunk_start = time.perf_counter()

            chunk_data = chunk["data"]  # [channels, chunk_samples]
            chunk_audio_length = chunk_data.shape[1]

            # STFT
            stft_complex = stft(chunk_data)
            mix_mag = magnitude(stft_complex)
            actual_frames = stft_complex.shape[2]

            # Flatten for model input
            input_flat = _flatten_magnitude(
                mix_mag, channels, actual_frames
            )
            shape = (1, channels, NB_BINS, NB_EXPECTED_FRAMES)

            # Run all 4 models
            estimates: dict[str, np.ndarray] = {}
            for stem in STEMS:
                output_flat = infer_spectrogram(
                    sessions[stem], input_flat, shape
                )
                estimates[stem] = _unpack_estimate(
                    output_flat, channels, actual_frames
                )

            # Apply masking
            if use_wiener:
                for stem in STEMS:
                    separated = wiener_separate(
                        stft_complex,
                        estimates,
                        stem,
                        audio_length=chunk_audio_length,
                        power=wiener_power,
                    )
                    result_chunk = chunk.copy()
                    result_chunk["data"] = separated
                    stem_chunks[stem].append(result_chunk)
            else:
                for stem in STEMS:
                    separated = ratio_mask_separate(
                        stft_complex,
                        mix_mag,
                        estimates[stem],
                        audio_length=chunk_audio_length,
                    )
                    result_chunk = chunk.copy()
                    result_chunk["data"] = separated
                    stem_chunks[stem].append(result_chunk)

            chunk_ms = (
                (time.perf_counter() - chunk_start) * 1000.0
            )
            inference_ms_total += chunk_ms
            logger.debug(
                f"Chunk {chunk_idx + 1}/{len(chunks)}: "
                f"{chunk_ms:.1f} ms"
            )

        timings["inference_total_ms"] = inference_ms_total
        timings["inference_per_chunk_ms"] = (
            inference_ms_total / len(chunks) if chunks else 0.0
        )

        # --------------------------------------------------------------
        # 6. Reassemble chunks
        # --------------------------------------------------------------
        separated_stems: dict[str, np.ndarray] = {}
        for stem in STEMS:
            separated_stems[stem] = reassemble_chunks(
                stem_chunks[stem], overlap_samples, total_samples
            )

        # --------------------------------------------------------------
        # 7. Compute metrics
        # --------------------------------------------------------------
        metrics: dict[str, dict] = {}

        # Always compute relative metrics
        for stem in STEMS:
            stem_metrics: dict[str, float] = {}
            stem_metrics["energy_ratio"] = compute_energy_ratio(
                separated_stems[stem], audio
            )
            metrics[stem] = stem_metrics

        metrics["cross_stem_leakage"] = compute_cross_stem_leakage(
            separated_stems
        )
        metrics["reconstruction_error"] = (
            compute_reconstruction_error(audio, separated_stems)
        )

        # SDR/SIR/SAR if reference stems provided
        if reference_stems is not None:
            for stem in STEMS:
                if stem in reference_stems:
                    bss = compute_bss_metrics(
                        reference_stems[stem],
                        separated_stems[stem],
                        sr=sr,
                    )
                    metrics[stem].update(bss)
                    logger.info(
                        f"{stem}: SDR={bss['sdr']:.2f} dB, "
                        f"SIR={bss['sir']:.2f} dB, "
                        f"SAR={bss['sar']:.2f} dB"
                    )

    finally:
        # Always close sessions
        for session in sessions.values():
            del session
        logger.debug("ONNX sessions released")

    total_ms = (time.perf_counter() - total_start) * 1000.0
    timings["total_ms"] = total_ms

    results = {
        "experiment": exp_name,
        "config": {
            "use_wiener": use_wiener,
            "wiener_power": wiener_power,
            "chunk_seconds": chunk_seconds,
            "overlap_seconds": overlap_seconds,
            "quantization": quantization,
        },
        "audio": {
            "source": audio_path or "synthetic",
            "channels": channels,
            "samples": total_samples,
            "duration_seconds": total_samples / sr,
            "sample_rate": sr,
        },
        "timings": timings,
        "metrics": metrics,
        "n_chunks": len(chunks),
    }

    logger.info(
        f"Experiment '{exp_name}' completed in "
        f"{total_ms:.0f} ms "
        f"(recon_error={metrics['reconstruction_error']:.6f})"
    )

    return results
