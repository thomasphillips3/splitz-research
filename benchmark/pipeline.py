"""
Python port of the Kotlin StftProcessor from the Splitz Android app.

Replicates the on-device STFT/iSTFT/Wiener pipeline exactly:
  n_fft=4096, hop_length=1024, periodic Hann window, center=True,
  reflect padding, overlap-add synthesis with window energy
  normalization.

All functions operate on numpy arrays with shape conventions matching
the Android code: [channels, samples] for audio, [channels, bins,
frames] for spectrograms.
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants — must match StftProcessor.kt and StemSeparationEngine.kt
# ---------------------------------------------------------------------------

N_FFT = 4096
HOP_LENGTH = 1024
NB_BINS = N_FFT // 2 + 1  # 2049
NB_EXPECTED_FRAMES = 431
SR = 44100


def _periodic_hann(n_fft: int) -> np.ndarray:
    """Periodic Hann window matching PyTorch hann_window(N, periodic=True).

    periodic Hann: w[n] = 0.5 * (1 - cos(2*pi*n / N))  for n in [0, N)
    Note the divisor is N, not N-1 (that would be the symmetric variant).
    """
    n = np.arange(n_fft, dtype=np.float64)
    return (0.5 * (1.0 - np.cos(2.0 * np.pi * n / n_fft))).astype(
        np.float32
    )


_HANN_WINDOW = _periodic_hann(N_FFT)


# ---------------------------------------------------------------------------
# Forward STFT
# ---------------------------------------------------------------------------


def _reflect_pad(audio: np.ndarray, pad_size: int) -> np.ndarray:
    """Reflect-pad a 1-D signal by *pad_size* samples on each side.

    Mirrors the Kotlin ``reflectPad`` exactly: left padding reflects
    audio[1], audio[2], ... and right padding reflects audio[n-2],
    audio[n-3], ...
    """
    return np.pad(audio, pad_size, mode="reflect")


def stft(
    audio: np.ndarray,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH,
) -> np.ndarray:
    """Compute the one-sided STFT of multi-channel audio.

    Parameters
    ----------
    audio : np.ndarray
        Shape ``[channels, samples]``.
    n_fft : int
        FFT size.
    hop_length : int
        Hop between frames.

    Returns
    -------
    np.ndarray
        Complex STFT of shape ``[channels, n_fft//2+1, n_frames]``.
    """
    channels, audio_length = audio.shape
    n_frames = audio_length // hop_length + 1
    nb_bins = n_fft // 2 + 1

    window = _periodic_hann(n_fft)
    result = np.zeros((channels, nb_bins, n_frames), dtype=np.complex64)

    for ch in range(channels):
        padded = _reflect_pad(audio[ch], n_fft // 2)
        for frame in range(n_frames):
            start = frame * hop_length
            end = start + n_fft
            segment = padded[start:end].astype(np.float32)
            if len(segment) < n_fft:
                segment = np.pad(
                    segment, (0, n_fft - len(segment))
                )
            windowed = segment * window
            spectrum = np.fft.rfft(windowed)
            result[ch, :, frame] = spectrum[:nb_bins]

    return result


# ---------------------------------------------------------------------------
# Magnitude
# ---------------------------------------------------------------------------


def magnitude(stft_complex: np.ndarray) -> np.ndarray:
    """Compute magnitude spectrogram: sqrt(re^2 + im^2).

    Parameters
    ----------
    stft_complex : np.ndarray
        Complex STFT of shape ``[channels, bins, frames]``.

    Returns
    -------
    np.ndarray
        Real-valued magnitude, same shape.
    """
    return np.abs(stft_complex).astype(np.float32)


# ---------------------------------------------------------------------------
# Inverse STFT
# ---------------------------------------------------------------------------


def istft(
    stft_complex: np.ndarray,
    audio_length: int,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH,
) -> np.ndarray:
    """Inverse STFT via overlap-add with Hann synthesis window.

    Matches the Kotlin ``istft`` exactly: conjugate-symmetric
    extension, overlap-add, window-energy normalization, center trim.

    Parameters
    ----------
    stft_complex : np.ndarray
        One-sided complex STFT, shape ``[channels, bins, frames]``.
    audio_length : int
        Original (un-padded) audio length in samples.
    n_fft : int
        FFT size.
    hop_length : int
        Hop between frames.

    Returns
    -------
    np.ndarray
        Reconstructed waveform, shape ``[channels, audio_length]``.
    """
    channels, nb_bins, n_frames = stft_complex.shape
    window = _periodic_hann(n_fft)

    padded_length = audio_length + n_fft
    output = np.zeros((channels, padded_length), dtype=np.float64)
    window_sum_sq = np.zeros(padded_length, dtype=np.float64)

    # Window energy accumulator (channel-independent)
    for frame in range(n_frames):
        start = frame * hop_length
        end = min(start + n_fft, padded_length)
        seg_len = end - start
        window_sum_sq[start:end] += (
            window[:seg_len].astype(np.float64) ** 2
        )

    for ch in range(channels):
        for frame in range(n_frames):
            # Build full-spectrum from one-sided via conjugate symmetry
            one_sided = stft_complex[ch, :, frame]
            full_spectrum = np.zeros(n_fft, dtype=np.complex128)
            full_spectrum[:nb_bins] = one_sided
            # Negative frequencies: X[N-k] = conj(X[k]) for k=1..N/2-1
            for k in range(1, n_fft // 2):
                full_spectrum[n_fft - k] = np.conj(
                    full_spectrum[k]
                )

            time_signal = np.fft.ifft(full_spectrum).real

            start = frame * hop_length
            end = min(start + n_fft, padded_length)
            seg_len = end - start
            output[ch, start:end] += (
                time_signal[:seg_len]
                * window[:seg_len].astype(np.float64)
            )

    # Normalize by window energy and remove center padding
    result = np.zeros((channels, audio_length), dtype=np.float32)
    half_pad = n_fft // 2
    for ch in range(channels):
        for i in range(audio_length):
            padded_idx = i + half_pad
            wsum = window_sum_sq[padded_idx]
            if wsum > 1e-8:
                result[ch, i] = output[ch, padded_idx] / wsum
            else:
                result[ch, i] = 0.0

    return result


# ---------------------------------------------------------------------------
# Masking strategies
# ---------------------------------------------------------------------------


def wiener_separate(
    mix_stft: np.ndarray,
    all_estimates: dict[str, np.ndarray],
    target_stem: str,
    audio_length: int,
    power: float = 2.0,
) -> np.ndarray:
    """Wiener power-mask separation.

    Computes ``mask_i = |est_i|^p / sum_j(|est_j|^p)`` across all
    stem estimates, applies to the mix STFT, and reconstructs via
    iSTFT.  Masks sum to 1.0 across sources — no bleed or
    amplification.

    Parameters
    ----------
    mix_stft : np.ndarray
        Complex STFT of the mix, ``[channels, bins, frames]``.
    all_estimates : dict[str, np.ndarray]
        Stem name -> magnitude array ``[channels, bins, frames]``.
    target_stem : str
        Which stem to extract.
    audio_length : int
        Original audio length for iSTFT.
    power : float
        Masking exponent (2.0 = Wiener, 1.0 = softmask).

    Returns
    -------
    np.ndarray
        Reconstructed stem waveform, ``[channels, samples]``.
    """
    if target_stem not in all_estimates:
        raise ValueError(
            f"No estimate for target stem: {target_stem}"
        )

    target_mag = all_estimates[target_stem]
    target_pow = np.power(target_mag, power)

    total_pow = np.zeros_like(target_pow)
    for est_mag in all_estimates.values():
        total_pow += np.power(est_mag, power)

    mask = np.where(
        total_pow > 1e-10, target_pow / total_pow, 0.0
    ).astype(np.float32)

    masked_stft = mix_stft * mask
    return istft(masked_stft, audio_length)


def ratio_mask_separate(
    mix_stft: np.ndarray,
    mix_mag: np.ndarray,
    estimate: np.ndarray,
    audio_length: int,
    epsilon: float = 1e-10,
) -> np.ndarray:
    """Ratio soft-mask separation (single-stem fallback).

    ``mask = estimate / (mix_mag + eps)``, clamped to [0, 1].

    Parameters
    ----------
    mix_stft : np.ndarray
        Complex STFT of the mix, ``[channels, bins, frames]``.
    mix_mag : np.ndarray
        Magnitude of the mix STFT, same shape.
    estimate : np.ndarray
        Estimated magnitude for the target stem.
    audio_length : int
        Original audio length for iSTFT.
    epsilon : float
        Numerical stability term.

    Returns
    -------
    np.ndarray
        Reconstructed stem waveform, ``[channels, samples]``.
    """
    mask = np.clip(estimate / (mix_mag + epsilon), 0.0, 1.0).astype(
        np.float32
    )
    masked_stft = mix_stft * mask
    return istft(masked_stft, audio_length)


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------


def chunk_audio(
    audio: np.ndarray,
    chunk_seconds: float = 10.0,
    overlap_seconds: float = 0.5,
    sr: int = SR,
) -> list[dict]:
    """Split audio into overlapping chunks for model inference.

    Parameters
    ----------
    audio : np.ndarray
        Shape ``[channels, samples]``.
    chunk_seconds : float
        Duration of each chunk in seconds.
    overlap_seconds : float
        Overlap between adjacent chunks in seconds.
    sr : int
        Sample rate.

    Returns
    -------
    list[dict]
        Each dict has keys: ``data`` ([channels, chunk_samples]),
        ``start_sample``, ``is_first``, ``is_last``.
    """
    total_samples = audio.shape[1]
    chunk_samples = int(chunk_seconds * sr)
    overlap_samples = int(overlap_seconds * sr)
    step = chunk_samples - overlap_samples

    chunks: list[dict] = []
    start = 0
    idx = 0

    while start < total_samples:
        end = min(start + chunk_samples, total_samples)
        chunk_data = audio[:, start:end]

        # Zero-pad the last chunk if shorter than expected
        if chunk_data.shape[1] < chunk_samples:
            pad_width = chunk_samples - chunk_data.shape[1]
            chunk_data = np.pad(
                chunk_data, ((0, 0), (0, pad_width))
            )

        is_first = idx == 0
        is_last = end >= total_samples

        chunks.append(
            {
                "data": chunk_data,
                "start_sample": start,
                "is_first": is_first,
                "is_last": is_last,
            }
        )

        if is_last:
            break
        start += step
        idx += 1

    logger.debug(
        f"Chunked {total_samples} samples into {len(chunks)} "
        f"chunks ({chunk_seconds}s, {overlap_seconds}s overlap)"
    )
    return chunks


def reassemble_chunks(
    chunks: list[dict],
    overlap_samples: int,
    total_samples: int,
) -> np.ndarray:
    """Reassemble overlapping chunks with linear crossfade.

    Parameters
    ----------
    chunks : list[dict]
        Output of ``chunk_audio``, each with a ``data`` key
        containing the processed audio.
    overlap_samples : int
        Number of overlap samples between adjacent chunks.
    total_samples : int
        Original total sample count (output is trimmed to this).

    Returns
    -------
    np.ndarray
        Reassembled audio, shape ``[channels, total_samples]``.
    """
    if not chunks:
        raise ValueError("No chunks to reassemble")

    channels = chunks[0]["data"].shape[0]
    output = np.zeros((channels, total_samples), dtype=np.float32)

    if len(chunks) == 1:
        length = min(chunks[0]["data"].shape[1], total_samples)
        output[:, :length] = chunks[0]["data"][:, :length]
        return output

    # Linear crossfade ramps
    fade_in = np.linspace(0.0, 1.0, overlap_samples, dtype=np.float32)
    fade_out = np.linspace(1.0, 0.0, overlap_samples, dtype=np.float32)

    for i, chunk in enumerate(chunks):
        data = chunk["data"]
        start = chunk["start_sample"]
        chunk_len = data.shape[1]
        end = min(start + chunk_len, total_samples)
        usable = end - start

        if i == 0:
            # First chunk: no fade-in, fade-out on tail
            segment = data[:, :usable].copy()
            if not chunk["is_last"] and usable > overlap_samples:
                ov_start = usable - overlap_samples
                segment[:, ov_start:usable] *= fade_out[
                    : usable - ov_start
                ]
            output[:, start:end] += segment
        elif chunk["is_last"]:
            # Last chunk: fade-in on head, no fade-out
            segment = data[:, :usable].copy()
            ov_len = min(overlap_samples, usable)
            segment[:, :ov_len] *= fade_in[:ov_len]
            output[:, start:end] += segment
        else:
            # Middle chunk: fade-in on head, fade-out on tail
            segment = data[:, :usable].copy()
            ov_len = min(overlap_samples, usable)
            segment[:, :ov_len] *= fade_in[:ov_len]
            if usable > overlap_samples:
                ov_start = usable - overlap_samples
                segment[:, ov_start:usable] *= fade_out[
                    : usable - ov_start
                ]
            output[:, start:end] += segment

    return output


# ---------------------------------------------------------------------------
# Padding helper
# ---------------------------------------------------------------------------


def pad_to_frames(
    mag: np.ndarray, expected_frames: int = NB_EXPECTED_FRAMES
) -> np.ndarray:
    """Zero-pad magnitude spectrogram along the frame axis.

    Parameters
    ----------
    mag : np.ndarray
        Shape ``[channels, bins, frames]``.
    expected_frames : int
        Target frame count.

    Returns
    -------
    np.ndarray
        Padded array with shape ``[channels, bins, expected_frames]``.
    """
    current_frames = mag.shape[2]
    if current_frames >= expected_frames:
        return mag[:, :, :expected_frames]

    pad_width = expected_frames - current_frames
    return np.pad(mag, ((0, 0), (0, 0), (0, pad_width)))
