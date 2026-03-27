"""
Synthetic test signal generator for benchmark evaluation.

Generates a stereo mix from four known stem components (vocals, drums,
bass, other) so that ground-truth SDR/SIR/SAR can be computed against
the separation output.

Each stem occupies a distinct spectral region to give the separator
a realistic (but solvable) challenge:

- **vocals**: sine sweeps 200-2000 Hz with vibrato, panned slightly left
- **drums**: periodic impulse bursts at ~120 BPM, centered
- **bass**: 80 Hz fundamental with subtle harmonics, centered
- **other**: bandpass-filtered pink noise bursts (1-8 kHz), panned right
"""

import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pink_noise(n: int, rng: np.random.Generator) -> np.ndarray:
    """Generate approximate pink noise (1/f) via spectral shaping."""
    white = rng.standard_normal(n)
    spectrum = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n)
    freqs[0] = 1.0  # avoid division by zero
    spectrum /= np.sqrt(freqs)
    return np.fft.irfft(spectrum, n=n).astype(np.float64)


def _bandpass(
    signal: np.ndarray,
    low_hz: float,
    high_hz: float,
    sr: int,
) -> np.ndarray:
    """Simple spectral bandpass filter."""
    n = len(signal)
    spectrum = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(n, d=1.0 / sr)
    mask = ((freqs >= low_hz) & (freqs <= high_hz)).astype(
        np.float64
    )
    # Smooth edges to reduce ringing
    edge_width = max(1, int(n * 50.0 / sr))
    for i in range(len(mask)):
        if mask[i] == 1.0 and i > 0 and mask[i - 1] == 0.0:
            for j in range(max(0, i - edge_width), i):
                mask[j] = (j - (i - edge_width)) / edge_width
        if mask[i] == 0.0 and i > 0 and mask[i - 1] == 1.0:
            for j in range(i, min(len(mask), i + edge_width)):
                mask[j] = 1.0 - (j - i) / edge_width
    spectrum *= mask
    return np.fft.irfft(spectrum, n=n).astype(np.float64)


def _to_stereo(
    mono: np.ndarray, pan: float = 0.0
) -> np.ndarray:
    """Convert mono to stereo with panning.

    Parameters
    ----------
    mono : np.ndarray
        1-D mono signal.
    pan : float
        -1.0 = hard left, 0.0 = center, 1.0 = hard right.

    Returns
    -------
    np.ndarray
        Shape ``[2, samples]``.
    """
    # Constant-power panning
    angle = (pan + 1.0) * np.pi / 4.0  # 0..pi/2
    left_gain = np.cos(angle)
    right_gain = np.sin(angle)
    return np.stack(
        [mono * left_gain, mono * right_gain], axis=0
    ).astype(np.float32)


# ---------------------------------------------------------------------------
# Stem generators
# ---------------------------------------------------------------------------


def _generate_vocals(
    duration: float, sr: int, rng: np.random.Generator
) -> np.ndarray:
    """Sine sweeps 200-2000 Hz with vibrato, panned slightly left."""
    n = int(duration * sr)
    t = np.linspace(0.0, duration, n, dtype=np.float64)

    # Sweep: log frequency from 200 to 2000 Hz over the duration
    f_start, f_end = 200.0, 2000.0
    phase = (
        2.0
        * np.pi
        * f_start
        * duration
        * (
            np.exp(t / duration * np.log(f_end / f_start)) - 1.0
        )
        / np.log(f_end / f_start)
    )

    # AM vibrato at 5 Hz
    vibrato = 1.0 + 0.3 * np.sin(2.0 * np.pi * 5.0 * t)
    mono = (np.sin(phase) * vibrato * 0.5).astype(np.float64)

    # Fade in/out to avoid clicks
    fade = int(0.05 * sr)
    mono[:fade] *= np.linspace(0.0, 1.0, fade)
    mono[-fade:] *= np.linspace(1.0, 0.0, fade)

    return _to_stereo(mono, pan=-0.3)


def _generate_drums(
    duration: float, sr: int, rng: np.random.Generator
) -> np.ndarray:
    """Periodic impulse bursts at ~120 BPM, sharp transients, centered."""
    n = int(duration * sr)
    mono = np.zeros(n, dtype=np.float64)

    bpm = 120.0
    beat_samples = int(60.0 / bpm * sr)
    hit_length = int(0.02 * sr)  # 20ms transient

    pos = 0
    while pos < n:
        end = min(pos + hit_length, n)
        length = end - pos
        # Exponentially decaying noise burst
        burst = rng.standard_normal(length)
        envelope = np.exp(-np.linspace(0.0, 8.0, length))
        mono[pos:end] += burst * envelope * 0.7
        pos += beat_samples

    return _to_stereo(mono, pan=0.0)


def _generate_bass(
    duration: float, sr: int, rng: np.random.Generator
) -> np.ndarray:
    """Low sine at 80 Hz with subtle harmonics, centered."""
    n = int(duration * sr)
    t = np.linspace(0.0, duration, n, dtype=np.float64)

    fundamental = 80.0
    mono = (
        0.6 * np.sin(2.0 * np.pi * fundamental * t)
        + 0.15 * np.sin(2.0 * np.pi * fundamental * 2 * t)
        + 0.05 * np.sin(2.0 * np.pi * fundamental * 3 * t)
    )

    # Gentle fade
    fade = int(0.05 * sr)
    mono[:fade] *= np.linspace(0.0, 1.0, fade)
    mono[-fade:] *= np.linspace(1.0, 0.0, fade)

    return _to_stereo(mono, pan=0.0)


def _generate_other(
    duration: float, sr: int, rng: np.random.Generator
) -> np.ndarray:
    """Bandpass-filtered pink noise bursts (1-8 kHz), panned right."""
    n = int(duration * sr)
    noise = _pink_noise(n, rng)
    filtered = _bandpass(noise, 1000.0, 8000.0, sr)

    # Rhythmic gating: 2 Hz on/off
    t = np.linspace(0.0, duration, n, dtype=np.float64)
    gate = (np.sin(2.0 * np.pi * 2.0 * t) > 0.0).astype(
        np.float64
    )
    # Smooth the gate to avoid clicks
    kernel_size = int(0.01 * sr)
    if kernel_size > 1:
        kernel = np.ones(kernel_size) / kernel_size
        gate = np.convolve(gate, kernel, mode="same")

    mono = (filtered * gate * 0.3).astype(np.float64)

    return _to_stereo(mono, pan=0.5)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_test_signal(
    duration: float = 15.0, sr: int = 44100, seed: int = 42
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Generate a synthetic stereo mix with known stem components.

    Parameters
    ----------
    duration : float
        Signal duration in seconds.
    sr : int
        Sample rate.
    seed : int
        RNG seed for reproducibility.

    Returns
    -------
    tuple[np.ndarray, dict[str, np.ndarray]]
        ``(mix, stems)`` where *mix* has shape ``[2, samples]`` and
        *stems* maps ``"vocals"``, ``"drums"``, ``"bass"``,
        ``"other"`` to arrays of the same shape.
    """
    rng = np.random.default_rng(seed)

    vocals = _generate_vocals(duration, sr, rng)
    drums = _generate_drums(duration, sr, rng)
    bass = _generate_bass(duration, sr, rng)
    other = _generate_other(duration, sr, rng)

    mix = (vocals + drums + bass + other).astype(np.float32)

    stems = {
        "vocals": vocals.astype(np.float32),
        "drums": drums.astype(np.float32),
        "bass": bass.astype(np.float32),
        "other": other.astype(np.float32),
    }

    return mix, stems
