# Splitz Research — Stem Separation Engine

ML models and export tooling for on-device audio stem separation in [Splitz](https://github.com/thomasphillips3/splitz).

## Models

ONNX models are hosted on GitHub Pages (`gh-pages` branch) at:
```
https://thomasphillips3.github.io/splitz-research/models/manifest.json
```

### Open-Unmix (4-stem)
| Stem | Size | Quantization | Input Shape |
|------|------|-------------|-------------|
| vocals | 34 MB | fp16 | [1, 2, 2049, 431] |
| drums | 34 MB | fp16 | [1, 2, 2049, 431] |
| bass | 34 MB | fp16 | [1, 2, 2049, 431] |
| other | 34 MB | fp16 | [1, 2, 2049, 431] |

Input: STFT magnitude spectrogram (n_fft=4096, hop=1024, sr=44100)
Output: Estimated stem magnitude (same shape)

## Export

```bash
pip install torch openunmix onnx onnxruntime
python scripts/export_models_onnx.py
```

## Architecture

The Android app handles the full signal processing pipeline:
1. **AudioPreprocessor** → FFmpeg WAV conversion, float32 normalization
2. **StftProcessor** → STFT (Hann window, n_fft=4096, hop=1024)
3. **OnnxInferenceSession** → Run ONNX model on magnitude spectrogram
4. **Wiener filter** → mask_i = |est_i|² / Σ|est_j|² across all stems
5. **StftProcessor** → iSTFT with overlap-add reconstruction
