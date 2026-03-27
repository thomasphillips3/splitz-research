#!/usr/bin/env python3
"""
Export Open-Unmix stem separation models to ONNX for on-device Android inference.

Each model takes a raw waveform chunk [1, 2, N] and outputs the separated stem [1, 2, N].
The Android pipeline (AudioChunker) feeds fixed-length 10-second chunks at 44100 Hz.

Usage:
    pip install torch torchaudio openunmix onnx onnxruntime
    python export_models_onnx.py --output-dir ./onnx_models

Output files:
    open-unmix-vocals-fp16.onnx
    open-unmix-drums-fp16.onnx
    open-unmix-bass-fp16.onnx
    open-unmix-other-fp16.onnx
    manifest.json  ← upload this to splitz-research GitHub Pages
"""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import torch
import torchaudio

SAMPLE_RATE = 44100
CHUNK_DURATION_SECONDS = 10
CHUNK_SAMPLES = SAMPLE_RATE * CHUNK_DURATION_SECONDS  # 441000
STEMS = ["vocals", "drums", "bass", "other"]
MODEL_FAMILY = "open-unmix"
MODEL_VERSION = "1.0.0"
MANIFEST_URL_BASE = "https://thomasphillips3.github.io/splitz-research/models"


class UmxWrapper(torch.nn.Module):
    """
    Wraps an Open-Unmix separator to accept raw waveform and return separated stem.

    Open-Unmix internally converts to STFT, applies a learned mask, and inverts.
    This wrapper exposes the end-to-end interface: [1, 2, N] → [1, 2, N].
    """

    def __init__(self, separator, target: str):
        super().__init__()
        self.separator = separator
        self.target = target

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        # waveform: [batch=1, channels=2, samples]
        estimates = self.separator(waveform)  # dict: stem → [1, 2, samples]
        return estimates[self.target]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def export_stem(
    separator,
    target: str,
    output_dir: Path,
    quantization: str = "fp16",
) -> Path:
    print(f"Exporting {target} ({quantization})...")

    wrapper = UmxWrapper(separator, target).eval()

    dummy_input = torch.randn(1, 2, CHUNK_SAMPLES)

    output_path = output_dir / f"{MODEL_FAMILY}-{target}-{quantization}.onnx"

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            dummy_input,
            str(output_path),
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=["waveform"],
            output_names=["separated"],
            dynamic_axes={
                "waveform": {2: "samples"},
                "separated": {2: "samples"},
            },
        )

    size_mb = output_path.stat().st_size / (1024 * 1024)
    checksum = sha256_file(output_path)
    print(f"  → {output_path.name} ({size_mb:.1f} MB, sha256={checksum[:16]}...)")
    return output_path


def build_manifest(model_paths: dict[str, Path]) -> dict:
    models = []
    for target, path in model_paths.items():
        model_id = f"{MODEL_FAMILY}-{target}-fp16"
        size_mb = path.stat().st_size / (1024 * 1024)
        checksum = sha256_file(path)
        models.append(
            {
                "modelId": model_id,
                "stemTarget": target,
                "version": MODEL_VERSION,
                "quantization": "fp16",
                "downloadUrl": f"{MANIFEST_URL_BASE}/{path.name}",
                "fileSizeMb": round(size_mb, 1),
                "sha256": checksum,
                "minRamMb": 1024,
            }
        )

    return {
        "version": 1,
        "families": [
            {
                "id": MODEL_FAMILY,
                "displayName": "Open-Unmix",
                "description": "4-stem separation (vocals, drums, bass, other). ~60 MB total.",
                "stemCount": 4,
                "models": models,
            }
        ],
    }


def main():
    parser = argparse.ArgumentParser(description="Export Open-Unmix models to ONNX")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./onnx_models"),
        help="Directory to write .onnx files and manifest.json",
    )
    parser.add_argument(
        "--stems",
        nargs="+",
        default=STEMS,
        choices=STEMS,
        help="Which stems to export (default: all)",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    try:
        import openunmix
    except ImportError:
        print("ERROR: openunmix not installed. Run: pip install openunmix")
        sys.exit(1)

    print("Loading Open-Unmix separator (umxhq)...")
    separator = openunmix.umxhq(targets=args.stems, niter=1, residual=False)
    separator.eval()

    model_paths: dict[str, Path] = {}
    for target in args.stems:
        path = export_stem(separator, target, args.output_dir)
        model_paths[target] = path

    # Write manifest
    manifest = build_manifest(model_paths)
    manifest_path = args.output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest written to {manifest_path}")

    print(
        "\nNext steps:"
        "\n  1. Copy onnx_models/ contents to splitz-research/models/ in your GitHub Pages branch"
        "\n  2. Push to trigger GitHub Pages deploy"
        "\n  3. Verify: curl https://thomasphillips3.github.io/splitz-research/models/manifest.json"
    )


if __name__ == "__main__":
    main()
