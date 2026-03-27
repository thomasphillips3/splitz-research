"""
CLI entry point for the Splitz benchmark.

Usage::

    python -m benchmark.cli \\
        --config experiments.yaml \\
        --experiment wiener_power_sweep \\
        --output results/wiener.json

    # With real audio (no SDR — relative metrics only):
    python -m benchmark.cli \\
        --config experiments.yaml \\
        --experiment baseline \\
        --audio test_tracks/song.wav \\
        --output results/baseline.json
"""

import argparse
import datetime
import json
import logging
import platform
import sys
from pathlib import Path

import yaml

from .runner import run_experiment
from .test_signal import generate_test_signal

logger = logging.getLogger(__name__)


def _setup_logging(verbose: bool = False) -> None:
    """Configure root logger for CLI output."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _load_config(config_path: str) -> dict:
    """Load and parse a YAML configuration file."""
    path = Path(config_path)
    if not path.exists():
        logger.error(f"Config file not found: {config_path}")
        sys.exit(1)

    with open(path) as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        logger.error(f"Invalid config format in {config_path}")
        sys.exit(1)

    return config


def _find_experiment(config: dict, name: str) -> dict:
    """Find an experiment by name in the config."""
    experiments = config.get("experiments", [])
    for exp in experiments:
        if exp.get("name") == name:
            return exp

    available = [e.get("name", "?") for e in experiments]
    logger.error(
        f"Experiment '{name}' not found in config. "
        f"Available: {available}"
    )
    sys.exit(1)


def _build_metadata() -> dict:
    """Collect system metadata for the results file."""
    return {
        "timestamp": datetime.datetime.now(
            tz=datetime.timezone.utc
        ).isoformat(),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python": platform.python_version(),
        },
        "benchmark_version": "1.0.0",
    }


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Splitz Benchmark — stem separation quality measurement",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Example:\n"
            "  python -m benchmark.cli \\\n"
            "    --config experiments.yaml \\\n"
            "    --experiment baseline \\\n"
            "    --output results/baseline.json\n"
        ),
    )
    parser.add_argument(
        "--config",
        required=True,
        help="YAML config file with experiment definitions",
    )
    parser.add_argument(
        "--experiment",
        required=True,
        help="Experiment name from the config file",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output JSON path for results",
    )
    parser.add_argument(
        "--audio",
        help=(
            "Path to test audio WAV file. "
            "If omitted, a synthetic test signal is generated "
            "(with reference stems for SDR/SIR/SAR)."
        ),
    )
    parser.add_argument(
        "--cache-dir",
        help="Model cache directory (default: ~/.cache/splitz-research/models/)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable debug-level logging",
    )

    args = parser.parse_args()
    _setup_logging(args.verbose)

    logger.info("Splitz Benchmark starting")

    # Load config and find experiment
    config = _load_config(args.config)
    experiment = _find_experiment(config, args.experiment)

    cache_dir = Path(args.cache_dir) if args.cache_dir else None

    # Determine audio source
    audio_path = None
    audio_array = None
    reference_stems = None

    if args.audio:
        audio_path = args.audio
        audio_file = Path(audio_path)
        if not audio_file.exists():
            logger.error(f"Audio file not found: {audio_path}")
            sys.exit(1)
        logger.info(
            f"Using real audio: {audio_path} "
            f"(relative metrics only — no ground truth)"
        )
    else:
        duration = experiment.get("synthetic_duration", 15.0)
        logger.info(
            f"Generating synthetic test signal "
            f"({duration}s, with ground-truth stems)"
        )
        audio_array, reference_stems = generate_test_signal(
            duration=duration
        )

    # Run experiment
    results = run_experiment(
        config=experiment,
        audio_path=audio_path,
        audio_array=audio_array,
        reference_stems=reference_stems,
        cache_dir=cache_dir,
    )

    # Attach metadata
    results["metadata"] = _build_metadata()

    # Write output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"Results written to {output_path}")
    logger.info(
        f"Total time: {results['timings']['total_ms']:.0f} ms"
    )

    # Print summary to stdout
    _print_summary(results)


def _print_summary(results: dict) -> None:
    """Print a human-readable results summary."""
    metrics = results.get("metrics", {})

    sys.stdout.write("\n")
    sys.stdout.write(
        f"{'=' * 60}\n"
        f"  Experiment: {results['experiment']}\n"
        f"  Audio: {results['audio']['source']} "
        f"({results['audio']['duration_seconds']:.1f}s)\n"
        f"  Chunks: {results['n_chunks']}\n"
        f"{'=' * 60}\n\n"
    )

    # Per-stem metrics
    header = f"  {'Stem':<10} {'Energy':>8}"
    has_sdr = any(
        "sdr" in metrics.get(s, {})
        for s in ["vocals", "drums", "bass", "other"]
    )
    if has_sdr:
        header += f" {'SDR(dB)':>9} {'SIR(dB)':>9} {'SAR(dB)':>9}"
    sys.stdout.write(f"{header}\n")
    sys.stdout.write(f"  {'-' * len(header.strip())}\n")

    for stem in ["vocals", "drums", "bass", "other"]:
        if stem not in metrics:
            continue
        m = metrics[stem]
        line = f"  {stem:<10} {m.get('energy_ratio', 0.0):>8.4f}"
        if has_sdr and "sdr" in m:
            line += (
                f" {m['sdr']:>9.2f} {m['sir']:>9.2f} "
                f"{m['sar']:>9.2f}"
            )
        sys.stdout.write(f"{line}\n")

    # Aggregate metrics
    sys.stdout.write(f"\n  Reconstruction error: ")
    sys.stdout.write(f"{metrics.get('reconstruction_error', 0.0):.6f}\n")

    leakage = metrics.get("cross_stem_leakage", {})
    if leakage:
        max_pair = max(leakage, key=lambda k: abs(leakage[k]))
        sys.stdout.write(
            f"  Max cross-stem leakage: "
            f"{max_pair} = {leakage[max_pair]:.4f}\n"
        )

    # Timings
    timings = results.get("timings", {})
    sys.stdout.write(
        f"\n  Total time: {timings.get('total_ms', 0.0):.0f} ms\n"
        f"  Model load: {timings.get('model_load_ms', 0.0):.0f} ms\n"
        f"  Inference: {timings.get('inference_total_ms', 0.0):.0f} ms "
        f"({timings.get('inference_per_chunk_ms', 0.0):.0f} ms/chunk)\n"
    )
    sys.stdout.write(f"{'=' * 60}\n\n")


if __name__ == "__main__":
    main()
