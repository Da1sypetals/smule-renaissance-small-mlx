#!/usr/bin/env python3
"""Generate PyTorch reference tensors for Rust/MLX numerical alignment."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torchaudio


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument(
        "--python-source",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "smule-renaissance",
    )
    return parser.parse_args()


def save(directory: Path, name: str, tensor: torch.Tensor) -> None:
    value = tensor.detach().cpu().contiguous().numpy()
    np.save(directory / f"{name}.npy", value)
    finite = np.isfinite(value)
    print(
        f"{name}: shape={list(value.shape)} dtype={value.dtype} "
        f"min={value[finite].min():.9g} max={value[finite].max():.9g} "
        f"mean={value[finite].mean():.9g} rms={np.sqrt(np.mean(value[finite] ** 2)):.9g}"
    )


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(args.python_source.resolve()))
    from model import Renaissance
    from spectral_ops import STFT, iSTFT

    args.output_directory.mkdir(parents=True, exist_ok=True)
    waveform, sample_rate = torchaudio.load(args.input)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sample_rate != 48_000:
        waveform = torchaudio.transforms.Resample(sample_rate, 48_000)(waveform)
    waveform = torchaudio.functional.highpass_biquad(waveform, 48_000, cutoff_freq=60.0)
    save(args.output_directory, "preprocessed", waveform)

    normalization_factor = waveform.abs().max()
    normalized = waveform / normalization_factor if normalization_factor > 0 else waveform
    save(args.output_directory, "normalized", normalized)

    stft = STFT(4096, 2048, 4096)
    input_stft = stft(normalized)
    input_stft_natural = input_stft.permute(0, 2, 1, 3).contiguous()
    save(args.output_directory, "input_stft", input_stft_natural)

    model = Renaissance()
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    with torch.inference_mode():
        enhanced_stft = model(input_stft)
    enhanced_stft_natural = enhanced_stft.permute(0, 2, 1, 3).contiguous()
    save(args.output_directory, "enhanced_stft", enhanced_stft_natural)

    enhanced = iSTFT(4096, 2048, 4096)(enhanced_stft) * normalization_factor
    save(args.output_directory, "enhanced_waveform", enhanced)


if __name__ == "__main__":
    main()
