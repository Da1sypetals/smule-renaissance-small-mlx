#!/usr/bin/env python3
"""Record major PyTorch SRS boundaries on a short saved STFT tensor."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_stft", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--frames", type=int, default=9)
    parser.add_argument(
        "--python-source",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "smule-renaissance",
    )
    return parser.parse_args()


def save(directory: Path, name: str, value: torch.Tensor) -> None:
    array = value.detach().float().cpu().contiguous().numpy()
    np.save(directory / f"{name}.npy", array)
    print(
        f"{name}: shape={list(array.shape)} min={array.min():.9g} "
        f"max={array.max():.9g} mean={array.mean():.9g}"
    )


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(args.python_source.resolve()))
    from model import BSNet, Renaissance

    args.output_directory.mkdir(parents=True, exist_ok=True)
    natural = torch.from_numpy(np.load(args.input_stft))[:, : args.frames]
    canonical = natural.permute(0, 2, 1, 3).contiguous()
    np.save(args.output_directory / "input.npy", natural.numpy())

    model = Renaissance()
    model.load_state_dict(
        torch.load(args.checkpoint, map_location="cpu", weights_only=True), strict=True
    )
    model.eval()

    handles = []
    for index, module in enumerate(model.feature_extractor_layers):
        handles.append(
            module.register_forward_hook(
                lambda _module, _inputs, output, index=index: save(
                    args.output_directory,
                    f"feature_extractor_layers.{index}",
                    output.transpose(1, 2),
                )
            )
        )
    for index, module in enumerate(model.net):
        assert isinstance(module, BSNet)
        handles.append(
            module.register_forward_hook(
                lambda _module, _inputs, output, index=index: save(
                    args.output_directory,
                    f"net.{index}",
                    output.permute(0, 3, 1, 2),
                )
            )
        )
    for index, module in enumerate(model.output_layers):
        width = model.band_width[index]
        handles.append(
            module.register_forward_hook(
                lambda _module, _inputs, output, index=index, width=width: save(
                    args.output_directory,
                    f"output_layers.{index}",
                    output.view(output.shape[0], width, 2, output.shape[-1]).permute(0, 3, 1, 2),
                )
            )
        )

    with torch.inference_mode():
        features = model.feature_extraction(canonical)
        save(args.output_directory, "features", features.permute(0, 3, 1, 2))
        processed = features
        for layer in model.net:
            processed = layer(processed)
        processed = processed + features
        save(args.output_directory, "processed", processed.permute(0, 3, 1, 2))

        bands = []
        for index, module in enumerate(model.output_layers):
            output = module(processed[:, index])
            width = model.band_width[index]
            bands.append(output.view(output.shape[0], width, 2, output.shape[-1]).permute(0, 1, 3, 2))
        output = torch.cat(bands, dim=1)
        save(args.output_directory, "output", output.permute(0, 2, 1, 3))

    for handle in handles:
        handle.remove()


if __name__ == "__main__":
    main()
