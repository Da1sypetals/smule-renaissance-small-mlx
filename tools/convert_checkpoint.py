#!/usr/bin/env python3
"""Convert the released PyTorch SRS checkpoint to MLX-native safetensors.

Convolution kernels are transposed while saving so the Rust runtime never
performs checkpoint-layout conversion after loading.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--python-source",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "smule-renaissance",
        help="Directory containing the reference model.py",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(args.python_source.resolve()))
    from model import Renaissance

    checkpoint = args.checkpoint.resolve()
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    reference = Renaissance()
    reference.load_state_dict(state, strict=True)

    conv_weights = {
        f"{name}.weight"
        for name, module in reference.named_modules()
        if isinstance(module, torch.nn.Conv1d)
    }

    converted: dict[str, torch.Tensor] = {}
    for name, tensor in state.items():
        value = tensor.detach().contiguous()
        if name in conv_weights:
            if value.ndim != 3:
                raise ValueError(f"Expected rank-3 Conv1d weight {name}, got {value.shape}")
            value = value.permute(0, 2, 1).contiguous()
        converted[name] = value

    missing = conv_weights.difference(converted)
    if missing:
        raise KeyError(f"Checkpoint is missing Conv1d weights: {sorted(missing)}")

    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    metadata = {
        "architecture": "Smule Renaissance Small",
        "source_format": "pytorch_state_dict",
        "source_sha256": digest,
        "conv1d_weight_layout": "out_kernel_in_per_group",
        "linear_weight_layout": "out_in",
        "dtype": "float32",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file(converted, args.output, metadata=metadata)

    total_parameters = sum(tensor.numel() for tensor in converted.values())
    print(f"Converted {len(converted)} tensors ({total_parameters:,} values)")
    print(f"Transposed {len(conv_weights)} Conv1d kernels to MLX layout")
    print(f"Source SHA-256: {digest}")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
