#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

try:
    import numpy as np
except ImportError:
    np = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare a LoftQ-inspired quantization residual matrix in raw row-major FP32 format."
    )
    parser.add_argument("--model-id", type=str, required=True, help="Hugging Face model id.")
    parser.add_argument("--tensor-name", type=str, required=True, help="Tensor key to extract.")
    parser.add_argument("--output", type=Path, default=Path("matrix.f32"), help="Output .f32 path.")
    parser.add_argument("--meta", type=Path, default=Path("meta.json"), help="Output metadata path.")
    parser.add_argument("--cache-dir", type=Path, default=Path("hf_cache"), help="Cache directory.")
    parser.add_argument(
        "--group-size",
        type=int,
        default=0,
        help="Optional group size along columns; 0 means row-wise quantization.",
    )
    return parser.parse_args()


def require_numpy():
    if np is None:
        raise SystemExit("numpy is required. Install it with `pip install numpy`.")
    return np


def load_tensor(model_id: str, tensor_name: str, cache_dir: Path):
    np_mod = require_numpy()
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise SystemExit(
            "huggingface_hub is required. Install it with `pip install huggingface_hub safetensors`."
        ) from exc

    try:
        from safetensors import safe_open
    except ImportError as exc:
        raise SystemExit(
            "safetensors is required. Install it with `pip install safetensors`."
        ) from exc

    cache_dir.mkdir(parents=True, exist_ok=True)

    try:
        index_path = hf_hub_download(
            repo_id=model_id,
            filename="model.safetensors.index.json",
            cache_dir=str(cache_dir),
        )
    except Exception:
        index_path = None

    if index_path is not None:
        with open(index_path, "r", encoding="utf-8") as handle:
            index_data = json.load(handle)
        weight_map = index_data.get("weight_map", {})
        if tensor_name not in weight_map:
            raise SystemExit(f"Tensor `{tensor_name}` not found in {model_id} safetensors index.")
        shard_name = weight_map[tensor_name]
        shard_path = hf_hub_download(
            repo_id=model_id,
            filename=shard_name,
            cache_dir=str(cache_dir),
        )
        with safe_open(shard_path, framework="pt", device="cpu") as handle:
            tensor = handle.get_tensor(tensor_name)
        return tensor.detach().cpu().numpy().astype(np_mod.float32, copy=False)

    single_path = hf_hub_download(
        repo_id=model_id,
        filename="model.safetensors",
        cache_dir=str(cache_dir),
    )
    with safe_open(single_path, framework="pt", device="cpu") as handle:
        if tensor_name not in handle.keys():
            raise SystemExit(f"Tensor `{tensor_name}` not found in {model_id} safetensors file.")
        tensor = handle.get_tensor(tensor_name)
    return tensor.detach().cpu().numpy().astype(np_mod.float32, copy=False)


def quantize_rowwise_symmetric_int4(weights, group_size: int):
    np_mod = require_numpy()
    weights = np_mod.asarray(weights, dtype=np_mod.float32, order="C")
    if weights.ndim != 2:
        raise SystemExit(f"Expected a 2D tensor, but got shape {weights.shape}.")

    if group_size <= 0:
        scales = np_mod.max(np_mod.abs(weights), axis=1, keepdims=True) / 7.0
        scales[scales == 0.0] = 1.0
        codes = np_mod.clip(np_mod.rint(weights / scales), -7, 7).astype(np_mod.int8)
        return codes.astype(np_mod.float32) * scales

    reconstructed = np_mod.empty_like(weights, dtype=np_mod.float32)
    for col_start in range(0, weights.shape[1], group_size):
        col_end = min(col_start + group_size, weights.shape[1])
        block = weights[:, col_start:col_end]
        scales = np_mod.max(np_mod.abs(block), axis=1, keepdims=True) / 7.0
        scales[scales == 0.0] = 1.0
        codes = np_mod.clip(np_mod.rint(block / scales), -7, 7).astype(np_mod.int8)
        reconstructed[:, col_start:col_end] = codes.astype(np_mod.float32) * scales
    return reconstructed


def write_outputs(
    residual,
    output_path: Path,
    meta_path: Path,
    model_id: str,
    tensor_name: str,
    quantization: str,
    residual_ratio: float,
) -> None:
    np_mod = require_numpy()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.parent.mkdir(parents=True, exist_ok=True)

    residual = np_mod.asarray(residual, dtype=np_mod.float32, order="C")
    residual.tofile(output_path)

    meta = {
        "name": f"loftq_residual_{model_id.replace('/', '_')}_{tensor_name.replace('.', '_')}",
        "rows": int(residual.shape[0]),
        "cols": int(residual.shape[1]),
        "dtype": "float32",
        "layout": "row_major",
        "file": output_path.name,
        "source_model": model_id,
        "source_tensor": tensor_name,
        "matrix_meaning": "R = W - dequantize(quantize(W)), quantization residual of one LLM weight matrix",
        "quantization": quantization,
        "preprocessing": "load W, quantize/dequantize to Q(W), compute residual R",
        "residual_norm_ratio": residual_ratio,
        "notes": "LoftQ-inspired residual SVD benchmark; not a full LoftQ reproduction",
    }
    with meta_path.open("w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)
        handle.write("\n")


def print_sanity_stats(weights, residual, output_path: Path) -> None:
    np_mod = require_numpy()
    expected_bytes = residual.shape[0] * residual.shape[1] * np_mod.dtype(np_mod.float32).itemsize
    actual_bytes = output_path.stat().st_size
    residual_ratio = float(np_mod.linalg.norm(residual) / max(np_mod.linalg.norm(weights), 1e-12))

    print(f"shape: {residual.shape}")
    print(f"dtype: {residual.dtype}")
    print(f"mean: {float(residual.mean()):.8f}")
    print(f"std: {float(residual.std()):.8f}")
    print(f"mean abs: {float(np_mod.mean(np_mod.abs(residual))):.8f}")
    print(f"max abs: {float(np_mod.max(np_mod.abs(residual))):.8f}")
    print(f"has NaN: {bool(np_mod.isnan(residual).any())}")
    print(f"has Inf: {bool(np_mod.isinf(residual).any())}")
    print(f"||R||_F / ||W||_F: {residual_ratio:.8f}")
    print(f"file size bytes: {actual_bytes}")
    print(f"expected bytes: {expected_bytes}")
    if actual_bytes != expected_bytes:
        raise SystemExit(f"File size mismatch for {output_path}.")


def main() -> None:
    args = parse_args()
    np_mod = require_numpy()
    weights = load_tensor(args.model_id, args.tensor_name, args.cache_dir)
    reconstructed = quantize_rowwise_symmetric_int4(weights, args.group_size)
    residual = np_mod.asarray(weights - reconstructed, dtype=np_mod.float32, order="C")

    if np_mod.isnan(residual).any() or np_mod.isinf(residual).any():
        raise SystemExit("Residual matrix contains NaN or Inf after preprocessing.")

    residual_ratio = float(np_mod.linalg.norm(residual) / max(np_mod.linalg.norm(weights), 1e-12))
    quantization = (
        "row-wise symmetric signed 4-bit quantization"
        if args.group_size <= 0
        else f"group-wise symmetric signed 4-bit quantization, group_size={args.group_size}"
    )

    write_outputs(
        residual=residual,
        output_path=args.output,
        meta_path=args.meta,
        model_id=args.model_id,
        tensor_name=args.tensor_name,
        quantization=quantization,
        residual_ratio=residual_ratio,
    )
    print_sanity_stats(weights, residual, args.output)


if __name__ == "__main__":
    main()
