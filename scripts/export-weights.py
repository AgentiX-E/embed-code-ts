#!/usr/bin/env python3
"""
Export nomic-embed-code weights from HuggingFace to embed-code-ts binary format.

Extracts all PyTorch model weights, quantizes them to int8 (per-channel),
and writes a single binary file with header + offset table + payload.

Generates .weights.bin format (legacy format, ONNX is the current approach).
The output file is directly consumable by WeightBuffer in TypeScript.

Usage:
    python3 scripts/export-weights.py \
        --model nomic-ai/nomic-embed-text-v1.5 \
        --output models/nomic-embed-code-v1-int8.weights.bin

Requirements:
    pip install torch transformers
"""

import argparse
import hashlib
import json
import os
import struct
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import OrderedDict

import numpy as np
import torch

# ─── Binary format constants ──────────────────────────────────────────

MAGIC = b"EMBCODE1"
HEADER_SIZE = 256
ENTRY_NAME_LEN = 64
ENTRY_SIZE = 114  # 64(name) + 8(offset) + 8(size) + 2(rank) + 8(dims) + 8(scale_off) + 16(reserved)
VERSION = 1


# ─── Quantization ─────────────────────────────────────────────────────


def quantize_per_channel(w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Quantize float32 weights to int8, per output channel.

    For a weight matrix [out_features, in_features]:
        scale[i] = max(|w[i, :]|) / 127.0
        w_int8[i, j] = round(clip(w[i, j] / scale[i], -128, 127))

    Returns (w_int8, scales) where w_int8 has shape [out_features, in_features]
    and scales has shape [out_features].
    """
    n = w.shape[0]  # output channels (first dimension)
    flat = w.reshape(n, -1)

    # Compute max absolute value per row (output channel)
    max_abs = np.max(np.abs(flat), axis=1)
    scales = max_abs / 127.0
    scales = np.maximum(scales, 1e-8)  # avoid division by zero

    # Quantize
    w_int8 = np.clip(np.round(flat / scales[:, np.newaxis]), -128, 127).astype(np.int8)

    return w_int8.reshape(w.shape), scales.astype(np.float32)


def quantize_1d(w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Quantize 1D bias/layernorm to shared-channel int8."""
    n = w.shape[0]
    max_abs = np.max(np.abs(w))
    scale = np.full((n,), max(max_abs / 127.0, 1e-8), dtype=np.float32)
    w_int8 = np.clip(np.round(w / scale), -128, 127).astype(np.int8)
    return w_int8, scale


# ─── Binary writer ────────────────────────────────────────────────────


class BinaryWriter:
    def __init__(self):
        self.header = bytearray(HEADER_SIZE)
        self.entries: list[bytes] = []
        self.payload = bytearray()
        # Track current position in payload for offset calculation
        self._payload_pos = 0

    def write_header(self, arch_params: dict):
        h = self.header
        h[0:8] = MAGIC
        struct.pack_into("<I", h, 8, VERSION)
        struct.pack_into("<H", h, 12, arch_params["num_layers"])
        struct.pack_into("<H", h, 14, arch_params["hidden_size"])
        struct.pack_into("<I", h, 16, arch_params["intermediate_size"])
        struct.pack_into("<I", h, 20, arch_params["vocab_size"])
        struct.pack_into("<I", h, 24, arch_params["max_positions"])
        struct.pack_into("<H", h, 28, arch_params["num_heads"])
        struct.pack_into("<H", h, 30, arch_params["head_dim"])

    def set_total_size(self, size: int):
        struct.pack_into("<Q", self.header, 32, size)

    def add_tensor(self, name: str, data: np.ndarray, scales: np.ndarray):
        """Add one weight tensor + its per-channel scales to the payload."""
        tensor_data = data.tobytes()
        scale_data = scales.tobytes()
        tensor_offset = self._payload_pos
        scale_offset = tensor_offset + len(tensor_data)

        # Append to payload
        self.payload.extend(tensor_data)
        self.payload.extend(scale_data)
        self._payload_pos += len(tensor_data) + len(scale_data)

        # Build entry
        rank = data.ndim
        dims = list(data.shape)
        while len(dims) < 4:
            dims.append(0)

        entry = bytearray(ENTRY_SIZE)
        name_bytes = name.encode("ascii")[: ENTRY_NAME_LEN - 1]
        entry[0 : len(name_bytes)] = name_bytes
        struct.pack_into("<Q", entry, 64, tensor_offset)
        struct.pack_into("<Q", entry, 72, len(tensor_data))
        struct.pack_into("<H", entry, 80, rank)
        for i, d in enumerate(dims[:4]):
            struct.pack_into("<H", entry, 82 + i * 2, d)
        struct.pack_into("<Q", entry, 90, scale_offset)

        self.entries.append(bytes(entry))

    def finalize(self, output_path: Path) -> int:
        """Write the complete binary file."""
        # Compute offsets
        table_start = HEADER_SIZE
        table_size = len(self.entries) * ENTRY_SIZE
        payload_abs = table_start + table_size

        # Update each entry's offsets to absolute
        finalized = []
        for entry_bytes in self.entries:
            entry = bytearray(entry_bytes)
            rel_tensor_off = struct.unpack_from("<Q", entry, 64)[0]
            rel_scale_off = struct.unpack_from("<Q", entry, 90)[0]
            struct.pack_into("<Q", entry, 64, payload_abs + rel_tensor_off)
            struct.pack_into("<Q", entry, 90, payload_abs + rel_scale_off)
            finalized.append(bytes(entry))

        total_size = payload_abs + len(self.payload)
        self.set_total_size(total_size)

        # Write
        with open(output_path, "wb") as f:
            f.write(self.header)
            for e in finalized:
                f.write(e)
            f.write(self.payload)

        return total_size


# ─── Main export logic ─────────────────────────────────────────────────


def resolve_arch_params(model) -> dict:
    """Extract architecture parameters from a HuggingFace model."""
    config = model.config
    return {
        "num_layers": getattr(config, "num_hidden_layers", 12),
        "hidden_size": getattr(config, "hidden_size", 768),
        "intermediate_size": getattr(config, "intermediate_size", 3072),
        "vocab_size": getattr(config, "vocab_size", 30522),
        "max_positions": getattr(config, "max_position_embeddings", 8192),
        "num_heads": getattr(config, "num_attention_heads", 12),
        "head_dim": getattr(config, "hidden_size", 768)
        // getattr(config, "num_attention_heads", 12),
    }


def remap_weight_name(name: str) -> str | list[tuple[str, int, int]]:
    """
    Map PyTorch weight names to our canonical embed-code-ts names.
    
    Returns either a string (direct mapping) or a list of (name, start_row, end_row)
    tuples for weights that need splitting (e.g., fused QKV).
    """
    # Embedding layer norm (Nomic calls it emb_ln)
    if name == "emb_ln.weight":
        return "embeddings.LayerNorm.weight"
    if name == "emb_ln.bias":
        return "embeddings.LayerNorm.bias"
    
    # Position embeddings
    if name == "embeddings.position_embeddings.weight":
        return "embeddings.position_embeddings.weight"
    
    # Encoder layers
    import re
    m = re.match(r"encoder\.layers\.(\d+)\.(.+)", name)
    if m:
        layer_num = int(m.group(1))
        sub = m.group(2)
        prefix = f"encoder.layer.{layer_num}"
        
        # Fused QKV → split into query, key, value
        if sub == "attn.Wqkv.weight":
            return [
                (f"{prefix}.attention.self.query.weight", 0, 768),     # first 768 rows
                (f"{prefix}.attention.self.key.weight", 768, 1536),     # next 768 rows
                (f"{prefix}.attention.self.value.weight", 1536, 2304),  # last 768 rows
            ]
        if sub == "attn.Wqkv.bias":
            return [
                (f"{prefix}.attention.self.query.bias", 0, 768),
                (f"{prefix}.attention.self.key.bias", 768, 1536),
                (f"{prefix}.attention.self.value.bias", 1536, 2304),
            ]
        
        # Output projection
        if sub == "attn.out_proj.weight":
            return f"{prefix}.attention.output.dense.weight"
        if sub == "attn.out_proj.bias":
            return f"{prefix}.attention.output.dense.bias"
        
        # LayerNorm: Nomic uses norm1 (attention) and norm2 (FFN)
        if sub == "norm1.weight":
            return f"{prefix}.attention.output.LayerNorm.weight"
        if sub == "norm1.bias":
            return f"{prefix}.attention.output.LayerNorm.bias"
        if sub == "norm2.weight":
            return f"{prefix}.output.LayerNorm.weight"
        if sub == "norm2.bias":
            return f"{prefix}.output.LayerNorm.bias"
        
        # Legacy naming (also used by some models)
        if sub == "attn_norm.weight":
            return f"{prefix}.attention.output.LayerNorm.weight"
        if sub == "attn_norm.bias":
            return f"{prefix}.attention.output.LayerNorm.bias"
        
        # FFN
        if sub == "mlp.fc1.weight" or sub == "mlp.fc11.weight":
            return f"{prefix}.intermediate.dense.weight"
        if sub == "mlp.fc1.bias" or sub == "mlp.fc11.bias":
            return f"{prefix}.intermediate.dense.bias"
        if sub == "mlp.fc2.weight" or sub == "mlp.fc12.weight":
            return f"{prefix}.output.dense.weight"
        if sub == "mlp.fc2.bias" or sub == "mlp.fc12.bias":
            return f"{prefix}.output.dense.bias"
        
        # FFN LayerNorm
        if sub == "mlp_norm.weight":
            return f"{prefix}.output.LayerNorm.weight"
        if sub == "mlp_norm.bias":
            return f"{prefix}.output.LayerNorm.bias"
    
    # Default: keep as-is (already canonical or unknown)
    return name


def export_weights(model_id: str, output_path: Path, skip_large: bool = False):
    """Main export function."""
    print(f"\n📦 Exporting {model_id} to embed-code-ts binary format")
    print(f"   Output: {output_path}")
    print()

    # 1. Load model
    print("  Step 1: Loading model from HuggingFace...")
    from transformers import AutoModel, AutoTokenizer

    model = AutoModel.from_pretrained(model_id, torch_dtype=torch.float32, trust_remote_code=True)
    model.eval()

    arch = resolve_arch_params(model)
    print(f"  Architecture: {arch['num_layers']} layers, {arch['hidden_size']} hidden, "
          f"{arch['num_heads']} heads, {arch['vocab_size']} vocab")
    print(f"  Estimated int8 size: ~{sum(p.numel() for p in model.parameters()) / 1e6:.0f}M params")

    # 2. Export weights
    print("\n  Step 2: Quantizing and packing weights...")
    writer = BinaryWriter()
    writer.write_header(arch)

    tensor_count = 0
    total_bytes = 0

    state_dict = model.state_dict()
    for name, param in state_dict.items():
        canonical = remap_weight_name(name)
        data = param.detach().cpu().numpy()

        # Handle fused QKV split: returns list of (sub_name, start, end)
        if isinstance(canonical, list):
            for sub_name, start, end in canonical:
                sub_data = data[start:end]
                if sub_data.ndim >= 2:
                    w_int8, scales = quantize_per_channel(sub_data)
                else:
                    w_int8, scales = quantize_1d(sub_data)
                writer.add_tensor(sub_name, w_int8, scales)
                tensor_count += 1
                total_bytes += w_int8.nbytes + scales.nbytes
        else:
            if data.ndim >= 2:
                w_int8, scales = quantize_per_channel(data)
            else:
                w_int8, scales = quantize_1d(data)
            writer.add_tensor(canonical, w_int8, scales)
            tensor_count += 1
            total_bytes += w_int8.nbytes + scales.nbytes

        if tensor_count % 20 == 0:
            print(f"    {tensor_count} tensors... ({total_bytes / 1024**2:.0f} MB)")

    print(f"    {tensor_count} tensors total, {total_bytes / 1024**2:.1f} MB int8 weights")

    # 3. Finalize and write
    print("\n  Step 3: Writing binary file...")
    file_size = writer.finalize(output_path)

    print(f"\n✅ Export complete: {output_path}")
    print(f"   Size: {file_size / 1024**2:.1f} MB")
    print(f"   Tensors: {tensor_count}")
    print(f"   Architecture: {arch['num_layers']} layers × {arch['hidden_size']} hidden")

    # 4. Update model-descriptor.json
    update_descriptor(output_path, model_id, arch, file_size)


def update_descriptor(output_path: Path, model_id: str, arch: dict, file_size: int):
    """Update model-descriptor.json with export metadata."""
    descriptor_path = output_path.parent / "model-descriptor.json"

    sha256 = hashlib.sha256()
    with open(output_path, "rb") as f:
        while chunk := f.read(64 * 1024):
            sha256.update(chunk)

    if descriptor_path.exists():
        with open(descriptor_path) as f:
            descriptor = json.load(f)
    else:
        descriptor = {}

    # Update weights section
    if "weights" not in descriptor:
        descriptor["weights"] = {}
    descriptor["weights"]["sha256"] = sha256.hexdigest()
    descriptor["weights"]["size_bytes"] = file_size
    descriptor["weights"]["input_ids_name"] = "input_ids"
    descriptor["weights"]["attention_mask_name"] = "attention_mask"
    descriptor["weights"]["output_name"] = "last_hidden_state"

    # Update architecture
    if "architecture" not in descriptor:
        descriptor["architecture"] = {}
    descriptor["architecture"]["num_layers"] = arch["num_layers"]
    descriptor["architecture"]["num_heads"] = arch["num_heads"]
    descriptor["architecture"]["hidden_size"] = arch["hidden_size"]
    descriptor["architecture"]["intermediate_size"] = arch["intermediate_size"]
    descriptor["architecture"]["vocab_size"] = arch["vocab_size"]
    descriptor["architecture"]["head_dim"] = arch["head_dim"]
    descriptor["architecture"]["embedding_dim"] = arch["hidden_size"]
    descriptor["architecture"]["max_position_embeddings"] = arch["max_positions"]

    # Update model section
    if "model" not in descriptor:
        descriptor["model"] = {}
    descriptor["model"]["hf_repository"] = model_id
    descriptor["model"]["exported_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    descriptor["model"]["precision"] = "int8"
    descriptor["model"]["base_architecture"] = "BERT-base"

    with open(descriptor_path, "w") as f:
        json.dump(descriptor, f, indent=2)
        f.write("\n")

    print(f"  ✓ Updated descriptor: {descriptor_path}")
    print(f"    SHA256: {sha256.hexdigest()}")


def main():
    parser = argparse.ArgumentParser(description="Export nomic-embed-code to int8 binary format")
    parser.add_argument("--output", "-o", required=True, help="Output binary file path")
    parser.add_argument(
        "--model", "-m", default="nomic-ai/nomic-embed-text-v1.5", help="HuggingFace model ID"
    )
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        export_weights(args.model, output_path)
    except ImportError as e:
        print(f"❌ Missing dependencies: {e}")
        print("Install with: pip install torch transformers")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Export failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
