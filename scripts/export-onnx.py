#!/usr/bin/env python3
"""
Export nomic-embed-code from PyTorch (HuggingFace) to ONNX int8 format.

Uses HuggingFace Optimum for ONNX conversion with int8 quantization.

Usage:
    python3 scripts/export-onnx.py --output models/nomic-embed-code-v1-int8.onnx
    python3 scripts/export-onnx.py --output model.onnx --model nomic-ai/nomic-embed-code
    python3 scripts/export-onnx.py --output model.onnx --model nomic-ai/nomic-embed-text-v1.5

Requirements:
    pip install optimum onnx onnxruntime torch transformers

Environment:
    HF_TOKEN      — HuggingFace token for gated/private models
    HF_ENDPOINT   — HuggingFace mirror (e.g., https://hf-mirror.com)
"""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from datetime import datetime, timezone


def parse_args():
    parser = argparse.ArgumentParser(description="Export nomic-embed-code to ONNX int8")
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Output ONNX file path",
    )
    parser.add_argument(
        "--model", "-m",
        default="nomic-ai/nomic-embed-code",
        help="HuggingFace model ID (default: nomic-ai/nomic-embed-code)",
    )
    parser.add_argument(
        "--precision",
        default="int8",
        choices=["fp32", "fp16", "int8"],
        help="Export precision (default: int8)",
    )
    parser.add_argument(
        "--max-sequence-length",
        type=int,
        default=512,
        help="Max sequence length for the ONNX model (default: 512)",
    )
    parser.add_argument(
        "--check-latest",
        action="store_true",
        help="Check HuggingFace for the latest model version",
    )
    parser.add_argument(
        "--diff",
        action="store_true",
        help="Show diff between local descriptor and HF latest",
    )
    return parser.parse_args()


def compute_sha256(filepath: Path) -> str:
    """Compute SHA256 of a file (streaming for large files)."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(64 * 1024):
            h.update(chunk)
    return h.hexdigest()


def update_descriptor(output_path: Path, model_id: str, precision: str):
    """Update model-descriptor.json with the exported model's metadata."""
    descriptor_path = output_path.parent / "model-descriptor.json"

    if not descriptor_path.exists():
        print(f"  ⚠ model-descriptor.json not found at {descriptor_path}")
        print("    Creating minimal descriptor...")
        descriptor = {
            "schema": 1,
            "model": {
                "name": "nomic-embed-code",
                "version": "v1",
                "base_architecture": "Qwen2.5-7B",
                "hf_repository": model_id,
                "exported_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "precision": precision,
            },
            "onnx": {
                "input_ids_name": "input_ids",
                "attention_mask_name": "attention_mask",
                "output_name": "last_hidden_state",
                "input_shape": [1, 512],
                "output_shape": [1, 512, 3584],
                "opset": 20,
                "sha256": "",
                "size_bytes": 0,
            },
            "architecture": {},
            "tokenizer": {},
            "pooling": {"strategy": "last_token", "normalize": True},
            "task_prefixes": {
                "query": "search_query: ",
                "document": "search_document: ",
            },
        }
    else:
        with open(descriptor_path) as f:
            descriptor = json.load(f)

    # Update with export metadata
    file_size = output_path.stat().st_size
    sha256 = compute_sha256(output_path)

    descriptor["onnx"]["sha256"] = sha256
    descriptor["onnx"]["size_bytes"] = file_size
    descriptor["model"]["exported_at"] = datetime.now(timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    descriptor["model"]["precision"] = precision

    with open(descriptor_path, "w") as f:
        json.dump(descriptor, f, indent=2)
        f.write("\n")

    print(f"  ✓ Updated descriptor: {descriptor_path}")
    print(f"    SHA256: {sha256}")
    print(f"    Size:   {file_size:,} bytes ({file_size / 1024**2:.0f} MB)")


def check_latest(model_id: str, diff: bool = False):
    """Check HuggingFace for the latest model version."""
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        # Get latest commit info
        repo_info = api.repo_info(model_id, repo_type="model")
        latest_sha = repo_info.sha[:12] if repo_info.sha else "unknown"
        print(f"Latest {model_id}: {latest_sha}")

        if diff:
            descriptor_path = Path("models/model-descriptor.json")
            if descriptor_path.exists():
                with open(descriptor_path) as f:
                    desc = json.load(f)
                current = desc.get("model", {}).get("hf_revision", "unknown")
                print(f"  Current descriptor hf_revision: {current}")
                if current != latest_sha:
                    print(f"  ⚠ NEW VERSION available! Update with:")
                    print(f"    python3 scripts/export-onnx.py --output models/model.onnx")
            else:
                print("  No local descriptor found for comparison.")
    except ImportError:
        print("  huggingface_hub not installed. Install with: pip install huggingface_hub")
        sys.exit(1)


def main():
    args = parse_args()

    if args.check_latest:
        check_latest(args.model, args.diff)
        return

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n📦 Exporting {args.model} to ONNX ({args.precision})")
    print(f"   Output: {output_path}")
    print()

    try:
        import torch
        from transformers import AutoTokenizer, AutoModel
        from optimum.onnxruntime import ORTModelForFeatureExtraction
        from optimum.onnxruntime.configuration import AutoQuantizationConfig
    except ImportError as e:
        print(f"❌ Missing dependencies: {e}")
        print("Install with: pip install optimum onnx onnxruntime torch transformers")
        sys.exit(1)

    # Step 1: Export to ONNX
    print("  Step 1: Loading model from HuggingFace...")
    model = AutoModel.from_pretrained(
        args.model,
        torch_dtype=torch.float32,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=True,
    )

    print("  Step 2: Exporting to ONNX...")
    ort_model = ORTModelForFeatureExtraction.from_pretrained(
        args.model,
        export=True,
        trust_remote_code=True,
    )

    onnx_dir = output_path.parent / "onnx_export"
    ort_model.save_pretrained(onnx_dir)

    # Move the ONNX file to the target location
    onnx_file = list(onnx_dir.glob("*.onnx"))[0]
    onnx_file.rename(output_path)

    # Clean up temp dir
    import shutil
    shutil.rmtree(onnx_dir, ignore_errors=True)

    print(f"  ✓ ONNX model exported: {output_path} ({output_path.stat().st_size / 1024**2:.0f} MB)")

    # Step 2: Quantize (if int8 or fp16)
    if args.precision == "int8":
        print("  Step 3: Quantizing to int8...")
        from optimum.onnxruntime import ORTQuantizer
        from optimum.onnxruntime.configuration import AutoQuantizationConfig

        quantizer = ORTQuantizer.from_pretrained(args.model, file_name=output_path.name)
        dqconfig = AutoQuantizationConfig.avx512_vnni(is_static=False)

        quantized_dir = output_path.parent / "quantized"
        quantizer.quantize(
            save_dir=quantized_dir,
            quantization_config=dqconfig,
        )

        # Move quantized model to output
        quantized_file = list(quantized_dir.glob("*.onnx"))[0]
        output_path.unlink()
        quantized_file.rename(output_path)
        shutil.rmtree(quantized_dir, ignore_errors=True)

        print(f"  ✓ Int8 quantized: {output_path.stat().st_size / 1024**2:.0f} MB")

    # Step 3: Save tokenizer alongside model
    print("  Step 4: Saving tokenizer...")
    tokenizer.save_pretrained(output_path.parent)
    print(f"  ✓ Tokenizer saved to {output_path.parent}")

    # Step 4: Update descriptor with SHA256 and size
    print("  Step 5: Updating model descriptor...")
    update_descriptor(output_path, args.model, args.precision)

    print(f"\n✅ Export complete: {output_path}")
    print(f"   Size: {output_path.stat().st_size / 1024**2:.0f} MB")
    print(f"   SHA256: {compute_sha256(output_path)}")


if __name__ == "__main__":
    main()
