#!/usr/bin/env python3
"""
Export nomic-embed-code ONNX model with int8 dynamic quantization.

Pipeline: HuggingFace → float32 ONNX → int8 dynamic quantization → single-file ONNX

The int8 model is ~4× smaller and ~2-4× faster on CPU than float32,
with negligible accuracy loss (<1% cosine deviation) for embedding models.
"""
import argparse, sys, os, tempfile
from pathlib import Path
from transformers import AutoModel, AutoTokenizer
import torch


def main():
    parser = argparse.ArgumentParser(
        description="Export nomic-embed-code to int8-quantized ONNX"
    )
    parser.add_argument(
        "--model", "-m", default="nomic-ai/nomic-embed-text-v1.5",
        help="HuggingFace model ID"
    )
    parser.add_argument("--output", "-o", required=True, help="Output .onnx path")
    parser.add_argument(
        "--skip-quantize", action="store_true",
        help="Skip int8 quantization (produce float32 ONNX)"
    )
    args = parser.parse_args()

    # ── Step 1: Load model ──────────────────────────────────────
    print(f"[1/4] Loading model: {args.model}")
    model = AutoModel.from_pretrained(
        args.model, trust_remote_code=True, torch_dtype=torch.float32
    )
    model.eval()
    D = model.config.hidden_size
    L = model.config.num_hidden_layers
    print(f"      Architecture: BERT-base, {L} layers, dim={D}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    # ── Step 2: Export float32 to ONNX ──────────────────────────
    print("[2/4] Exporting float32 ONNX...")
    dummy = torch.randint(0, 1000, (1, 512))
    mask = torch.ones(1, 512, dtype=torch.int64)
    type_ids = torch.zeros(1, 512, dtype=torch.int64)

    # Use a temp file for float32 export, then quantize into output
    float32_path = args.output if args.skip_quantize else tempfile.mktemp(suffix=".onnx")

    torch.onnx.export(
        model,
        (dummy, mask, type_ids),
        float32_path,
        input_names=["input_ids", "attention_mask", "token_type_ids"],
        output_names=["last_hidden_state"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
            "attention_mask": {0: "batch", 1: "sequence"},
            "token_type_ids": {0: "batch", 1: "sequence"},
            "last_hidden_state": {0: "batch", 1: "sequence"},
        },
        opset_version=18,
    )

    float32_mb = os.path.getsize(float32_path) / 1024 / 1024
    print(f"      Float32 ONNX: {float32_mb:.1f} MB")

    # ── Step 3: Int8 dynamic quantization ───────────────────────
    if not args.skip_quantize:
        print("[3/4] Quantizing to int8 (dynamic, per-channel)...")
        from onnxruntime.quantization import quantize_dynamic, QuantType

        quantize_dynamic(
            model_input=float32_path,
            model_output=args.output,
            weight_type=QuantType.QInt8,
            per_channel=True,
            reduce_range=False,
            extra_options={"ActivationSymmetric": True},
        )

        # Clean up temp float32 file
        try:
            os.unlink(float32_path)
        except OSError:
            pass

        quant_mb = os.path.getsize(args.output) / 1024 / 1024
        print(f"      Int8 ONNX: {quant_mb:.1f} MB ({quant_mb / float32_mb * 100:.0f}% of float32)")
    else:
        print("[3/4] Skipped (--skip-quantize)")

    # ── Step 4: Merge to single file ────────────────────────────
    print("[4/4] Merging to single-file ONNX...")
    import onnx

    m = onnx.load(args.output, load_external_data=True)
    onnx.save(m, args.output, save_as_external_data=False)
    final_mb = os.path.getsize(args.output) / 1024 / 1024
    print(f"      Single-file ONNX: {final_mb:.1f} MB")

    # ── Summary ─────────────────────────────────────────────────
    print()
    print(f"✅ Done: {args.output}")
    print(f"   Architecture: BERT-base, {L} layers, dim={D}")
    print(f"   Precision:    {'int8' if not args.skip_quantize else 'float32'}")
    print(f"   Size:         {final_mb:.1f} MB")
    print(f"   Opset:        18")


if __name__ == "__main__":
    main()
