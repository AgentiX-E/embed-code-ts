#!/usr/bin/env python3
"""Export nomic-embed-code ONNX model for CI and local use."""
import argparse, sys
from pathlib import Path
from transformers import AutoModel, AutoTokenizer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", default="nomic-ai/nomic-embed-text-v1.5")
    parser.add_argument("--output", "-o", required=True)
    args = parser.parse_args()
    
    import torch
    model = AutoModel.from_pretrained(args.model, trust_remote_code=True, torch_dtype=torch.float32)
    model.eval()
    
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    
    # Export to ONNX
    D = model.config.hidden_size
    dummy = torch.randint(0, 1000, (1, 512))
    mask = torch.ones(1, 512, dtype=torch.int64)
    type_ids = torch.zeros(1, 512, dtype=torch.int64)
    
    torch.onnx.export(
        model,
        (dummy, mask, type_ids),
        args.output,
        input_names=['input_ids', 'attention_mask', 'token_type_ids'],
        output_names=['last_hidden_state'],
        dynamic_axes={
            'input_ids': {0: 'batch', 1: 'sequence'},
            'attention_mask': {0: 'batch', 1: 'sequence'},
            'token_type_ids': {0: 'batch', 1: 'sequence'},
            'last_hidden_state': {0: 'batch', 1: 'sequence'},
        },
        opset_version=14,
    )
    
    import os
    size_mb = os.path.getsize(args.output) / 1024 / 1024
    print(f"Exported: {args.output} ({size_mb:.1f} MB)")
    print(f"  Dim: {D}, Layers: {model.config.num_hidden_layers}")

if __name__ == "__main__":
    main()
