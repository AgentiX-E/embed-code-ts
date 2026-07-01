#!/usr/bin/env python3
"""
Generate reference embeddings using HuggingFace model for comparison with TS engine.

Usage:
    python3 scripts/generate-reference.py --output reference-embeddings.json
"""
import argparse
import json
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel


def mean_pooling(hidden_states, attention_mask):
    """Mask-aware mean pooling."""
    mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
    sum_hidden = torch.sum(hidden_states * mask_expanded, dim=1)
    sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
    return sum_hidden / sum_mask


def l2_normalize(embeddings):
    """L2 normalize embeddings."""
    return torch.nn.functional.normalize(embeddings, p=2, dim=1)


def generate(model_id: str, output_path: str):
    print(f"Loading model: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_id, trust_remote_code=True)
    model.eval()

    # Test texts — same as test-fixtures.ts
    texts = [
        "search_query: How to sort an array using quicksort?",
        "search_document: def quicksort(arr): return arr if len(arr) <= 1 else quicksort([x for x in arr[1:] if x <= arr[0]]) + [arr[0]] + quicksort([x for x in arr[1:] if x > arr[0]])",
        "search_query: Recursive factorial implementation",
        "search_document: def factorial(n): return 1 if n <= 1 else n * factorial(n - 1)",
        "search_query: Database query for user order counts",
        "search_document: SELECT u.id, u.name, COUNT(o.id) FROM users u LEFT JOIN orders o ON u.id = o.user_id GROUP BY u.id, u.name",
        "search_query: TypeScript interface for cache entries",
        "search_document: interface CacheEntry<T> { readonly key: string; readonly value: T; readonly expiresAt: number; }",
        "search_query: Compute the n-th Fibonacci number recursively",
        "search_document: function fibonacci(n) { return n <= 1 ? n : fibonacci(n-1) + fibonacci(n-2); }",
    ]

    print(f"Generating embeddings for {len(texts)} texts...")

    # Batch 1: single text
    encoded = tokenizer(texts[:1], padding=True, truncation=True, max_length=64, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**encoded)
    embeddings_1 = mean_pooling(outputs[0], encoded["attention_mask"])
    embeddings_1 = l2_normalize(embeddings_1)

    # Batch 4: 4 texts
    encoded = tokenizer(texts[:4], padding=True, truncation=True, max_length=64, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**encoded)
    embeddings_4 = mean_pooling(outputs[0], encoded["attention_mask"])
    embeddings_4 = l2_normalize(embeddings_4)

    # Batch all: 10 texts
    encoded = tokenizer(texts, padding=True, truncation=True, max_length=64, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**encoded)
    embeddings_all = mean_pooling(outputs[0], encoded["attention_mask"])
    embeddings_all = l2_normalize(embeddings_all)

    # Normalize to unit norm
    results = {
        "model": model_id,
        "dim": model.config.hidden_size,
        "texts": texts,
        "batch1_embeddings": embeddings_1.cpu().numpy().tolist(),
        "batch1_shape": [1, model.config.hidden_size],
        "batch4_embeddings": embeddings_4.cpu().numpy().tolist(),
        "batch4_shape": [4, model.config.hidden_size],
        "batch10_embeddings": embeddings_all.cpu().numpy().tolist(),
        "batch10_shape": [10, model.config.hidden_size],
    }

    with open(output_path, "w") as f:
        json.dump(results, f)

    # Print similarity metrics for query-document pairs
    e = embeddings_all.cpu().numpy()
    print(f"\nReference embeddings saved to {output_path}")
    print(f"  Shape: {e.shape}")
    print(f"  Dim: {model.config.hidden_size}")

    # Query-document similarity checks
    for i in range(0, len(texts), 2):
        if i + 1 < len(texts):
            sim = np.dot(e[i], e[i + 1]) / (np.linalg.norm(e[i]) * np.linalg.norm(e[i + 1]))
            print(f"  sim({i},{i+1}): {sim:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", "-o", default="reference-embeddings.json")
    parser.add_argument("--model", "-m", default="nomic-ai/nomic-embed-text-v1.5")
    args = parser.parse_args()

    generate(args.model, args.output)
