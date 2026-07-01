#!/usr/bin/env python3
"""
Generate reference embeddings using our exact pre-norm architecture.

This replicates the TS engine's forward pass exactly:
  - Word embeddings + token_type_embeddings
  - Embedding LayerNorm
  - 12× pre-norm transformer layers (attention + FFN)
  - Mean pooling + L2 normalize

Reads exported int8 weights and tokenizer.json, runs the same
computation in NumPy, producing reference values for validation.
"""
import argparse
import json
import struct
from pathlib import Path
import numpy as np

# ── Read binary weights ──────────────────────────────────────────

class WeightReader:
    def __init__(self, path: Path):
        data = path.read_bytes()
        self._buf = data
        self.arch = {}
        self.tensors = {}
        self._parse(data)

    def _parse(self, data):
        # Header
        magic = data[:8]
        assert magic == b"EMBCODE1", f"Bad magic: {magic}"
        off = 8
        version = struct.unpack_from("<I", data, off)[0]; off += 4
        self.arch["num_layers"] = struct.unpack_from("<H", data, off)[0]; off += 2
        self.arch["hidden_size"] = struct.unpack_from("<H", data, off)[0]; off += 2
        self.arch["intermediate_size"] = struct.unpack_from("<I", data, off)[0]; off += 4
        self.arch["vocab_size"] = struct.unpack_from("<I", data, off)[0]; off += 4
        self.arch["max_positions"] = struct.unpack_from("<I", data, off)[0]; off += 4
        self.arch["num_heads"] = struct.unpack_from("<H", data, off)[0]; off += 2
        self.arch["head_dim"] = struct.unpack_from("<H", data, off)[0]

        # Offset table
        total_size = struct.unpack_from("<Q", data, 32)[0]
        off = 256
        while off < total_size:
            # Entry names are printable ASCII (a-z, 0-9, ., _). If the first byte
            # is not in this range, we've hit the payload.
            if not (65 <= data[off] <= 122) and data[off] != ord('_'): break
            name_len = 0
            while name_len < 64 and data[off + name_len] != 0:
                name_len += 1
            name = data[off:off + name_len].decode("ascii")
            if not name: break
            off += 64
            tensor_off = struct.unpack_from("<Q", data, off)[0]; off += 8
            sz = struct.unpack_from("<Q", data, off)[0]; off += 8
            rank = struct.unpack_from("<H", data, off)[0]; off += 2
            dims = [struct.unpack_from("<H", data, off + i * 2)[0] for i in range(4)]
            off += 8
            scale_off = struct.unpack_from("<Q", data, off)[0]; off += 8
            off += 16  # reserved
            actual_dims = dims[:rank] if rank < 4 else dims[:4]
            self.tensors[name] = {
                "offset": tensor_off,
                "size": sz,
                "dims": actual_dims,
                "scale_offset": scale_off,
            }

    def get_int8(self, name):
        t = self.tensors[name]
        return np.frombuffer(self._buf, dtype=np.int8, count=t["size"], offset=t["offset"])

    def get_f32(self, name):
        t = self.tensors[name]
        n = t["dims"][0] if t["dims"] else 1
        return np.frombuffer(self._buf, dtype=np.float32, count=n, offset=t["scale_offset"])

    def linear(self, name, x):
        """Apply int8 linear: x @ dequantize(W_int8^T) + bias."""
        meta = self.tensors[name + ".weight"]
        dims = meta["dims"]  # [out_features, in_features]
        out_features, in_features = dims[0], dims[1]
        w = self.get_int8(name + ".weight").astype(np.float32).reshape(out_features, in_features)
        s = self.get_f32(name + ".weight").astype(np.float32)
        w_dequant = w * s[:, np.newaxis]  # [out_features, in_features]
        b = np.zeros(out_features, dtype=np.float32)
        try:
            b_i8 = self.get_int8(name + ".bias")
            b_s = self.get_f32(name + ".bias")
            b = (b_i8 * b_s).astype(np.float32)
        except: pass
        return x @ w_dequant.T + b


def layer_norm(x, w_name, b_name):
    """Apply LayerNorm with int8 weights."""
    w_i8 = wr.get_int8(w_name).astype(np.float32)
    w_s = wr.get_f32(w_name)[0]
    gamma = w_i8 * w_s
    b_i8 = wr.get_int8(b_name).astype(np.float32)
    b_s = wr.get_f32(b_name)[0]
    beta = b_i8 * b_s
    mean = x.mean(axis=-1, keepdims=True)
    var = ((x - mean) ** 2).mean(axis=-1, keepdims=True)
    x_norm = (x - mean) / np.sqrt(var + 1e-12)
    return x_norm * gamma + beta


def attention(qkv_weight_prefix, out_prefix, x, mask):
    B, L, D = x.shape
    H, d = wr.arch["num_heads"], wr.arch["head_dim"]
    
    # Q, K, V projections (no bias, using zero)
    Q = wr.linear(qkv_weight_prefix + ".attention.self.query", x)
    K = wr.linear(qkv_weight_prefix + ".attention.self.key", x)
    V = wr.linear(qkv_weight_prefix + ".attention.self.value", x)
    
    # Reshape to [B, H, L, d]
    Q = Q.reshape(B, L, H, d).transpose(0, 2, 1, 3)
    K = K.reshape(B, L, H, d).transpose(0, 2, 1, 3)
    V = V.reshape(B, L, H, d).transpose(0, 2, 1, 3)
    
    # Scores
    scores = Q @ K.transpose(0, 1, 3, 2) / np.sqrt(d)  # [B, H, L, L]
    
    # Mask
    mask_exp = np.where(mask[:, np.newaxis, np.newaxis, :] == 0, -1e9, 0.0)  # [B, 1, 1, L]
    scores = scores + mask_exp
    
    # Softmax
    scores = scores - scores.max(axis=-1, keepdims=True)
    scores = np.exp(scores)
    scores = scores / scores.sum(axis=-1, keepdims=True)
    
    # Context
    ctx = scores @ V  # [B, H, L, d]
    ctx = ctx.transpose(0, 2, 1, 3).reshape(B, L, D)
    
    # Output projection
    return wr.linear(out_prefix + ".attention.output.dense", ctx)


def transformer_layer(layer_num, x, mask):
    p = f"encoder.layer.{layer_num}"
    B, L, D = x.shape
    
    # Pre-norm + Attention
    saved = x.copy()
    x_norm = layer_norm(x, f"{p}.attention.output.LayerNorm.weight", f"{p}.attention.output.LayerNorm.bias")
    attn_out = attention(p, p, x_norm, mask)
    x = saved + attn_out
    
    # Pre-norm + FFN
    saved = x.copy()
    x_norm = layer_norm(x, f"{p}.output.LayerNorm.weight", f"{p}.output.LayerNorm.bias")
    
    intermediate = wr.linear(f"{p}.intermediate.dense", x_norm)
    # GELU
    intermediate = 0.5 * intermediate * (1 + np.tanh(0.7978845608028654 * (intermediate + 0.044715 * intermediate ** 3)))
    
    ff_out = wr.linear(f"{p}.output.dense", intermediate)
    return saved + ff_out


def embed(input_ids, mask):
    B, L = input_ids.shape
    D = wr.arch["hidden_size"]
    
    # Word embeddings
    w = wr.get_int8("embeddings.word_embeddings.weight").astype(np.float32).reshape(wr.arch["vocab_size"], D)
    s = wr.get_f32("embeddings.word_embeddings.weight")
    w_dequant = w * s[:, np.newaxis]  # [V, D]
    x = w_dequant[input_ids]  # [B, L, D]
    
    # Token type embeddings
    try:
        tw = wr.get_int8("embeddings.token_type_embeddings.weight").astype(np.float32).reshape(2, D)
        ts = wr.get_f32("embeddings.token_type_embeddings.weight")
        tw_dequant = tw * ts[:, np.newaxis]
        x = x + tw_dequant[np.zeros((B, L), dtype=np.int32)]
    except: pass
    
    # Embedding LayerNorm
    x = layer_norm(x.reshape(B * L, D), "embeddings.LayerNorm.weight", "embeddings.LayerNorm.bias").reshape(B, L, D)
    return x


# ── Main ──

def main():
    global wr
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="models/nomic-embed-code-v1-int8.weights.bin")
    parser.add_argument("--tokenizer", default="models/tokenizer.json")
    parser.add_argument("--output", default="models/reference-pre-norm.json")
    args = parser.parse_args()

    print(f"Loading weights: {args.weights}")
    wr = WeightReader(Path(args.weights))
    print(f"  Architecture: {wr.arch}")

    # Load tokenizer
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)
    D = wr.arch["hidden_size"]
    
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

    # Tokenize
    encoded = tok(texts, padding=True, truncation=True, max_length=64, return_tensors="np")
    input_ids = encoded["input_ids"].astype(np.int32)
    mask = encoded["attention_mask"].astype(np.int32)
    B, L = input_ids.shape

    print(f"Input: {B} texts × {L} tokens")

    # Forward pass
    x = embed(input_ids, mask)

    for layer in range(wr.arch["num_layers"]):
        x = transformer_layer(layer, x, mask)

    # Mean pooling
    mask_expanded = mask[:, :, np.newaxis].astype(np.float32)
    pooled = (x * mask_expanded).sum(axis=1) / np.maximum(mask_expanded.sum(axis=1), 1e-9)

    # L2 normalize
    norms = np.linalg.norm(pooled, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    pooled = pooled / norms

    embeddings = pooled.tolist()
    print(f"Output: {len(embeddings)} × {D}")

    # Query-document similarities
    for i in range(0, len(texts), 2):
        if i + 1 < len(texts):
            sim = np.dot(pooled[i], pooled[i + 1])
            print(f"  sim({i},{i+1}): {sim:.6f}")

    with open(args.output, "w") as f:
        json.dump({
            "arch": wr.arch,
            "texts": texts,
            "dim": D,
            "embeddings": embeddings,
            "shape": [len(texts), D],
        }, f)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
