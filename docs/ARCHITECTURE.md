# Architecture

> `embed-code-ts` — A TypeScript/Node.js implementation of Nomic's nomic-embed-code model with int8 incbin-style model delivery.

---

## High-Level Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                   @agentix-e/embed-code-cli                     │
│  Commander-based CLI: setup (model download) + embed          │
└──────────────────────────┬───────────────────────────────────┘
                           │ uses
┌──────────────────────────▼───────────────────────────────────┐
│                 @agentix-e/embed-code-core                      │
│  ┌──────────┐  ┌──────────┐  ┌───────────┐  ┌────────────┐  │
│  │  Model   │  │  Config  │  │ Tokenizer │  │  Pooler    │  │
│  │  (API)   │  │ (types)  │  │  (BPE)    │  │ (strategies)│  │
│  └────┬─────┘  └──────────┘  └─────┬─────┘  └─────┬──────┘  │
│       │                            │               │        │
│       │    ┌───────────────────────┴───────────────┘        │
│       │    │                                                 │
│       ▼    ▼                                                 │
│  ┌──────────────────────────────────────────┐              │
│  │         Embedding Pipeline                │              │
│  │  Step 1: Tokenize (BPE)                   │              │
│  │  Step 2: 
│  │  Step 3: Pool (last_token/mean/cls)       │              │
│  │  Step 4: Normalize (L2)                   │              │
│  └────────────────┬─────────────────────────┘              │
│                   │ uses                                     │
│  ┌────────────────▼─────────────────────────┐              │
│  │    EmbedCodeInferenceEngine               │              │
│  │  IInferenceEngine → pure-TypeScript backend   │              │
│  │  CPU / CUDA / DirectML                     │              │
│  └──────────────────────────────────────────┘              │
│                                                              │
│  Utilities: Model Downloader, Descriptor Resolver            │
│  Descriptor: model-descriptor.json (architecture contract)   │
└──────────────────────────────────────────────────────────────┘
```

---

## Component Design

### 1. EmbedCode (Public API)

**File**: `packages/embed-code-core/src/embed-code.ts`

The main entry point. Implements the public API via a private constructor + static factory.

```typescript
// Lifecycle
embedder = EmbedCode.fromPretrained({ modelPath, executionProvider });
results = embedder.embed(texts, { maxTokens, poolingStrategy, normalize });
sim = embedder.similarity(embA, embB);
embedder.dispose();
```

**Design decisions**:

- **Private constructor + static factory**: Ensures async initialization is enforced
- **Task prefix API**: `embedder.taskPrefixes` exposes `{ query, document }` for proper prompt formatting
- **Flat Float32Array output**: Optimized for zero-copy bulk similarity computation

### 2. Tokenizer Pipeline

**File**: `packages/embed-code-core/src/tokenizer.ts`

Transforms raw text into model-ready token IDs:

```
Raw Text → [Normalize] → [BPE Encode] → [Pad/Truncate] → [Attention Mask] → [Token IDs]
              │                │              │                   │
           Unicode NFC    Vocabulary       maxTokens          1=real, 0=pad
           + lowercase     lookup
```

### 3. Embedding Pipeline

```
Input Text(s)
    │
    ├─→ [tokenize()]
    │       ├─ Text normalization (Unicode NFC)
    │       ├─ BPE encode → token IDs
    │       ├─ Truncate/pad → maxTokens
    │       └─ Generate attention mask
    │
    ├─→ [infer()]
    │       ├─ Build build input tensors
    │       ├─ engine.forward() → hidden states
    │       └─ Extract last_hidden_state [B, L, 3584]
    │
    └─→ [pool + normalize()]
            ├─ Pool (last_token / mean / cls)
            ├─ L2 normalize → unit vectors
            └─ Return Float32Array(N × 3584)
```

### 4. TS Inference Engine

**File**: `packages/embed-code-core/src/inference/ts-engine.ts`

Implements `IInferenceEngine`:

- **Pluggable execution providers**: CPU / CUDA / DirectML
- **Dynamic shape handling**: Variable-length inputs auto-padded to model maxTokens
- **Concurrent batch inference**: `Promise.all` for parallel TS session calls
- **Proper resource cleanup**: Calls `session.release()` on dispose

### 5. Pooling Strategies

**File**: `packages/embed-code-core/src/pooling.ts`

| Strategy       | Description                                     | Use Case                        |
| -------------- | ----------------------------------------------- | ------------------------------- |
| **last_token** | Use the last non-padding token embedding        | Default, recommended for search |
| **Mean**       | Average of all token embeddings (ignoring pads) | Best for code classification    |
| **CLS**        | Use the [CLS] token embedding only              | BERT-style tasks                |

### 6. Model Downloader

**File**: `packages/embed-code-core/src/model-downloader.ts`

- **Streaming download**: Uses Node.js `fetch` reader → file `writeStream` (no large heap buffer for multi-GB models)
- **Proxy support**: Environment variables (`EMBED_CODE_PROXY_URL/USERNAME/PASSWORD`, `HTTPS_PROXY`) or programmatic `DownloadOptions.proxy` with username/password. Uses undici `ProxyAgent` for clean proxy handling without global environment mutations
- **SHA-256 integrity**: Hashes the TS file after download for verification
- **Cache management**: Platform-aware cache directory (`XDG_CACHE_HOME`)
- **Error hierarchy**: `ProxyAuthError` (HTTP 407), `DownloadError`, `ChecksumMismatchError`

### 7. Model Descriptor System

**File**: `packages/embed-code-core/src/model-descriptor.ts`

The `model-descriptor.json` file (committed to repo, distributed with releases) defines the architecture contract:

```
model-descriptor.json → resolveModelConfig() → ModelConfig
    │
    └─ Single source of truth for:
       ├─ model.hf_revision    (HuggingFace checkpoint)
       ├─ weights.sha256          (download integrity)
       ├─ architecture.*       (dim, maxTokens, vocabSize)
       └─ processing.*         (tokenizer config, pooling defaults)
```

---

## Data Flow

```
User Input (string | string[])
    │
    ▼
[EmbedCode.embed()]
    │
    ├─→ [tokenize()]
    │       ├─ Normalize: Unicode NFC + lowercase
    │       ├─ BPE encode → token IDs
    │       ├─ Pad/truncate → maxTokens (32768)
    │       └─ Attention mask generation
    │
    ├─→ [inference()]
    │       ├─ Build BigInt64Array input_ids [B, L]
    │       ├─ Build BigInt64Array attention_mask [B, L]
    │       ├─ engine.forward({ input_ids, attention_mask })
    │       └─ Extract last_hidden_state [B, L, 3584]
    │
    └─→ [pool + normalize()]
            ├─ Pool: select strategy (last_token/mean/cls)
            ├─ L2 normalize: x / ||x||₂
            └─ Return { embeddings: Float32Array(N×3584), elapsedMs }
```

---

## Type System

```
EmbedOptions         — mutable options (passed to embed())
ModelConfig          — frozen, read-only (resolved from descriptor)
EmbeddingResult      — { embeddings: Float32Array, shape: [number, number], elapsedMs: number }
EmbedProgress        — { phase: 'tokenize' | 'inference' | 'pool' | 'normalize', step: number, total: number }
IInferenceEngine     — pluggable backend (TS)
```

---

## Key Design Principles

1. **Incbin-inspired delivery**: npm package is code-only (~50 KB); model is downloaded on-demand and cached
2. **Functional core, imperative shell**: All utility functions are pure; only `EmbedCode` manages state
3. **Interface-based abstraction**: `IInferenceEngine` decouples the model from pure-TypeScript
4. **Self-describing models**: `model-descriptor.json` is the single source of truth for architecture constants — `fromPretrained()` resolves `ModelConfig` via `resolveModelConfig()` from the descriptor
5. **Zero-dependency core for basic usage**: Only `onnxruntime-node` (dynamic import) at inference time
6. **Python parity**: Every source file cites the corresponding HuggingFace/Python source for cross-verification
7. **Progressive disclosure**: Public API exports both high-level (`EmbedCode`) and advanced (`tokenize`, `pool`, `normalize`) APIs

---

## Package Sizes

| Package                      | Code Size | Dependencies                 |
| ---------------------------- | --------- | ---------------------------- |
| `@agentix-e/embed-code-core` | ~50 KB    | embedded weights |
| `@agentix-e/embed-code-cli`  | ~10 KB    | `commander`                  |

Model weights (~137 MB int8 TS for nomic-embed-code 7B) are downloaded separately from GitHub Releases.
