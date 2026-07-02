# embed-code-ts

> ONNX-powered code embeddings for Node.js and browser — int8 quantized, zero network dependency after install.

[![CI](https://github.com/AgentiX-E/embed-code-ts/actions/workflows/ci.yml/badge.svg)](https://github.com/AgentiX-E/embed-code-ts/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/docs-TypeDoc-blue)](https://agentix-e.github.io/embed-code-ts/api/)
[![Benchmark Report](https://img.shields.io/badge/benchmark-latest-blue)](https://agentix-e.github.io/embed-code-ts/benchmark/)
[![Coverage](https://img.shields.io/badge/coverage-report-blue)](https://agentix-e.github.io/embed-code-ts/coverage/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Node.js](https://img.shields.io/badge/Node.js-%3E%3D22-green)](https://nodejs.org/)

## Architecture

```
Text → [WordPiece Tokenizer] → [ONNX Runtime] → [Mean Pool] → [L2 Norm] → 768d Embedding
         (40856 vocab, 512 max)   (int8 quantized)     (mask-aware)
```

## Packages

| Package                      | Platform  | Runtime                                         | Size   |
| ---------------------------- | --------- | ----------------------------------------------- | ------ |
| `@agentix-e/embed-code-core` | Universal | Pure TypeScript (tokenizer, pooler, interfaces) | ~50KB  |
| `@agentix-e/embed-code-node` | Node.js   | `onnxruntime-node` (AVX2 native)                | ~137MB |
| `@agentix-e/embed-code-web`  | Browser   | `onnxruntime-web` (WASM/WebGPU)                 | ~137MB |
| `@agentix-e/embed-code-cli`  | CLI       | Model lifecycle management                      | ~50KB  |

## Quick Start

### Node.js

```typescript
import { NodeEmbedder } from '@agentix-e/embed-code-node';

const embedder = await NodeEmbedder.create({
  modelPath: './models/nomic-embed-code-v1.5.int8.onnx',
});

const embedding = await embedder.embed('function hello() { return "world"; }');
// → Float32Array(768), L2 normalized

await embedder.dispose();
```

### Browser

```typescript
import { WebEmbedder } from '@agentix-e/embed-code-web';

const embedder = await WebEmbedder.create('/models/model.onnx', tokenizerJson);
const embedding = await embedder.embed('const x = 42;');
```

### CLI

```bash
embed-code info                          # Model + system info
embed-code download nomic-ai/nomic-embed-code-v1.5  # Fetch from HF Hub
embed-code convert nomic-ai/nomic-embed-code-v1.5   # PyTorch → ONNX
embed-code quantize model.onnx           # float32 → int8
embed-code verify model.int8.onnx        # Accuracy benchmark
embed-code embed "your code here"        # Generate embedding
```

## Task Prefixes

For optimal retrieval performance, prefix inputs:

| Role            | Prefix                    |
| --------------- | ------------------------- |
| Query           | `search_query: {query}`   |
| Code / Document | `search_document: {code}` |

## Model

- **Architecture**: BERT-base, 12 layers, 768 hidden, 12 heads
- **Parameters**: 137M
- **Quantization**: int8 dynamic (~137MB)
- **Max tokens**: 512
- **Training**: [CoRNStack](https://arxiv.org/abs/2412.01007) dataset
- **Source**: [nomic-ai/nomic-embed-text-v1.5](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5)

## Performance (int8 ONNX)

| Backend            | Hardware | Latency |
| ------------------ | -------- | ------- |
| `onnxruntime-node` | AVX-512  | ~5ms    |
| `onnxruntime-node` | AVX2     | ~12ms   |
| `onnxruntime-web`  | WASM     | ~50ms   |
| `onnxruntime-web`  | WebGPU   | ~20ms   |

## Documentation & Reports

| Resource | Description | URL |
|---|---|---|
| 📚 **API Docs** | Full TypeDoc reference for all packages | [agentix-e.github.io/embed-code-ts/api/](https://agentix-e.github.io/embed-code-ts/api/) |
| 📊 **Benchmark** | Inference latency, throughput & accuracy reports | [agentix-e.github.io/embed-code-ts/benchmark/](https://agentix-e.github.io/embed-code-ts/benchmark/) |
| 📈 **Coverage** | Line, branch, function & statement coverage | [agentix-e.github.io/embed-code-ts/coverage/](https://agentix-e.github.io/embed-code-ts/coverage/) |

## License

Apache 2.0. Model weights from [nomic-ai/nomic-embed-text-v1.5](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5) (Apache 2.0).
