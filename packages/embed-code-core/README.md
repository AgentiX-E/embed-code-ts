# @agentix-e/embed-code-core

> **Zero-dependency TypeScript core for code embeddings** — WordPiece tokenizer, pooling, normalization, and platform-agnostic interfaces for ONNX Runtime-based code embeddings.

[![npm](https://img.shields.io/npm/v/@agentix-e/embed-code-core?color=blue)](https://www.npmjs.com/package/@agentix-e/embed-code-core)
[![npm downloads](https://img.shields.io/npm/dm/@agentix-e/embed-code-core?color=blue)](https://www.npmjs.com/package/@agentix-e/embed-code-core)
[![API Docs](https://img.shields.io/badge/docs-TypeDoc-blue)](https://agentix-e.github.io/embed-code-ts/api/modules/embed-code-core_src.html)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/AgentiX-E/embed-code-ts/blob/main/LICENSE)

## Overview

`@agentix-e/embed-code-core` is the **platform-agnostic core engine** of embed-code-ts. It provides everything you need for code text tokenization and post-processing — without any ONNX Runtime dependency. This makes it ideal for environments where you want to pre-process text on the client and run inference on the server, or vice versa.

**Key capabilities**: WordPiece tokenization (BERT-compatible), attention-mask-aware mean pooling, L2 normalization, cosine similarity, batch processing, and comprehensive error types.

## Installation

```bash
npm install @agentix-e/embed-code-core
```

Requires Node.js ≥ 22. Zero external dependencies.

## Quick Start

```typescript
import {
  WordPieceTokenizer,
  loadVocab,
  preTokenize,
  meanPool,
  l2Normalize,
  cosineSimilarity,
} from '@agentix-e/embed-code-core';

// Load the WordPiece vocabulary
const vocab = await loadVocab('./vocab.txt');

// Create a tokenizer
const tokenizer = new WordPieceTokenizer(vocab);

// Tokenize with search intent prefixes
const tokens = tokenizer.encode('search_query: Calculate the n-th factorial');
console.log(tokens.inputIds); // Int32Array — token IDs
console.log(tokens.attentionMask); // Int32Array — attention mask

// After running ONNX inference to get token embeddings...
const pooled = meanPool(tokenEmbeddings, tokens.attentionMask);
const normalized = l2Normalize(pooled);
// normalized → Float32Array(768), ready for vector search

// Compare embeddings
const similarity = cosineSimilarity(embeddingA, embeddingB);
```

## Key Exports

| Category | Exports |
| --- | --- |
| **Types** | `IEmbedder`, `BatchOptions`, `ModelInfo`, `TokenizedInput`, `EmbeddingResult`, `EmbedProgress`, `DownloadOptions`, `ProxyConfig`, `IOrtBackend`, `IOrtSession`, `IOrtTensor` |
| **Tokenizer** | `WordPieceTokenizer`, `loadVocab`, `preTokenize`, `isPunctuation` |
| **Post-processing** | `meanPool`, `l2Normalize`, `cosineSimilarity`, `clsPool`, `lastTokenPool` |
| **Batch** | `processBatch` |
| **Errors** | `EmbedCodeError`, `ModelNotFoundError`, `DownloadError`, `ChecksumMismatchError`, `ProxyAuthError`, `InferenceError`, `TokenizationError`, `ModelNotCompiledError` |

## Use Cases

- **Semantic code search**: Tokenize code and queries, run inference, compare with cosine similarity
- **RAG pipelines**: Pre-process documents for retrieval-augmented generation
- **Code deduplication**: Find similar code snippets across repositories
- **Custom embeddings**: Build your own embedding pipeline with your ONNX model

## API Documentation

📚 **Full API reference**: [agentix-e.github.io/embed-code-ts/api/modules/embed-code-core_src.html](https://agentix-e.github.io/embed-code-ts/api/modules/embed-code-core_src.html)

## License

Apache 2.0
