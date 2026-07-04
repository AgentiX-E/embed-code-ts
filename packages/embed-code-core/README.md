# @agentix-e/embed-code-core

> Zero-dependency core engine for @agentix-e/embed-code-ts — provides WordPiece tokenizer, pooling, normalization, and platform-agnostic interfaces for ONNX Runtime-based code embeddings.

[![npm](https://img.shields.io/npm/v/@agentix-e/embed-code-core?color=blue)](https://www.npmjs.com/package/@agentix-e/embed-code-core)
[![API Docs](https://img.shields.io/badge/docs-TypeDoc-blue)](https://agentix-e.github.io/embed-code-ts/api/modules/embed-code-core_src.html)

## Overview

`@agentix-e/embed-code-core` is the platform-agnostic core of embed-code-ts. It defines the interfaces and utilities shared by all platform packages — tokenization, pooling, normalization, batch processing, and error types — without depending on any ONNX Runtime backend.

## Installation

```bash
npm install @agentix-e/embed-code-core
```

Requires Node.js ≥ 22.

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

// Tokenize input text
const tokens = tokenizer.encode('search_query: Calculate the n-th factorial');
console.log(tokens.inputIds); // Int32Array — token IDs
console.log(tokens.attentionMask); // Int32Array — attention mask

// After running ONNX inference to get token embeddings (Float32Array of shape [seq_len, 768])…
// Apply mean pooling and L2 normalization
const pooled = meanPool(tokenEmbeddings, tokens.attentionMask);
const normalized = l2Normalize(pooled);
// normalized is a Float32Array of length 768

// Compare two embeddings with cosine similarity
const similarity = cosineSimilarity(embeddingA, embeddingB);
```

## Key Exports

| Category             | Exports                                                                                                                                                                      |
| -------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Types**            | `IEmbedder`, `BatchOptions`, `ModelInfo`, `TokenizedInput`, `EmbeddingResult`, `EmbedProgress`, `DownloadOptions`, `ProxyConfig`, `IOrtBackend`, `IOrtSession`, `IOrtTensor` |
| **Tokenizer**        | `WordPieceTokenizer`, `loadVocab`, `preTokenize`, `isPunctuation`                                                                                                            |
| **Post-processing**  | `meanPool`, `l2Normalize`, `cosineSimilarity`, `clsPool`, `lastTokenPool`                                                                                                    |
| **Batch processing** | `processBatch`                                                                                                                                                               |
| **Errors**           | `EmbedCodeError`, `ModelNotFoundError`, `DownloadError`, `ChecksumMismatchError`, `ProxyAuthError`, `InferenceError`, `TokenizationError`, `ModelNotCompiledError`           |

## API Documentation

Full API reference: [agentix-e.github.io/embed-code-ts/api/modules/embed-code-core_src.html](https://agentix-e.github.io/embed-code-ts/api/modules/embed-code-core_src.html)

## License

Apache 2.0
