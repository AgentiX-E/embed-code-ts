# Getting Started with embed-code-ts

## Installation

```bash
npm install @agentix-e/embed-code-core onnxruntime-node
```

Optional: CLI for command-line usage.

```bash
npm install -g @agentix-e/embed-code-cli
```

## 1. Download the Model

First, download the int8-quantized ONNX model. This is a one-time operation — the model is cached for future runs.

```typescript
import { downloadModel } from '@agentix-e/embed-code-core';

const modelPath = await downloadModel({
  onProgress: (received, total, speed) => {
    console.log(`${received.toFixed(0)} / ${total.toFixed(0)} MB`);
  },
});
// → ~/.cache/agentix-embed-code-ts/nomic-embed-code-v1-int8.onnx
```

Or via CLI:

```bash
embed-code setup
```

## 2. Create an Embedder

```typescript
import { EmbedCode } from '@agentix-e/embed-code-core';

const embedder = await EmbedCode.fromPretrained({
  modelPath: '/path/to/model.onnx',
});
```

## 3. Generate Embeddings

nomic-embed-code uses task prefixes. The API provides helpers:

```typescript
const { query, document } = embedder.taskPrefixes;
// query → "search_query: "
// document → "search_document: "

const results = await embedder.embed([
  query + 'Calculate the nth factorial',
  document + 'def fact(n): return 1 if n <= 1 else n * fact(n-1)',
]);

console.log(results.embeddings); // Float32Array [2, 3584]
console.log(results.elapsedMs);  // Inference time in ms
```

### CLI Usage

```bash
# Query-code similarity
embed-code embed --query "Calculate factorial" --doc "def fact(n): return 1 if n <= 1 else n * fact(n-1)"

# Embed a code file
embed-code embed --file src/main.py

# Pipe from stdin
cat code.py | embed-code embed

# JSON output
embed-code embed --format json --query "Sort array"
```

## 4. Compute Similarity

```typescript
const sim = embedder.similarity(results.embeddings.slice(0, 3584), results.embeddings.slice(3584));
console.log(sim); // Cosine similarity between query and code
```

## 5. Clean Up

```typescript
await embedder.dispose();
```

## Proxy Configuration

For corporate networks behind a firewall:

```bash
export EMBED_CODE_PROXY_URL=http://proxy.company.com:8080
export EMBED_CODE_PROXY_USERNAME=user
export EMBED_CODE_PROXY_PASSWORD=pass

embed-code setup
```

Or programmatically:

```typescript
await downloadModel({
  proxy: {
    url: 'http://proxy.company.com:8080',
    username: 'user',
    password: 'pass',
  },
});
```

## Environment Variables

| Variable | Purpose |
|----------|---------|
| `EMBED_CODE_PROXY_URL` | Proxy for model download |
| `EMBED_CODE_PROXY_USERNAME` | Proxy username |
| `EMBED_CODE_PROXY_PASSWORD` | Proxy password |
| `EMBED_CODE_MODEL_PATH` | Override model path |
| `XDG_CACHE_HOME` | Override cache directory root |

## Exporting Models

To export nomic-embed-code from PyTorch to ONNX:

```bash
pip install optimum onnx onnxruntime torch transformers

python3 scripts/export-onnx.py \
  --output models/nomic-embed-code-v1-int8.onnx \
  --model nomic-ai/nomic-embed-code \
  --precision int8
```
