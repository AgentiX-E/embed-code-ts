# @agentix-e/embed-code-core

Core inference engine for nomic-embed-code — int8 code embeddings for Node.js.

## Install

```bash
npm install @agentix-e/embed-code-core onnxruntime-node
```

## Usage

```typescript
import { EmbedCode, downloadModel } from '@agentix-e/embed-code-core';

const modelPath = await downloadModel();
const embedder = await EmbedCode.fromPretrained({ modelPath });
const result = await embedder.embed(['search_query: Calculate factorial']);
await embedder.dispose();
```

## API Documentation

Full API reference: [embed-code-ts API Docs](https://agentix-e.github.io/embed-code-ts/api/)

## License

Apache-2.0
