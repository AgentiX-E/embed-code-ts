# @agentix-e/embed-code-cli

> **Command-line tool for nomic-embed-code** — download models, generate embeddings, and inspect ONNX models. Offline-first with proxy support.

[![npm](https://img.shields.io/npm/v/@agentix-e/embed-code-cli?color=blue)](https://www.npmjs.com/package/@agentix-e/embed-code-cli)
[![npm downloads](https://img.shields.io/npm/dm/@agentix-e/embed-code-cli?color=blue)](https://www.npmjs.com/package/@agentix-e/embed-code-cli)
[![API Docs](https://img.shields.io/badge/docs-TypeDoc-blue)](https://agentix-e.github.io/embed-code-ts/api/modules/embed-code-cli_src_cli.html)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/AgentiX-E/embed-code-ts/blob/main/LICENSE)

## Overview

`@agentix-e/embed-code-cli` provides the `embed-code` command-line tool for working with nomic-embed-code embeddings. It handles model downloads (with full proxy/auth support), generates text and file embeddings, and displays model information — all from the terminal.

## Installation

```bash
npm install -g @agentix-e/embed-code-cli

# Or use npx (no global install needed)
npx @agentix-e/embed-code-cli embed "your code here"
```

## Quick Start

```bash
# Step 1: Download the int8 ONNX model (~137 MB, one-time)
embed-code setup

# Step 2: Generate embeddings
embed-code embed "function add(a, b) { return a + b; }"

# Step 3: Embed from a file
embed-code embed -f src/index.ts

# Step 4: Inspect model
embed-code info
```

## Proxy Support

For corporate networks and air-gapped environments:

```bash
# Basic proxy
embed-code setup --proxy-url http://proxy.company.com:8080

# Authenticated proxy
embed-code setup \
  --proxy-url http://proxy:8080 \
  --proxy-username alice \
  --proxy-password s3cret

# Password via environment variable (recommended for CI)
EMBED_CODE_PROXY_PASSWORD=s3cret embed-code setup --proxy-url http://proxy:8080 --proxy-username alice

# Docker/Kubernetes secrets
EMBED_CODE_PROXY_PASSWORD_FILE=/run/secrets/proxy-password embed-code setup \
  --proxy-url http://proxy:8080 \
  --proxy-username alice
```

## Commands

### `setup` — Download model

Downloads and caches the ONNX model for offline inference.

```bash
embed-code setup                          # Default cache (~/.cache/embed-code-ts/)
embed-code setup --cache-dir ./models     # Custom cache directory
embed-code setup --force                  # Re-download even if cached
```

### `embed` — Generate embeddings

```bash
embed-code embed "function sort(arr) { return arr.sort(); }"
embed-code embed -f src/quicksort.ts
embed-code embed -m ./custom.onnx -f source.py --pooling mean
```

### `info` — Model information

```bash
embed-code info                           # Default model in cache
embed-code info -m ./models/model.onnx    # Custom model path
```

## Model Path Resolution

The `embed` command resolves the model in order:

1. `--model` / `-m` CLI flag
2. `EMBED_CODE_MODEL_PATH` environment variable
3. Default cache: `~/.cache/embed-code-ts/`
4. Auto-download on first use

## Use Cases

- **CI/CD pipelines**: Generate embeddings for code diff analysis
- **Pre-indexing**: Batch embed code repositories before loading into a vector DB
- **Ad-hoc queries**: Quick semantic searches from the terminal
- **Testing**: Verify embedding quality across model versions

## API Documentation

📚 **Full API reference**: [agentix-e.github.io/embed-code-ts/api/modules/embed-code-cli_src_cli.html](https://agentix-e.github.io/embed-code-ts/api/modules/embed-code-cli_src_cli.html)

## License

Apache 2.0
