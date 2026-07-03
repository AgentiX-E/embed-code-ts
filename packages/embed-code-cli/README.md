# @agentix-e/embed-code-cli

> Command-line interface for nomic-embed-code — download models and generate code embeddings.

[![npm](https://img.shields.io/npm/v/@agentix-e/embed-code-cli?color=blue)](https://www.npmjs.com/package/@agentix-e/embed-code-cli)
[![API Docs](https://img.shields.io/badge/docs-TypeDoc-blue)](https://agentix-e.github.io/embed-code-ts/api/modules/embed-code-cli_src_cli.html)

## Overview

`@agentix-e/embed-code-cli` provides the `embed-code` command-line tool for code-aware text embeddings. It handles model downloads, text/file embedding generation, and command-line argument parsing via Commander.

## Installation

```bash
npm install -g @agentix-e/embed-code-cli
# or use npx
npx @agentix-e/embed-code-cli embed "your code here"
```

## Quick Start

```bash
# Download model (first time only, ~7 GB)
embed-code setup

# With proxy (corporate network)
embed-code setup --proxy-url http://proxy.company.com:8080
embed-code setup --proxy-url http://proxy:8080 --proxy-username user --proxy-password pass

# Password via environment variable (recommended for security)
EMBED_CODE_PROXY_PASSWORD=pass embed-code setup --proxy-url http://proxy:8080 --proxy-username user

# Or via file (Docker/Kubernetes secrets)
EMBED_CODE_PROXY_PASSWORD_FILE=/run/secrets/proxy-password embed-code setup --proxy-url http://proxy:8080 --proxy-username user

# Generate embedding
embed-code embed "function add(a, b) { return a + b; }"

# Embed from a file
embed-code embed -f source.py

# Custom model path and options
embed-code embed -m ./custom.weights.bin -f source.ts --pooling mean

# Show model info
embed-code info -m ./models/nomic-embed-text-v1.5-int8.weights.bin
```

## Model Path Resolution

The `embed` command resolves the model in this order:

1. `--model` CLI flag
2. `EMBED_CODE_MODEL_PATH` environment variable
3. Default cache (`~/.cache/embed-code-ts/`)
4. Auto-download

## API Documentation

📚 **Full API reference**: [agentix-e.github.io/embed-code-ts/api/modules/embed-code-cli_src_cli.html](https://agentix-e.github.io/embed-code-ts/api/modules/embed-code-cli_src_cli.html)

Key exports:

- CLI entry point (Commander-based) — `embed-code setup`, `embed-code embed`, `embed-code info`
- File and text embedding with full config control

## License

Apache 2.0
