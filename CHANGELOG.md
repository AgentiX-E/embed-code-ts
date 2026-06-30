# Changelog

## 0.1.0 (2026-06-30)

### Added

- `@agentix-e/embed-code-core` — Core inference engine for nomic-embed-code
  - `EmbedCode.fromPretrained()` factory pattern
  - `downloadModel()` with streaming fetch, SHA-256 verification, and local caching
  - BPE tokenizer compatible with Qwen2.5 tokenizer
  - Last-token and mean pooling strategies
  - ONNX Runtime inference engine (CPU/CUDA/DML)
  - `model-descriptor.json` as single source of truth for architecture config
  - Proxy support for corporate network environments
  - ESM + CJS dual format build output

- `@agentix-e/embed-code-cli` — Command-line interface
  - `embed-code setup` — Download model
  - `embed-code embed` — Generate embeddings from CLI/file/stdin
  - `embed-code info` — Show model metadata

- Build pipeline
  - `scripts/pipeline.js` — CI/CD orchestration
  - `scripts/export-onnx.py` — PyTorch to ONNX int8 conversion

- CI/CD
  - GitHub Actions CI workflow
  - Model release workflow
  - ESLint + Prettier + husky pre-commit hooks
  - Changesets for version management
