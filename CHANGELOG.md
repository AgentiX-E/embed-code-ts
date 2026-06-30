# Changelog

All notable changes to embed-code-ts will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **`docs/ARCHITECTURE.md`** — comprehensive architecture documentation covering component design, data flow, type system, and design principles
- **`docs/MODEL-UPDATE.md`** — model update guide with dual-channel release architecture, incbin weight flow, and troubleshooting
- **`docs/index.html`** — GitHub Pages landing page with navigation cards for API docs, benchmarks, coverage, and source code
- **`docs/GETTING-STARTED.md`** — comprehensive usage documentation with quick start, API usage, CLI tools, configuration reference, output description, troubleshooting, and performance guide
- **Restructured README** — aligned with project standards: badges, architecture diagram, packages table, quick start (3 options), config reference, output shape reference, project structure, development guide, references, known limitations, documentation & reports links, system requirements, CLI quick reference, license with compatibility table
- **Restructured CONTRIBUTING** — development workflow, pre-submit checklist, model management, commit convention, code style, versioning & publishing guide

### Changed

- **Package READMEs** — updated with npm badges, API docs badges, standardized sections (overview, installation, quick start, API documentation, license)

## [0.1.0] — Initial

### Added

- Initial project scaffold with 2-package monorepo
- Int8 ONNX model inference engine (`@agentix-e/embed-code-core`)
- BPE tokenizer with code-aware vocabulary
- CLI tool with `embed-code setup` and `embed-code embed` commands (`@agentix-e/embed-code-cli`)
- Model downloader with proxy support (3-tier cascade: options → env vars → standard vars)
- SHA-256 checksum verification for downloaded models
- Last-token, mean, and CLS pooling strategies
- L2 normalization for output embeddings
- Cosine similarity computation
- Task prefix support (`search_query:`, `search_document:`)
- Full CI/CD pipeline: lint, unit test, build check, integration test, benchmark, deploy to GitHub Pages
- Model release workflow with automated validation
- TypeScript strict mode with 95%+ coverage thresholds
- Model descriptor system for future-proof model version management
