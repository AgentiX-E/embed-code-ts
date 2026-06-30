# Contributing to embed-code-ts

## Development Environment

```bash
git clone https://github.com/AgentiX-E/embed-code-ts.git
cd embed-code-ts
npm install && npm run build:all
```

**Prerequisites**: Node.js ≥ 22.x, npm ≥ 10.x

## Project Structure

```
embed-code-ts/
├── packages/
│   ├── embed-code-core/       # Core inference engine
│   └── embed-code-cli/        # CLI tool
├── docs/                      # Documentation
├── scripts/                   # Pipeline/export scripts
├── .github/workflows/         # CI/CD automation
└── models/                    # Model descriptors (weights gitignored)
```

## Development Workflow

```bash
# Install dependencies and build
npm install && npm run build:all

# Run all tests (requires ONNX model)
npm test

# Watch mode
npm run test:watch

# Unit tests only (no model needed, fast — covers pure logic)
npm run test:unit

# Unit tests with coverage (≥95% thresholds enforced)
npm run test:unit:coverage

# Full coverage report (unit + integration, requires ONNX model, ≥95% thresholds)
npm run test:coverage

# Lint
npm run lint
npm run lint:fix

# Format
npm run format
npm run format:check
```

### Pre-submit Checklist

Before submitting a PR, ensure:

1. **Unit tests pass with ≥95% coverage**: `npm run test:unit:coverage`
2. **Integration tests pass with ≥95% coverage** (requires ONNX model): `npm run test:coverage`
3. **Lint passes**: `npm run lint && npm run format:check`
4. **Build succeeds**: `npm run build:all`

> **Local ↔ CI parity**: The local test configs use the exact same vitest configs as CI. If it passes locally, it will pass on CI — no surprises.

### Setting Up the ONNX Model Locally

For integration tests and benchmarks, you need the ~140 MB int8 ONNX model:

```bash
# Option A: One-click pipeline (exports model, runs tests, runs benchmarks)
npm run pipeline

# Option B: Export only (requires Python 3.10+ and PyTorch)
python3 scripts/export-onnx.py --output models/nomic-embed-text-v1.5-int8.onnx

# Option C: Download from GitHub Releases (requires network)
node -e "const {downloadModel}=require('@agentix-e/embed-code-core');downloadModel({dest:'./models/nomic-embed-text-v1.5-int8.onnx'})"
```

The model file is gitignored — it should be at `models/nomic-embed-text-v1.5-int8.onnx` for local development.

### Pre-commit Hook

On `git commit`, the **pre-commit hook** runs lint and format checks automatically.

## Model Management

```bash
# Export ONNX model
python3 scripts/export-onnx.py

# Check latest HuggingFace version
npm run check:latest

# Validate existing models
node scripts/check-model.js

# Run inference benchmarks
npm run benchmark
```

## Commit Convention

Use [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add new feature
fix: fix bug in X
docs: update documentation
test: add tests for Y
chore: update dependencies
perf: improve performance of Z
refactor: restructure module W
```

## Code Style

- TypeScript strict mode
- Functions over classes (utils module)
- Float32Array/Int32Array for all tensor operations
- No `any` types (unless required by ONNX Runtime interfaces)
- Clear export naming: `camelCase` function names, `PascalCase` class names
- Interfaces use `I` prefix (e.g. `IEmbedCode`, `IInferenceEngine`)

## Adding New Features

1. Implement in the corresponding `packages/*/src/`
2. Write vitest tests (unit + integration)
3. Run `npm run test:coverage` to confirm coverage thresholds
4. Update relevant documentation
5. Submit PR

## Model Updates

nomic-embed-code model versions are published on HuggingFace. Check for updates:

```bash
npm run check:latest
```

Export new model and run full regression:

```bash
npm run pipeline
```

## Versioning & Publishing

This project uses [Changesets](https://github.com/changesets/changesets) for version management:

```bash
# 1. Create a changeset (describes your changes)
npx changeset

# 2. Version packages (updates package.json versions + CHANGELOGs)
npx changeset version

# 3. Publish to npm (run by CI on release)
npx changeset publish
```

Both packages (`embed-code-core`, `embed-code-cli`) are **fixed together** — they always share the same version number.

**Important**: CI automatically publishes to npm when release tags are pushed. The release workflow handles OIDC-based npm provenance attestation.
