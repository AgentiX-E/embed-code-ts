# nomic-embed-code Model Update Guide

> This document describes the automated and manual model update flows for embed-code-ts.

---

## Automated Model Updates (Default)

The project uses a **dual-channel release architecture**:

### Code Channel

```
git tag v* → release.yml
              ├─ quality (lint + test)
              ├─ publish-npm (OIDC + provenance)
              └─ github-release (TypeDoc + release notes)
```

Code releases publish npm packages only. They do **not** include weight files.

### Model Channel

```
nightly.yml (cron: 2 AM UTC daily)
    │
    ├─ Compare HF revision of nomic-ai/nomic-embed-code
    │  against committed models/model-descriptor.json
    │
    ├─ New revision detected? → Trigger model-release.yml
    │     ├─ detect (idempotency check via model-<sha> tag)
    │     ├─ export-model (PyTorch → int8 weights with validation)
    │     ├─ validate (full test suite + benchmark)
    │     ├─ github-release (model-<sha> + model-latest tags)
    │     └─ update-manifest (commit descriptor → auto-close issue)
    │
    └─ No change → no-op
```

### Download Channel

```
npm install @agentix-e/embed-code-core
node -e "const {downloadModel}=require('@agentix-e/embed-code-core');downloadModel()"
    │
    └─ Downloads from: github.com/.../releases/download/model-latest/nomic-embed-text-v1.5-int8.weights.bin
       + model-descriptor.json for SHA-256 verification
```

The `model-latest` tag is a rolling pointer that always points to the most recently validated model release. Users always get the latest model without upgrading their npm package.

---

## Incbin-Inspired Architecture

The incbin-inspired approach keeps the npm package code-only while delivering models via on-demand download:

```
npm package (@agentix-e/embed-code-core)
│  ~50 KB TypeScript/JS code only
│  + model-descriptor.json (SHA-256 fingerprint)
│
└─ On first use: downloadModel()
   ├─ Fetches int8 weights from GitHub Releases
   ├─ SHA-256 verifies against descriptor
   └─ Caches to ~/.cache/embed-code-ts/

Subsequent runs: pure filesystem read — zero network, zero latency
```

---

## Manual Model Update

If you need to manually trigger a model export and release:

### 1. Trigger model release workflow

```bash
# Automatic (force re-release of current HF revision)
gh workflow run model-release.yml -f force=true

# Or via GitHub UI: Actions → Model Release → Run workflow
```

### 2. Local export for testing

```bash
# Full pipeline: check HF → export → validate → test → benchmark
npm run pipeline

# Export only (skip tests)
npm run pipeline:export
```

### 3. Update the committed descriptor

```bash
# After local export with a new HF revision
git add models/model-descriptor.json
git commit -m "chore(model): update descriptor to HF rev <sha>"
```

---

## ModelDescriptor Contract

The `model-descriptor.json` file (committed to the repo and distributed with each weight file release) defines the architecture contract between the model and the TypeScript engine:

| Field               | Source                    | Purpose                                    |
| ------------------- | ------------------------- | ------------------------------------------ |
| `schema`            | Constant (1)              | Forward compatibility version              |
| `model.hf_revision` | HuggingFace API           | Traceability to exact PyTorch checkpoint   |
| `weights`           | weight file               | Runtime shape validation                   |
| `weights.sha256`    | Computed from weight file | Download integrity verification            |
| `architecture.*`    | PyTorch model params      | Configures the TypeScript engine           |
| `tokenizer.*`       | Model config              | Vocabulary size, maxTokens, special tokens |

The engine reads this descriptor at runtime via `loadModelDescriptor()` and converts it to a `ModelConfig` via `descriptorToModelConfig()`. All hardcoded architecture constants in the TypeScript code have been eliminated — the descriptor is the single source of truth.

---

## Version Compatibility

| Schema | Engine Requirement                   | Notes                                                           |
| ------ | ------------------------------------ | --------------------------------------------------------------- |
| 1      | `@agentix-e/embed-code-core` ≥ 0.1.0 | Current                                                         |
| > 1    | Upgrade required                     | Engine logs warning and falls back to `NOMIC_EMBED_CODE_CONFIG` |

If a future nomic-embed-code model version has a different architecture, updating `export-weights.py` to generate the new descriptor is sufficient — no TypeScript code changes are needed as long as the schema version is compatible.

---

## Model Update Flow

When a new model version is released:

```
1. New HF model detected
       │
2. Export int8 weights (~137 MB)
       │
3. Validate:
   ├─ Model loads correctly ✓
   ├─ Embedding dimensions match ✓
   ├─ Full test suite passes ✓
   └─ Benchmark regression ≤ 5% ✓
       │
4. Release:
   ├─ GitHub Release: publish int8 weights + descriptor
   └─ Update model-latest tag
```

---

## Troubleshooting

| Issue                                | Solution                                                                                          |
| ------------------------------------ | ------------------------------------------------------------------------------------------------- |
| Nightly check not detecting changes  | Verify `models/model-descriptor.json` is committed with the current HF revision                   |
| Model release fails                  | Check the workflow run logs; force re-run with `force: true`                                      |
| `downloadModel()` fails              | Verify `model-latest` release exists and contains both `.weights.bin` and `model-descriptor.json` |
| `downloadModel()` fails behind proxy | Set `HTTPS_PROXY` or `EMBED_CODE_PROXY_URL` environment variables (see README Proxy section)      |
| `downloadModel()` fails with 407     | Proxy authentication required — set `EMBED_CODE_PROXY_USERNAME` and `EMBED_CODE_PROXY_PASSWORD`   |
| Engine rejects model                 | Descriptor schema > `ENGINE_SUPPORTED_SCHEMA` — upgrade `@agentix-e/embed-code-core`              |
| Local tests need weight file         | Export locally: `python3 scripts/export-weights.py` or `npm run pipeline:export`                  |
