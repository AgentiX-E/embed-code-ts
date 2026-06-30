# Contributing to embed-code-ts

## Development Setup

```bash
git clone https://github.com/AgentiX-E/embed-code-ts.git
cd embed-code-ts
npm install
```

## Project Structure

```
embed-code-ts/
├── packages/
│   ├── embed-code-core/     # Core inference engine
│   └── embed-code-cli/      # Command-line interface
├── scripts/
│   ├── pipeline.js          # CI/CD pipeline
│   └── export-onnx.py       # Model export script
├── models/
│   └── model-descriptor.json
└── docs/
```

## Commands

```bash
npm run build          # Build core package
npm run build:all      # Build all packages
npm test               # Run tests
npm run lint           # Lint
npm run format         # Format code
npm run format:check   # Check formatting
npm run typecheck      # Type check
npm run docs:generate  # Generate API docs
```

## Testing

- Unit tests run without requiring the ONNX model
- Integration tests are skipped when the model is not available
- Use `npm test` to run the full suite

## Code Style

- TypeScript strict mode
- ESLint with `@typescript-eslint/recommended-type-checked`
- Prettier for formatting
- Pre-commit hooks via husky + lint-staged

## Pull Requests

1. Create a feature branch from `main`
2. Make your changes
3. Run `npm run build:all && npm test && npm run lint && npm run format:check`
4. Submit a PR

## Releasing

Uses [Changesets](https://github.com/changesets/changesets):

```bash
npx changeset        # Create a changeset
npx changeset version # Bump versions
npm run build:all     # Build
npm publish -r        # Publish packages
```

## Model Export

To export the model:

```bash
pip install optimum onnx onnxruntime torch transformers
python3 scripts/export-onnx.py --output models/nomic-embed-code-v1-int8.onnx
```

## License

Apache-2.0
