/**
 * Build Orchestrator — the master build script for embed-code-ts.
 *
 * This script orchestrates the full incbin build pipeline:
 *  1. Read the model registry to determine what to embed
 *  2. Download models and tokenizers from HuggingFace
 *  3. Run incbin to convert binary files to TypeScript constants
 *  4. Generate weight index files and manifests
 *  5. Update the model registry with computed hashes
 *
 * Usage:
 *   tsx scripts/build.ts                    # Full build
 *   tsx scripts/build.ts --model nomic-embed-text-v1.5  # Build specific model
 *   tsx scripts/build.ts --dry-run          # Show what would be built
 *   tsx scripts/build.ts --tokenizer-only   # Only build tokenizer embeds
 */

import * as fs from 'fs';
import * as path from 'path';
import * as crypto from 'crypto';
import { incbin } from './incbin';
import { downloadModelFiles } from './download-model';

// ─── Configuration ─────────────────────────────────────────

const MODELS_DIR = path.resolve(process.cwd(), 'models');
const WEIGHTS_DIR = path.resolve(process.cwd(), 'src/weights');
const REGISTRY_PATH = path.resolve(process.cwd(), 'src/registry.ts');

interface BuildConfig {
  models?: string[]; // Specific models to build (default: all in registry)
  dryRun?: boolean;
  tokenizerOnly?: boolean;
  force?: boolean; // Force rebuild even if outputs exist
}

// ─── Main Build Function ───────────────────────────────────

export async function build(config: BuildConfig = {}): Promise<void> {
  console.log('🔨 embed-code-ts Build Orchestrator');
  console.log('═══════════════════════════════════\n');

  if (config.dryRun) {
    console.log('📋 DRY RUN — no files will be written\n');
  }

  // Ensure directories
  if (!config.dryRun) {
    fs.mkdirSync(WEIGHTS_DIR, { recursive: true });
    fs.mkdirSync(MODELS_DIR, { recursive: true });
  }

  // Load model definitions from the registry
  // We dynamically import the registry to avoid circular deps
  const registryPath = REGISTRY_PATH;
  const registryContent = fs.readFileSync(registryPath, 'utf-8');

  // Parse model entries from the registry
  const modelEntries = parseRegistryModels(registryContent);
  const modelsToBuild = config.models
    ? modelEntries.filter((m) => config.models!.includes(m.id))
    : modelEntries;

  if (modelsToBuild.length === 0) {
    console.log('ℹ No models to build.');
    return;
  }

  console.log(`Models to process: ${modelsToBuild.length}`);
  for (const model of modelsToBuild) {
    console.log(`  - ${model.id} (provider: ${model.provider})`);
  }
  console.log('');

  // Process each model
  for (const model of modelsToBuild) {
    await buildModel(model, config);
  }

  // Generate weights index
  if (!config.dryRun) {
    generateWeightsIndex(modelsToBuild);
  }

  console.log('\n✅ Build complete!\n');
}

// ─── Build Single Model ────────────────────────────────────

async function buildModel(
  model: ParsedModelEntry,
  config: BuildConfig,
): Promise<void> {
  console.log(`\n📦 Building: ${model.id}`);
  console.log('─'.repeat(50));

  const modelDir = path.join(MODELS_DIR, model.id);
  const repo = model.repository;
  const onnxFile = model.onnxFile;
  const tokenizerFile = model.tokenizerFile;

  // Step 1: Download model files
  if (!config.tokenizerOnly && !config.dryRun) {
    const filesToDownload = [onnxFile];
    if (tokenizerFile) filesToDownload.push(tokenizerFile);

    // Check if already downloaded
    const onnxPath = path.join(modelDir, onnxFile);
    const tokenizerPath = path.join(modelDir, tokenizerFile!);

    if (
      !config.force &&
      fs.existsSync(onnxPath) &&
      (!tokenizerFile || fs.existsSync(tokenizerPath))
    ) {
      console.log('  ✓ Model files already downloaded');
    } else {
      try {
        await downloadModelFiles(repo, filesToDownload, modelDir);
      } catch (err) {
        console.error(
          `  ⚠ Failed to download ${model.id}: ${(err as Error).message}`,
        );
        console.log('  ℹ Skipping this model. Run again when network is available.\n');
        return;
      }
    }
  }

  // Step 2: Incbin the tokenizer (always embedded, Layer 1)
  if (tokenizerFile && !config.dryRun) {
    const tokenizerPath = path.join(modelDir, tokenizerFile);
    if (fs.existsSync(tokenizerPath)) {
      const outputPath = path.join(WEIGHTS_DIR, 'tokenizer.inc.ts');
      console.log(`\n  🗜 Incbin: tokenizer → ${path.relative(process.cwd(), outputPath)}`);
      await incbin({
        inputPath: tokenizerPath,
        outputPath,
        exportName: 'TOKENIZER',
        compress: 'gzip',
        splitChunks: false, // Tokenizer is always single-file (< 20MB)
      });
    }
  }

  // Step 3: Incbin the model (depends on provider type)
  if (!config.tokenizerOnly && !config.dryRun) {
    const onnxPath = path.join(modelDir, onnxFile);
    if (!fs.existsSync(onnxPath)) {
      console.log(`  ⚠ ONNX file not found: ${onnxPath}`);
      console.log('  ℹ Skipping model embedding. Download first.\n');
      return;
    }

    const fileStats = fs.statSync(onnxPath);
    const fileSize = fileStats.size;
    const sha256 = crypto
      .createHash('sha256')
      .update(fs.readFileSync(onnxPath))
      .digest('hex');

    console.log(`\n  🗜 Incbin: model → weights/`);
    console.log(`     Size: ${formatBytes(fileSize)}`);
    console.log(`     SHA256: ${sha256.substring(0, 16)}...`);

    if (model.provider === 'embedded' && fileSize < 200 * 1024 * 1024) {
      // Layer 1: Fully embed
      const outputPath = path.join(WEIGHTS_DIR, 'model.inc.ts');
      await incbin({
        inputPath: onnxPath,
        outputPath,
        exportName: 'MODEL',
        compress: 'gzip',
        splitChunks: false,
      });
    } else if (model.provider === 'embedded' || model.provider === 'chunked') {
      // Layer 2: Chunked embed
      const outputPath = path.join(WEIGHTS_DIR, 'model.inc.ts');
      await incbin({
        inputPath: onnxPath,
        outputPath,
        exportName: 'MODEL',
        compress: 'gzip',
        splitChunks: true,
        generateManifest: true,
        chunkSize: 50 * 1024 * 1024, // 50MB chunks
      });
    } else if (model.provider === 'remote') {
      // Layer 3: Remote — just generate manifest
      console.log('     ℹ Layer 3 (Remote) — generating manifest only');
      generateRemoteManifest(model, fileSize, sha256);
    }
  }

  console.log(`  ✅ ${model.id} processed\n`);
}

// ─── Remote Manifest Generator ─────────────────────────────

function generateRemoteManifest(
  model: ParsedModelEntry,
  fileSize: number,
  sha256: string,
): void {
  const manifest = {
    modelId: model.id,
    label: model.label,
    provider: 'remote',
    precision: 'int8',
    target: {
      repository: model.repository,
      onnxFile: model.onnxFile,
    },
    runtime: model.runtime,
    remote: {
      baseUrl: `https://huggingface.co/${model.repository}/resolve/main`,
      chunkCount: Math.ceil(fileSize / (500 * 1024 * 1024)), // 500MB chunks
      sha256,
      totalSize: fileSize,
    },
    version: model.version,
  };

  const manifestPath = path.join(WEIGHTS_DIR, 'model.manifest.ts');
  const content = `/**
 * @generated — DO NOT EDIT
 * Remote model manifest for ${model.id}
 * Generated at ${new Date().toISOString()}
 */

export const MODEL_MANIFEST = ${JSON.stringify(manifest, null, 2)} as const;
`;

  fs.writeFileSync(manifestPath, content, 'utf-8');
  console.log(`     ✓ Manifest written: ${path.relative(process.cwd(), manifestPath)}`);
}

// ─── Weights Index Generator ───────────────────────────────

function generateWeightsIndex(models: ParsedModelEntry[]): void {
  const indexPath = path.join(WEIGHTS_DIR, 'index.ts');
  const content = `/**
 * @generated — DO NOT EDIT
 * Auto-generated weights index
 * Generated at ${new Date().toISOString()}
 */

// Model manifest exports
export { MODEL_MANIFEST } from './model.manifest';

// Embedded tokenizer
export { TOKENIZER_BASE64, TOKENIZER_SHA256, TOKENIZER_SIZE } from './tokenizer.inc';

// Model weights (if fully embedded)
// export { MODEL_BASE64, MODEL_SHA256, MODEL_SIZE } from './model.inc';

// Chunk manifest (if chunked)
// export { CHUNK_MANIFEST } from './chunk.manifest';
`;

  fs.writeFileSync(indexPath, content, 'utf-8');
  console.log(`  📄 Generated: ${path.relative(process.cwd(), indexPath)}`);
}

// ─── Registry Parser ───────────────────────────────────────

interface ParsedModelEntry {
  id: string;
  label: string;
  provider: 'embedded' | 'chunked' | 'remote';
  repository: string;
  onnxFile: string;
  tokenizerFile: string;
  runtime: {
    embeddingDim: number;
    maxTokens: number;
    poolingStrategy: string;
    normalize: boolean;
  };
  version: number;
}

function parseRegistryModels(content: string): ParsedModelEntry[] {
  const models: ParsedModelEntry[] = [];

  // Match each model entry in the registry
  // This is a simple parser for the TypeScript object literal format
  const modelRegex =
    /'([^']+)':\s*\{[^}]*(?:{[^}]*}[^}]*)*\}/g;

  let match: RegExpExecArray | null;
  while ((match = modelRegex.exec(content)) !== null) {
    try {
      const id = match[1]!;
      const block = match[0];

      const provider = extractString(block, 'provider') as
        | 'embedded'
        | 'chunked'
        | 'remote';
      const label = extractString(block, 'label');
      const repository = extractNestedString(block, 'target', 'repository');
      const onnxFile = extractNestedString(block, 'target', 'onnxFile');
      const tokenizerFile =
        extractNestedString(block, 'target', 'tokenizerFile') ||
        'tokenizer.json';
      const embeddingDim = extractNestedNumber(block, 'runtime', 'embeddingDim');
      const maxTokens = extractNestedNumber(block, 'runtime', 'maxTokens');
      const poolingStrategy = extractNestedString(block, 'runtime', 'poolingStrategy');
      const normalize = block.includes('normalize: true');
      const version = extractNumber(block, 'version');

      models.push({
        id,
        label,
        provider,
        repository,
        onnxFile,
        tokenizerFile,
        runtime: {
          embeddingDim,
          maxTokens,
          poolingStrategy,
          normalize,
        },
        version,
      });
    } catch (err) {
      console.warn(`  ⚠ Failed to parse model entry: ${(err as Error).message}`);
    }
  }

  return models;
}

function extractString(block: string, key: string): string {
  const match = block.match(new RegExp(`${key}:\\s*'([^']+)'`));
  return match?.[1] ?? '';
}

function extractNumber(block: string, key: string): number {
  const match = block.match(new RegExp(`${key}:\\s*(\\d+)`));
  return parseInt(match?.[1] ?? '0', 10);
}

function extractNestedString(
  block: string,
  parent: string,
  key: string,
): string {
  // Find the parent block
  const parentRegex = new RegExp(`${parent}:\\s*\\{([^}]*)\\}`, 's');
  const parentMatch = block.match(parentRegex);
  if (!parentMatch) return '';

  return extractString(parentMatch[1]!, key);
}

function extractNestedNumber(
  block: string,
  parent: string,
  key: string,
): number {
  const parentRegex = new RegExp(`${parent}:\\s*\\{([^}]*)\\}`, 's');
  const parentMatch = block.match(parentRegex);
  if (!parentMatch) return 0;

  return extractNumber(parentMatch[1]!, key);
}

// ─── Helpers ───────────────────────────────────────────────

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  if (bytes < 1024 * 1024 * 1024)
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  return `${(bytes / (1024 * 1024 * 1024)).toFixed(2)} GB`;
}

// ─── CLI Entry ─────────────────────────────────────────────

if (import.meta.url === `file://${process.argv[1]}`) {
  const args = process.argv.slice(2);

  build({
    models: extractFlag(args, '--model')
      ? extractFlag(args, '--model')!.split(',')
      : undefined,
    dryRun: args.includes('--dry-run'),
    tokenizerOnly: args.includes('--tokenizer-only'),
    force: args.includes('--force'),
  }).catch((err) => {
    console.error('❌ Build failed:', err.message);
    process.exit(1);
  });
}

function extractFlag(args: string[], flag: string): string | undefined {
  const idx = args.indexOf(flag);
  if (idx === -1) return undefined;
  return args[idx + 1];
}
