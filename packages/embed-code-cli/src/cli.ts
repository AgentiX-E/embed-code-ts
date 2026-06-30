#!/usr/bin/env node
/**
 * Embed Code CLI — command-line code embeddings with nomic-embed-code.
 *
 * Usage:
 *   # First time: download model
 *   embed-code setup
 *
 *   # With proxy (corporate network)
 *   embed-code setup --proxy-url http://proxy.company.com:8080
 *
 *   # Show model info
 *   embed-code info
 *
 *   # Generate embeddings
 *   embed-code embed "def fact(n): return 1 if n <= 1 else n * fact(n-1)"
 *   embed-code embed --query "Calculate factorial" --doc "def fact(n): ..."
 *   embed-code embed --file code.py
 *
 * @module embed-code-cli
 */

import { Command } from 'commander';
import { existsSync } from 'node:fs';
import { readFileSync } from 'node:fs';

const program = new Command();

program
  .name('embed-code')
  .description('State-of-the-art code embeddings with nomic-embed-code')
  .version('0.1.0');

// ─── info — model metadata ─────────────────────────────────

program
  .command('info')
  .description('Show model metadata and system information')
  .option('-m, --model <path>', 'Path to ONNX model')
  .action(async (options: Record<string, unknown>) => {
    try {
      const core = await import('@agentix-e/embed-code-core');
      const modelPath =
        (options.model as string) || process.env.EMBED_CODE_MODEL_PATH || core.defaultModelPath();

      console.log('Embed Code CLI  —  @agentix-e/embed-code-cli  v0.1.0');
      console.log(`Model path:  ${modelPath}`);
      if (existsSync(modelPath)) {
        const { resolveModelConfig } = core;
        const { config, descriptor } = await resolveModelConfig(modelPath);
        if (descriptor) {
          console.log(`Model:       ${descriptor.model.name} ${descriptor.model.version}`);
          console.log(`Architecture: ${descriptor.model.base_architecture}`);
          console.log(`Precision:    ${descriptor.model.precision}`);
          console.log(`Embedding:    ${config.embeddingDim} dims`);
          console.log(`Max tokens:   ${config.maxTokens}`);
          console.log(`Pooling:      ${config.poolingStrategy}`);
          console.log(`ONNX size:    ${(descriptor.onnx.size_bytes / 1024 ** 2).toFixed(0)} MB`);
          console.log(`SHA256:       ${descriptor.onnx.sha256.slice(0, 16)}...`);
        }

        // System info
        const os = await import('node:os');
        console.log(`\nSystem:`);
        console.log(`  Platform: ${os.platform()} / ${os.arch()}`);
        console.log(`  Node.js:  ${process.version}`);
        console.log(`  CPU:      ${os.cpus()[0]?.model ?? 'unknown'} × ${os.cpus().length}`);
        console.log(`  RAM:      ${(os.totalmem() / 1024 ** 3).toFixed(1)} GB`);
      } else {
        console.log('Model not found. Run "embed-code setup" to download.');
      }
    } catch (err) {
      console.error('Error:', err instanceof Error ? err.message : String(err));
      process.exit(1);
    }
  });

// ─── setup — download model ────────────────────────────────

let _lastSetupPath: string | null = null;

program
  .command('setup')
  .description('Download the nomic-embed-code int8 ONNX model')
  .option('-f, --force', 'Force re-download even if already cached')
  .option('-o, --output <path>', 'Custom output path (default: ~/.cache/agentix-embed-code-ts/)')
  .option('--proxy-url <url>', 'Proxy URL for downloading through corporate firewall')
  .option('--proxy-username <user>', 'Proxy authentication username')
  .option('--proxy-password <pass>', 'Proxy authentication password (prefer env variable)')
  .option('--precision <int8|text-int8>', 'Model precision variant (default: int8)')
  .action(async (options: Record<string, unknown>) => {
    try {
      const core = await import('@agentix-e/embed-code-core');

      const proxyUrl =
        (options.proxyUrl as string) || process.env.EMBED_CODE_PROXY_URL || undefined;
      const proxyUsername =
        (options.proxyUsername as string) || process.env.EMBED_CODE_PROXY_USERNAME || undefined;
      const proxyPassword =
        (options.proxyPassword as string) || process.env.EMBED_CODE_PROXY_PASSWORD || undefined;

      const proxyConfig = proxyUrl
        ? { url: proxyUrl, username: proxyUsername, password: proxyPassword }
        : undefined;

      const dest = await core.downloadModel({
        force: options.force === true,
        dest: options.output as string | undefined,
        proxy: proxyConfig,
        precision: (options.precision as string) || 'int8',
      });

      _lastSetupPath = dest;
      console.log(`\nModel ready: ${dest}`);
      console.log(`   Run: embed-code embed "def fact(n): return 1"`);
    } catch (err) {
      console.error('Error:', err instanceof Error ? err.message : String(err));
      process.exit(1);
    }
  });

// ─── embed — generate embeddings ───────────────────────────

async function resolveModelPath(explicitPath: string | undefined): Promise<string> {
  if (explicitPath) return explicitPath;

  const core = await import('@agentix-e/embed-code-core');

  const envPath = process.env.EMBED_CODE_MODEL_PATH;
  if (envPath && existsSync(envPath)) return envPath;

  if (_lastSetupPath && existsSync(_lastSetupPath)) return _lastSetupPath;

  const cached = core.getCachedModelPath();
  if (cached) return cached;

  console.error('No model found. Downloading nomic-embed-code int8…');
  return core.downloadModel();
}

program
  .command('embed')
  .description('Generate embeddings for code or text')
  .argument('[text...]', 'Text to embed (or use --query/--doc/--file)')
  .option('-q, --query <text>', 'Query text (prepends "search_query: ")')
  .option('-d, --doc <text>', 'Document/code text (prepends "search_document: ")')
  .option('-f, --file <path>', 'Read input from file')
  .option('-m, --model <path>', 'Path to ONNX model (auto-download if omitted)')
  .option('--no-prefix', 'Skip automatic task prefix (search_query:/search_document:)')
  .option('--format <json|text>', 'Output format', 'text')
  .action(async (textArgs: string[], options: Record<string, unknown>) => {
    try {
      const resolvedPath = await resolveModelPath(options.model as string | undefined);
      if (!resolvedPath) throw new Error('Failed to resolve model path.');

      const core = await import('@agentix-e/embed-code-core');
      const embedder = await core.EmbedCode.fromPretrained({ modelPath: resolvedPath });

      // Collect inputs
      const inputs: string[] = [];
      const usePrefix = options.prefix !== false;

      if (options.query) {
        const prefix = usePrefix ? embedder.taskPrefixes.query : '';
        inputs.push(prefix + (options.query as string));
      }
      if (options.doc) {
        const prefix = usePrefix ? embedder.taskPrefixes.document : '';
        inputs.push(prefix + (options.doc as string));
      }
      if (options.file) {
        const content = readFileSync(options.file as string, 'utf-8');
        const prefix = usePrefix ? embedder.taskPrefixes.document : '';
        inputs.push(prefix + content);
      }
      if (textArgs.length > 0) {
        const prefix = usePrefix ? embedder.taskPrefixes.document : '';
        for (const t of textArgs) inputs.push(prefix + t);
      }

      if (inputs.length === 0) {
        // Read from stdin
        const chunks: Buffer[] = [];
        for await (const chunk of process.stdin) chunks.push(Buffer.from(chunk as Buffer));
        const stdinText = Buffer.concat(chunks).toString('utf-8').trim();
        if (stdinText) {
          const prefix = usePrefix ? embedder.taskPrefixes.document : '';
          inputs.push(prefix + stdinText);
        }
      }

      if (inputs.length === 0) {
        console.error('No input provided. Use --query, --doc, --file, or pipe text to stdin.');
        process.exit(1);
      }

      // Generate embeddings
      const result = await embedder.embed(inputs);

      const fmt = (options.format as string) || 'text';
      if (fmt === 'json') {
        const out = {
          shape: result.shape,
          elapsedMs: result.elapsedMs,
          embeddings: Array.from(result.embeddings),
        };
        console.log(JSON.stringify(out));
      } else {
        // Text format: one embedding per line, space-separated
        const dim = result.shape[1];
        for (let i = 0; i < result.shape[0]; i++) {
          const start = i * dim;
          const values: number[] = [];
          for (let j = 0; j < Math.min(dim, 8); j++) {
            values.push(Number(result.embeddings[start + j].toFixed(6)));
          }
          const tail = dim > 8 ? ` ... (${dim - 8} more)` : '';
          console.log(`[${i}] ${values.join(' ')}${tail}`);
        }
        console.error(`\nShape: [${result.shape.join(', ')}]  |  ${result.elapsedMs.toFixed(1)}ms`);
      }

      await embedder.dispose();
    } catch (err) {
      console.error('Error:', err instanceof Error ? err.message : String(err));
      process.exit(1);
    }
  });

program.parse();
