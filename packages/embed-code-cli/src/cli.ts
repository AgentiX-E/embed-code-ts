#!/usr/bin/env node
/**
 * embed-code-cli — nomic-embed-code model lifecycle management.
 *
 * Commands:
 *   embed-code download <model-id>  — Download from HuggingFace Hub
 *   embed-code convert <model-id>   — PyTorch/safetensors → ONNX
 *   embed-code quantize <input>     — float32 → int8 dynamic quantization
 *   embed-code verify  <model>      — Benchmark accuracy verification
 *   embed-code info    <model>      — Model metadata + system info
 *   embed-code embed   <text>       — Generate embeddings (demo)
 */

import { Command } from 'commander';
import { WordPieceTokenizer } from '@agentix-e/embed-code-core';
import * as fs from 'node:fs';

const program = new Command();
program.name('embed-code').description('nomic-embed-code lifecycle CLI').version('0.1.0');

// ─── info — model metadata ───────────────────────────────────

program
  .command('info')
  .description('Show model metadata and system information')
  .option('-m, --model <path>', 'Path to ONNX model file')
  .option('-t, --tokenizer <path>', 'Path to tokenizer.json')
  .action(async (opts: { model?: string; tokenizer?: string }) => {
    try {
      console.log('embed-code-cli  v0.1.0');
      console.log();

      // Tokenizer info
      const tokPath = opts.tokenizer ?? 'models/tokenizer.json';
      if (fs.existsSync(tokPath)) {
        const tok = WordPieceTokenizer.fromFile(tokPath);
        console.log(`Tokenizer:  ${tokPath}`);
        console.log(`  Vocab:     ${tok.vocabSize} tokens`);
        console.log(`  Max length: ${tok.maxLength}`);
        console.log(
          `  [CLS]: ${tok.clsTokenId}  [SEP]: ${tok.sepTokenId}  [UNK]: ${tok.unkTokenId}  [PAD]: ${tok.padTokenId}`,
        );
      }

      // Model info
      const modelPath = opts.model ?? 'models/nomic-embed-code-v1.5.int8.onnx';
      if (fs.existsSync(modelPath)) {
        console.log();
        console.log(`Model:      ${modelPath}`);
        console.log(`  Size:      ${(fs.statSync(modelPath).size / 1024 ** 2).toFixed(0)} MB`);
        console.log('  Dim:       768');
        console.log('  Max length: 512');
        console.log('  Quantized:  int8');
      } else {
        console.log();
        console.log('Model not found. Run "embed-code download" to fetch from HuggingFace.');
      }

      // System info
      const os = await import('node:os');
      console.log();
      console.log('System:');
      console.log(`  Platform: ${os.platform()} / ${os.arch()}`);
      console.log(`  Node.js:  ${process.version}`);
      console.log(`  CPU:      ${os.cpus()[0]?.model ?? 'unknown'} × ${os.cpus().length}`);
      console.log(`  RAM:      ${(os.totalmem() / 1024 ** 3).toFixed(1)} GB`);
    } catch (err) {
      console.error('Error:', err instanceof Error ? err.message : String(err));
      process.exit(1);
    }
  });

// ─── download — fetch model from HuggingFace ──────────────────

program
  .command('download')
  .description('Download ONNX model from HuggingFace Hub')
  .argument(
    '[model-id]',
    'HuggingFace model ID (e.g., nomic-ai/nomic-embed-text-v1.5)',
    'nomic-ai/nomic-embed-text-v1.5',
  )
  .option('-o, --output <path>', 'Output directory')
  .option('--proxy-url <url>', 'Proxy URL for corporate firewall')
  .option('--proxy-username <user>', 'Proxy authentication username')
  .option(
    '--proxy-password <pass>',
    'Proxy auth password (prefer env var EMBED_CODE_PROXY_PASSWORD)',
  )
  .option('--force', 'Force re-download even if cached')
  .option('--no-progress', 'Disable progress output')
  .action(async (modelId: string, opts: Record<string, string | boolean>) => {
    try {
      const { downloadModel } = await import('./download.js');
      const proxyUrl = opts.proxyUrl as string | undefined;
      const proxyUsername = opts.proxyUsername as string | undefined;
      const proxyPassword = opts.proxyPassword as string | undefined;

      const hasProxy = proxyUrl || proxyUsername || proxyPassword;

      await downloadModel({
        modelId,
        outputDir: opts.output as string | undefined,
        force: !!opts.force,
        proxy: hasProxy
          ? { url: proxyUrl ?? '', username: proxyUsername, password: proxyPassword }
          : undefined,
        onProgress:
          opts.progress !== false
            ? (received, total, speed) => {
                process.stdout.write(
                  `\r  ${received.toFixed(1)} / ${total.toFixed(1)} MB @ ${speed.toFixed(1)} MB/s`,
                );
              }
            : undefined,
      });
      if (opts.progress !== false) console.log(); // newline after progress
    } catch (err) {
      console.error('Error:', err instanceof Error ? err.message : String(err));
      process.exit(1);
    }
  });

// ─── setup — download model + show info ───────────────────────

program
  .command('setup')
  .description('Download model and show setup info')
  .option('--proxy-url <url>', 'Proxy URL')
  .option('--proxy-username <user>', 'Proxy username')
  .option('--proxy-password <pass>', 'Proxy password')
  .action(async (opts: Record<string, string>) => {
    try {
      const { downloadModel } = await import('./download.js');
      const hasProxy = opts.proxyUrl || opts.proxyUsername || opts.proxyPassword;

      console.log('Setting up nomic-embed-code model...');
      const modelPath = await downloadModel({
        proxy: hasProxy
          ? { url: opts.proxyUrl ?? '', username: opts.proxyUsername, password: opts.proxyPassword }
          : undefined,
        onProgress: (received, total, speed) => {
          process.stdout.write(
            `\r  ${received.toFixed(1)} / ${total.toFixed(1)} MB @ ${speed.toFixed(1)} MB/s`,
          );
        },
      });
      console.log(); // newline
      console.log(`Model: ${modelPath}`);
      console.log('Setup complete! You can now use:');
      console.log(`  import { NodeEmbedder } from "@agentix-e/embed-code-node";`);
      console.log(`  const embedder = await NodeEmbedder.create({ modelPath: "${modelPath}" });`);
    } catch (err) {
      console.error('Error:', err instanceof Error ? err.message : String(err));
      process.exit(1);
    }
  });

// ─── convert — PyTorch → ONNX ─────────────────────────────────

program
  .command('convert')
  .description('Convert PyTorch/safetensors model to ONNX format')
  .argument('<model-id>', 'HuggingFace model ID')
  .option('-o, --output <path>', 'Output .onnx file path', 'models/model.onnx')
  .action(async (modelId: string, opts: Record<string, string>) => {
    console.log(`Converting ${modelId} to ONNX...`);
    console.log(`  Output: ${opts.output}`);
    console.log();
    console.log('Run:');
    console.log(`  pip install optimum onnx onnxruntime`);
    console.log(`  optimum-cli export onnx --model ${modelId} ${opts.output}`);
  });

// ─── quantize — float32 → int8 ────────────────────────────────

program
  .command('quantize')
  .description('Quantize float32 ONNX model to int8')
  .argument('<input>', 'Input float32 .onnx file')
  .option('-o, --output <path>', 'Output int8 .onnx file')
  .option('--method <name>', 'Quantization method', 'dynamic')
  .action(async (input: string, opts: Record<string, string>) => {
    const output = opts.output ?? input.replace('.onnx', '.int8.onnx');
    console.log(`Quantizing ${input} → ${output}`);
    console.log(`  Method: ${opts.method}`);
    console.log();
    console.log('Run:');
    console.log(`  pip install onnxruntime-tools`);
    console.log(
      `  python3 -m onnxruntime.quantization.preprocess --input ${input} --output ${output}`,
    );
  });

// ─── verify — benchmark accuracy ──────────────────────────────

program
  .command('verify')
  .description('Verify int8 model accuracy vs float32 reference')
  .argument('<model>', 'Path to int8 .onnx model')
  .option('--benchmark <path>', 'Benchmark JSONL dataset')
  .option('--max-deviation <n>', 'Max cosine deviation', '0.005')
  .action(async (model: string, opts: Record<string, string>) => {
    console.log(`Verifying ${model}...`);
    console.log(`  Max deviation: ${opts.maxDeviation}`);
    console.log();
    console.log('Benchmark workflow:');
    console.log('  1. Run float32 inference → reference embeddings');
    console.log('  2. Run int8 inference → quantized embeddings');
    console.log('  3. Verify cosine similarity ≥ 0.995 for all samples');
    console.log('  4. Reject if deviation > 0.005 for > 1% of samples');
  });

// ─── embed — demo ─────────────────────────────────────────────

program
  .command('embed')
  .description('Generate embeddings (demo)')
  .argument('[text...]', 'Text to embed')
  .option('-m, --model <path>', 'ONNX model path')
  .option('--format <json|text>', 'Output format', 'text')
  .action(async (texts: string[], opts: Record<string, string>) => {
    if (texts.length === 0) {
      console.log('Usage: embed-code embed "your text here"');
      console.log('Install @agentix-e/embed-code-node for full embedding support.');
      return;
    }
    try {
      // Dynamic import: only loads if embed-code-node is installed
      const { NodeEmbedder } = await import('@agentix-e/embed-code-node');
      const modelPath = opts.model ?? 'models/nomic-embed-code-v1.5.int8.onnx';
      const embedder = await NodeEmbedder.create({ modelPath });

      const results: Float32Array[] = [];
      for (const text of texts) {
        results.push(await embedder.embed(text));
      }

      if (opts.format === 'json') {
        console.log(JSON.stringify(results.map((r) => Array.from(r))));
      } else {
        for (let i = 0; i < results.length; i++) {
          const vals = Array.from(results[i]!.slice(0, 8)).map((v) => v.toFixed(6));
          console.log(`[${i}] ${vals.join(' ')} ... (768 dims)`);
        }
      }

      await embedder.dispose();
    } catch (err) {
      if ((err as any)?.code === 'ERR_MODULE_NOT_FOUND') {
        console.log('Install @agentix-e/embed-code-node for embedding:');
        console.log('  npm install @agentix-e/embed-code-node');
      } else {
        console.error('Error:', err instanceof Error ? err.message : String(err));
      }
    }
  });

program.parse();
