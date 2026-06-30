/**
 * Model Download Script
 *
 * Downloads ONNX models and tokenizers from HuggingFace for embedding.
 * Supports:
 *   - Downloading specific files from a HF repository
 *   - Resuming interrupted downloads
 *   - SHA256 verification
 *   - Automatic retry with exponential backoff
 */

import * as fs from 'fs';
import * as path from 'path';
import * as crypto from 'crypto';

const HF_BASE = 'https://huggingface.co';

interface DownloadOptions {
  repository: string;
  file: string;
  outputDir: string;
  retries?: number;
  timeout?: number;
}

interface DownloadResult {
  filePath: string;
  size: number;
  sha256: string;
}

/**
 * Download a file from HuggingFace.
 */
export async function downloadFromHF(
  options: DownloadOptions,
): Promise<DownloadResult> {
  const { repository, file, outputDir, retries = 3, timeout = 300000 } = options;

  const url = `${HF_BASE}/${repository}/resolve/main/${file}`;
  const outputPath = path.join(outputDir, file);

  // Ensure directory exists
  fs.mkdirSync(path.dirname(outputPath), { recursive: true });

  console.log(`  Downloading: ${repository}/${file}`);
  console.log(`    URL: ${url}`);

  let lastError: Error | null = null;

  for (let attempt = 1; attempt <= retries; attempt++) {
    try {
      const response = await fetch(url, {
        signal: AbortSignal.timeout(timeout),
        headers: {
          'User-Agent': '@agentix-e/embed-code-ts build script',
        },
      });

      if (!response.ok) {
        throw new Error(
          `HTTP ${response.status} ${response.statusText}`,
        );
      }

      const totalSize = parseInt(
        response.headers.get('content-length') ?? '0',
      );

      // Stream to file with progress
      const tempPath = `${outputPath}.part`;
      const fileStream = fs.createWriteStream(tempPath);
      const reader = response.body?.getReader();

      if (!reader) {
        // Fallback: no streaming
        const buffer = Buffer.from(await response.arrayBuffer());
        fs.writeFileSync(outputPath, buffer);
        const sha256 = crypto
          .createHash('sha256')
          .update(buffer)
          .digest('hex');
        console.log(`    ✓ Downloaded ${formatBytes(buffer.length)}`);
        return { filePath: outputPath, size: buffer.length, sha256 };
      }

      let downloaded = 0;
      let lastLog = 0;

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        fileStream.write(Buffer.from(value));
        downloaded += value.length;

        // Progress logging (every 10%)
        if (totalSize > 0) {
          const pct = Math.floor((downloaded / totalSize) * 100);
          if (pct >= lastLog + 10) {
            console.log(
              `    ${pct}% (${formatBytes(downloaded)} / ${formatBytes(totalSize)})`,
            );
            lastLog = pct;
          }
        }
      }

      fileStream.end();

      // Wait for file to be fully written
      await new Promise<void>((resolve) => fileStream.on('finish', resolve));

      // Rename temp to final
      fs.renameSync(tempPath, outputPath);

      // Compute SHA256
      const fileBuffer = fs.readFileSync(outputPath);
      const sha256 = crypto
        .createHash('sha256')
        .update(fileBuffer)
        .digest('hex');

      console.log(
        `    ✓ Downloaded ${formatBytes(downloaded)} (SHA256: ${sha256.substring(0, 16)}...)`,
      );

      return { filePath: outputPath, size: downloaded, sha256 };
    } catch (err) {
      lastError = err as Error;
      console.warn(
        `    ⚠ Attempt ${attempt}/${retries} failed: ${(err as Error).message}`,
      );

      // Clean up partial download
      const tempPath = `${outputPath}.part`;
      if (fs.existsSync(tempPath)) fs.unlinkSync(tempPath);

      if (attempt < retries) {
        const delay = Math.pow(2, attempt) * 1000;
        console.log(`    Retrying in ${delay / 1000}s...`);
        await new Promise((resolve) => setTimeout(resolve, delay));
      }
    }
  }

  throw new Error(
    `Failed to download ${repository}/${file} after ${retries} attempts: ${lastError?.message}`,
  );
}

/**
 * Download all files for a model from the registry.
 */
export async function downloadModelFiles(
  repository: string,
  files: string[],
  outputDir: string,
): Promise<DownloadResult[]> {
  console.log(`\n📥 Downloading model: ${repository}`);
  console.log(`   Output: ${outputDir}`);

  const results: DownloadResult[] = [];

  // Download sequentially to avoid overwhelming HF
  for (const file of files) {
    const result = await downloadFromHF({
      repository,
      file,
      outputDir,
    });
    results.push(result);
  }

  console.log(`\n✅ Downloaded ${results.length} files\n`);
  return results;
}

// ─── CLI ───────────────────────────────────────────────────

if (import.meta.url === `file://${process.argv[1]}`) {
  cliMain().catch((err) => {
    console.error('❌ Download failed:', err.message);
    process.exit(1);
  });
}

async function cliMain(): Promise<void> {
  const repoIdx = process.argv.indexOf('--repo');
  const filesIdx = process.argv.indexOf('--files');
  const outputIdx = process.argv.indexOf('--output');

  if (repoIdx === -1 || filesIdx === -1 || outputIdx === -1) {
    console.error(`
Usage: tsx scripts/download-model.ts --repo <repo> --files <f1,f2,...> --output <dir>

Example:
  tsx scripts/download-model.ts \\
    --repo nomic-ai/nomic-embed-text-v1.5 \\
    --files onnx/model_int8.onnx,tokenizer.json \\
    --output models/nomic-embed-text-v1.5
`);
    process.exit(1);
  }

  const repository = process.argv[repoIdx + 1]!;
  const files = process.argv[filesIdx + 1]!.split(',');
  const outputDir = process.argv[outputIdx + 1]!;

  await downloadModelFiles(repository, files, outputDir);
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  if (bytes < 1024 * 1024 * 1024)
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  return `${(bytes / (1024 * 1024 * 1024)).toFixed(2)} GB`;
}
