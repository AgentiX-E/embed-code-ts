import * as fs from 'node:fs';
import * as path from 'node:path';
import * as crypto from 'node:crypto';
import { createWriteStream } from 'node:fs';
import type { ProxyConfig } from '@agentix-e/embed-code-core';

export interface DownloadOptions {
  modelId?: string;
  outputDir?: string;
  proxy?: ProxyConfig;
  onProgress?: (received: number, total: number, speed: number) => void;
  force?: boolean;
  logger?: (msg: string) => void;
}

export async function downloadModel(options: DownloadOptions = {}): Promise<string> {
  const modelId = options.modelId ?? 'nomic-ai/nomic-embed-text-v1.5';
  const outputDir = options.outputDir ?? path.join(getCacheDir(), modelId.replace('/', '_'));
  const log = options.logger ?? console.log;

  const modelFile = path.join(outputDir, 'nomic-embed-code-v1.5.int8.onnx');
  const descriptorFile = path.join(outputDir, 'model-descriptor.json');

  if (fs.existsSync(modelFile) && !options.force) {
    log(`Model already cached at ${modelFile}`);
    return modelFile;
  }

  fs.mkdirSync(outputDir, { recursive: true });

  // Resolve proxy settings
  const proxyUrl =
    options.proxy?.url ??
    process.env.EMBED_CODE_PROXY_URL ??
    process.env.HTTPS_PROXY ??
    process.env.HTTP_PROXY ??
    process.env.https_proxy ??
    process.env.http_proxy;

  const proxyUsername = options.proxy?.username ?? process.env.EMBED_CODE_PROXY_USERNAME;
  const proxyPassword = options.proxy?.password ?? process.env.EMBED_CODE_PROXY_PASSWORD;

  if (proxyUrl) {
    log(`Using proxy: ${proxyUrl.replace(/\/\/.*@/, '//***@')}`);
  }

  // Download from HuggingFace Hub
  const hfUrl = `https://huggingface.co/${modelId}/resolve/main/onnx/model_int8.onnx`;
  log(`Downloading from ${hfUrl}...`);

  await downloadFile(hfUrl, modelFile, {
    proxyUrl,
    proxyUsername,
    proxyPassword,
    onProgress: options.onProgress,
  });

  // Download model-descriptor.json
  const descriptorUrl = `https://huggingface.co/${modelId}/resolve/main/model-descriptor.json`;
  try {
    await downloadFile(descriptorUrl, descriptorFile, {
      proxyUrl,
      proxyUsername,
      proxyPassword,
    });
  } catch {
    log('Model descriptor not found on HuggingFace, using local copy if available.');
  }

  // Verify SHA256 if descriptor exists
  if (fs.existsSync(descriptorFile)) {
    const descriptor = JSON.parse(fs.readFileSync(descriptorFile, 'utf-8'));
    const expectedSha = descriptor.weights?.sha256 ?? descriptor.onnx?.sha256;
    if (expectedSha) {
      const actualSha = await sha256File(modelFile);
      if (actualSha !== expectedSha) {
        throw new Error(
          `Checksum mismatch for ${modelFile}:\n  Expected: ${expectedSha}\n  Actual:   ${actualSha}`,
        );
      }
      log('SHA256 checksum verified.');
    }
  }

  log(`Model downloaded to ${modelFile}`);
  return modelFile;
}

async function downloadFile(
  url: string,
  dest: string,
  options: {
    proxyUrl?: string;
    proxyUsername?: string;
    proxyPassword?: string;
    onProgress?: (received: number, total: number, speed: number) => void;
  },
): Promise<void> {
  // Use native fetch with Node.js
  const controller = new AbortController();

  // Configure proxy via undici if proxyUrl is provided
  const fetchOptions: RequestInit = {};
  if (options.proxyUrl) {
    // Set proxy via environment or undici dispatcher
    const dispatcher = await createProxyDispatcher(
      options.proxyUrl,
      options.proxyUsername,
      options.proxyPassword,
    );
    (fetchOptions as any).dispatcher = dispatcher;
  }

  const response = await fetch(url, { ...fetchOptions, signal: controller.signal });

  if (!response.ok) {
    throw new Error(`HTTP ${response.status}: ${response.statusText} (${url})`);
  }

  const total = Number(response.headers.get('content-length')) || 0;
  let received = 0;
  const startTime = Date.now();

  const reader = response.body!.getReader();
  const writeStream = createWriteStream(dest);

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      received += value.length;
      writeStream.write(value);

      if (options.onProgress && total > 0) {
        const elapsed = (Date.now() - startTime) / 1000;
        const speed = received / elapsed / (1024 * 1024); // MB/s
        options.onProgress(received / (1024 * 1024), total / (1024 * 1024), speed);
      }
    }
  } finally {
    reader.releaseLock();
    writeStream.end();
    // Wait for write to finish
    await new Promise<void>((resolve) => writeStream.on('finish', resolve));
  }
}

async function createProxyDispatcher(
  proxyUrl: string,
  username?: string,
  password?: string,
): Promise<undefined> {
  // Node.js 22+ natively supports HTTPS_PROXY / HTTP_PROXY env vars for fetch()
  // Set them for the duration of the download
  if (username && password) {
    const proxyUri = new URL(proxyUrl);
    proxyUri.username = username;
    proxyUri.password = password;
    process.env.HTTPS_PROXY = proxyUri.toString();
  } else {
    process.env.HTTPS_PROXY = proxyUrl;
  }
  return undefined;
}

function getCacheDir(): string {
  const home = process.env.HOME ?? process.env.USERPROFILE ?? '/tmp';
  return path.join(home, '.cache', 'agentix-embed-code-ts');
}

async function sha256File(filePath: string): Promise<string> {
  const hash = crypto.createHash('sha256');
  const stream = fs.createReadStream(filePath);
  return new Promise((resolve, reject) => {
    stream.on('data', (chunk) => hash.update(chunk));
    stream.on('end', () => resolve(hash.digest('hex')));
    stream.on('error', reject);
  });
}

export function getCachedModelPath(modelId?: string): string {
  const id = modelId ?? 'nomic-ai/nomic-embed-text-v1.5';
  return path.join(getCacheDir(), id.replace('/', '_'), 'nomic-embed-code-v1.5.int8.onnx');
}

export function isModelCached(modelId?: string): boolean {
  return fs.existsSync(getCachedModelPath(modelId));
}
