/**
 * Embed Code Model Downloader
 *
 * Downloads the pre-exported nomic-embed-code int8 ONNX model from
 * GitHub Releases.  The npm package is code-only (~50 KB).  The model
 * is stored as a GitHub Release asset and fetched on first use.
 *
 * Features:
 *   - Streaming download (no large heap buffer for multi-GB models)
 *   - SHA-256 integrity verification (from model-descriptor.json in release)
 *   - Progress callback throttled to 200ms
 *   - Automatic cache management (~/.cache/agentix-embed-code-ts/)
 *   - Proxy support via environment variables or options parameter
 *   - Cross-platform zip extraction
 *
 * Usage:
 *   import { downloadModel, defaultModelPath } from '@agentix-e/embed-code-core';
 *   const modelPath = await downloadModel();
 */

import * as fs from 'node:fs';
import * as path from 'node:path';
import * as os from 'node:os';
import { createWriteStream, existsSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { spawn } from 'node:child_process';
import { DownloadError, ChecksumMismatchError, ProxyAuthError } from './errors';
import type { DownloadOptions, ProxyConfig } from './types';

// ─── Configuration ──────────────────────────────────────────

const REPO = 'AgentiX-E/embed-code-ts';
const DESCRIPTOR_FILENAME = 'model-descriptor.json';

interface PrecisionProfile {
  readonly suffix: string;
  readonly zipFilename: string;
  readonly onnxFilename: string;
  readonly expectedZipSize: number;
  readonly minCachedSize: number;
}

const PRECISION_PROFILES: Readonly<Record<string, PrecisionProfile>> = Object.freeze({
  int8: {
    suffix: '-int8',
    zipFilename: 'nomic-embed-code-v1-int8.zip',
    onnxFilename: 'nomic-embed-code-v1-int8.onnx',
    expectedZipSize: 7000 * 1024 * 1024, // ~7GB for 7B int8
    minCachedSize: 6000 * 1024 * 1024,
  },
  // nomic-embed-text-v1.5 (137M params, much smaller)
  'text-int8': {
    suffix: '-text-int8',
    zipFilename: 'nomic-embed-text-v1.5-int8.zip',
    onnxFilename: 'nomic-embed-text-v1.5-int8.onnx',
    expectedZipSize: 137 * 1024 * 1024,
    minCachedSize: 100 * 1024 * 1024,
  },
});

const DEFAULT_PRECISION = 'int8';

function precisionProfile(precision: string = DEFAULT_PRECISION): PrecisionProfile {
  return PRECISION_PROFILES[precision] ?? PRECISION_PROFILES[DEFAULT_PRECISION];
}

function releaseUrl(precision: string): string {
  const profile = precisionProfile(precision);
  const channel = `embed-code-latest-${precision}`;
  return `https://github.com/${REPO}/releases/download/${channel}/${profile.zipFilename}`;
}

function defaultCacheDir(): string {
  const base = process.env.XDG_CACHE_HOME || path.join(os.homedir(), '.cache');
  return path.join(base, 'agentix-embed-code-ts');
}

export function defaultModelPath(precision?: string): string {
  return path.join(defaultCacheDir(), precisionProfile(precision).onnxFilename);
}

export function getCachedModelPath(): string | null {
  for (const prec of Object.keys(PRECISION_PROFILES)) {
    const p = defaultModelPath(prec);
    if (isModelCachedAtPath(p, precisionProfile(prec).minCachedSize)) return p;
  }
  return null;
}

// ─── Proxy Resolution ───────────────────────────────────────

function resolveProxyConfig(options?: DownloadOptions): ProxyConfig | null {
  if (options?.proxy?.url) return options.proxy;

  const timesfmProxyUrl = process.env.EMBED_CODE_PROXY_URL;
  if (timesfmProxyUrl) {
    return {
      url: timesfmProxyUrl,
      username: process.env.EMBED_CODE_PROXY_USERNAME || undefined,
      password: process.env.EMBED_CODE_PROXY_PASSWORD || undefined,
    };
  }

  const standardUrl =
    process.env.HTTPS_PROXY ||
    process.env.https_proxy ||
    process.env.HTTP_PROXY ||
    process.env.http_proxy;

  if (standardUrl) {
    return { url: standardUrl };
  }

  return null;
}

async function applyProxyToFetch(
  proxy: ProxyConfig | null,
): Promise<{ fetchOptions: any; restoreEnv?: () => void }> {
  if (!proxy) return { fetchOptions: {} };

  try {
    // Dynamic import — undici ships with Node.js ≥ 20
    const undici: any = await import('undici');
    const ProxyAgent = undici.ProxyAgent;

    let proxyUrl = proxy.url;
    if (proxy.username || proxy.password) {
      try {
        const parsed = new URL(proxy.url);
        parsed.username = proxy.username || '';
        parsed.password = proxy.password || '';
        proxyUrl = parsed.toString();
      } catch { /* use raw URL */ }
    }

    const dispatcher = new ProxyAgent({
      uri: proxyUrl,
      keepAliveTimeout: 10_000,
      keepAliveMaxTimeout: 30_000,
    });

    return { fetchOptions: { dispatcher: dispatcher as any } };
  } catch {
    // Fallback: use env vars
    const saved: Record<string, string | undefined> = {
      HTTPS_PROXY: process.env.HTTPS_PROXY,
      https_proxy: process.env.https_proxy,
    };
    let envUrl = proxy.url;
    if (proxy.username || proxy.password) {
      try {
        const parsed = new URL(proxy.url);
        parsed.username = proxy.username || '';
        parsed.password = proxy.password || '';
        envUrl = parsed.toString();
      } catch { /* use raw URL */ }
    }
    process.env.HTTPS_PROXY = envUrl;
    process.env.https_proxy = envUrl;

    return {
      fetchOptions: {},
      restoreEnv: () => {
        if (saved.HTTPS_PROXY !== undefined) process.env.HTTPS_PROXY = saved.HTTPS_PROXY;
        else delete process.env.HTTPS_PROXY;
        if (saved.https_proxy !== undefined) process.env.https_proxy = saved.https_proxy;
        else delete process.env.https_proxy;
      },
    };
  }
}

// ─── Core Download ──────────────────────────────────────────

export async function downloadModel(options: DownloadOptions = {}): Promise<string> {
  const prec = options.precision ?? DEFAULT_PRECISION;
  const profile = precisionProfile(prec);
  const dest = path.resolve(options.dest ?? defaultModelPath(prec));
  const force = options.force ?? false;
  const log = options.logger ?? ((msg: string) => console.error(msg));

  // Already cached?
  if (!force && isModelCachedAtPath(dest, profile.minCachedSize)) {
    return dest;
  }

  const cacheDir = path.dirname(dest);
  fs.mkdirSync(cacheDir, { recursive: true });

  const url = options.url ?? releaseUrl(prec);
  const zipDest = path.join(cacheDir, profile.zipFilename);
  const tmpZip = zipDest + '.tmp';

  log(`Downloading nomic-embed-code int8 model (${(profile.expectedZipSize / 1024 ** 2).toFixed(0)} MB)...`);
  log(`  From: ${url}`);
  log(`  To:   ${dest}`);

  const proxyConfig = resolveProxyConfig(options);
  const { fetchOptions: proxyFetchOpts, restoreEnv } = await applyProxyToFetch(proxyConfig);

  const fetchOptions: RequestInit & { dispatcher?: any } = {
    redirect: 'follow',
    headers: { Accept: 'application/octet-stream' },
    ...proxyFetchOpts,
  };

  let response: Response;
  try {
    response = await fetch(url, fetchOptions);
  } catch (err) {
    restoreEnv?.();
    const message = (err as Error).message || String(err);
    if (proxyConfig && (message.includes('proxy') || message.includes('ECONNREFUSED') || message.includes('ENOTFOUND') || message.includes('tunnel') || message.includes('407'))) {
      throw new DownloadError(
        `Failed to connect through proxy (${proxyConfig.url}): ${message}\nVerify proxy configuration and connectivity.`,
        0,
      );
    }
    throw new DownloadError(
      `Failed to download model: ${message}\nURL: ${url}\nIf the model is not available, export it locally:\n  pip install optimum onnx onnxruntime torch\n  python scripts/export-onnx.py --output ${dest}`,
      0,
    );
  }

  if (!response.ok) {
    restoreEnv?.();
    if (response.status === 407) {
      throw new ProxyAuthError(
        `Proxy authentication required (HTTP 407). Set EMBED_CODE_PROXY_URL/USERNAME/PASSWORD env vars.`,
        response.status,
      );
    }
    throw new DownloadError(
      `Failed to download model (HTTP ${response.status}): ${url}`,
      response.status,
    );
  }

  const total = parseInt(response.headers.get('content-length') || '0', 10);
  const totalMB = total > 0 ? total / 1024 ** 2 : profile.expectedZipSize / 1024 ** 2;
  const totalBytes = total > 0 ? total : profile.expectedZipSize;

  const fileStream = createWriteStream(tmpZip);
  const reader = response.body?.getReader();
  if (!reader) throw new DownloadError('No response body', 0);

  let received = 0;
  const startTime = Date.now();
  let lastLogAt = 0;
  let lastProgressAt = 0;

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      if (value) {
        received += value.length;

        const writeOk = fileStream.write(value);
        if (!writeOk) {
          await new Promise<void>((resolve, reject) => {
            fileStream.once('drain', () => { fileStream.removeAllListeners('error'); resolve(); });
            fileStream.once('error', (err) => { fileStream.removeAllListeners('drain'); reject(err); });
          });
        }

        const receivedMB = received / 1024 ** 2;
        const elapsed = (Date.now() - startTime) / 1000;
        const speed = elapsed > 0 ? receivedMB / elapsed : 0;

        if (options.onProgress) {
          const now = Date.now();
          if (now - lastProgressAt >= 200) {
            lastProgressAt = now;
            options.onProgress(receivedMB, totalMB, speed);
          }
        }

        if (Math.floor(receivedMB / 50) > lastLogAt) {
          lastLogAt = Math.floor(receivedMB / 50);
          const pct = total > 0 ? ((received / totalBytes) * 100).toFixed(0) : '?';
          log(`  ${receivedMB.toFixed(0)} / ${totalMB.toFixed(0)} MB (${pct}%) @ ${speed.toFixed(1)} MB/s`);
        }
      }
    }

    await new Promise<void>((resolve, reject) => {
      fileStream.end(() => resolve());
      fileStream.once('error', reject);
    });

    if (options.onProgress) {
      const finalMB = received / 1024 ** 2;
      const finalElapsed = (Date.now() - startTime) / 1000;
      options.onProgress(finalMB, totalMB, finalElapsed > 0 ? finalMB / finalElapsed : 0);
    }

    // Extract zip
    log('  Extracting...');
    await extractZip(tmpZip, cacheDir);

    // Verify SHA-256
    let expectedSha256: string | null = null;
    const descriptorPath = path.join(cacheDir, DESCRIPTOR_FILENAME);
    try {
      const desc = JSON.parse(fs.readFileSync(descriptorPath, 'utf-8'));
      expectedSha256 = desc?.onnx?.sha256 ?? null;
    } catch { /* skip verification */ }

    if (expectedSha256) {
      const actualSha256 = await sha256File(dest);
      if (actualSha256 !== expectedSha256) {
        cleanupPartial(cacheDir, profile);
        restoreEnv?.();
        throw new ChecksumMismatchError(
          `Checksum mismatch!\n  Expected: ${expectedSha256}\n  Got:      ${actualSha256}`,
        );
      }
    }

    try { fs.unlinkSync(tmpZip); } catch { /* best-effort */ }

    const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
    log(`  Downloaded & extracted ${(received / 1024 ** 2).toFixed(0)} MB in ${elapsed}s → ${dest}`);
    restoreEnv?.();
    return dest;
  } catch (err) {
    restoreEnv?.();
    try { fs.unlinkSync(tmpZip); } catch { /* best-effort */ }
    throw err;
  }
}

// ─── Zip Extraction ─────────────────────────────────────────

async function extractZip(zipPath: string, outDir: string): Promise<void> {
  const backends: Array<() => Promise<void>> = [
    () => spawnExtractor('unzip', ['-o', zipPath, '-d', outDir]),
    () => spawnExtractor('7z', ['x', `-o${outDir}`, '-y', zipPath]),
    () => {
      const escapedZip = zipPath.replace(/'/g, "''");
      const escapedDir = outDir.replace(/'/g, "''");
      return spawnExtractor('powershell', ['-NoProfile', '-Command', `Expand-Archive -Path '${escapedZip}' -DestinationPath '${escapedDir}' -Force`]);
    },
  ];

  const errors: string[] = [];
  for (const backend of backends) {
    try { await backend(); return; } catch (err) { errors.push((err as Error).message); }
  }

  const platform = process.platform;
  const installHint = platform === 'win32' ? 'Install 7-Zip: winget install 7zip.7zip'
    : platform === 'darwin' ? 'Install unzip: brew install unzip'
    : 'Install unzip: apt-get install unzip';

  throw new DownloadError(
    `Failed to extract model zip:\n${errors.map((e, i) => `  ${i + 1}. ${e}`).join('\n')}\n\nFix: ${installHint}`,
    0,
  );
}

function spawnExtractor(command: string, args: string[]): Promise<void> {
  return new Promise((resolve, reject) => {
    const proc = spawn(command, args, { stdio: ['ignore', 'ignore', 'pipe'], timeout: 300_000 });
    let stderr = '';
    proc.stderr?.on('data', (d: Buffer) => { stderr += d.toString(); });
    proc.on('error', (err: NodeJS.ErrnoException) => {
      if (err.code === 'ENOENT') reject(new Error(`"${command}" not found on PATH`));
      else reject(new Error(`${command}: ${err.message}`));
    });
    proc.on('close', (code: number) => {
      if (code === 0) resolve();
      else reject(new Error(`${command} exited with code ${code}: ${stderr.slice(0, 200)}`));
    });
  });
}

// ─── SHA-256 ────────────────────────────────────────────────

async function sha256File(filePath: string): Promise<string> {
  const { statSync, createReadStream } = await import('node:fs');
  const { pipeline } = await import('node:stream/promises');

  const fileSize = statSync(filePath).size;
  if (fileSize < 100 * 1024 * 1024) {
    return sha256FileSync(filePath);
  }

  const hasher = createHash('sha256');
  const readStream = createReadStream(filePath, { highWaterMark: 64 * 1024 });
  try {
    await pipeline(readStream, hasher);
  } catch {
    return sha256FileSync(filePath);
  }
  return hasher.digest('hex');
}

function sha256FileSync(filePath: string): string {
  const hasher = createHash('sha256');
  const buf = Buffer.alloc(64 * 1024);
  const fd = fs.openSync(filePath, 'r');
  try {
    let bytes: number;
    while ((bytes = fs.readSync(fd, buf, 0, buf.length, null)) > 0) {
      hasher.update(buf.subarray(0, bytes));
    }
  } finally {
    fs.closeSync(fd);
  }
  return hasher.digest('hex');
}

function cleanupPartial(cacheDir: string, profile: PrecisionProfile): void {
  for (const f of [profile.onnxFilename, DESCRIPTOR_FILENAME]) {
    try { fs.unlinkSync(path.join(cacheDir, f)); } catch { /* ignore */ }
  }
}

// ─── Cache Helpers ──────────────────────────────────────────

function isModelCachedAtPath(p: string, minSize?: number): boolean {
  if (!existsSync(p)) return false;
  try {
    return fs.statSync(p).size >= (minSize ?? 100 * 1024 * 1024);
  } catch {
    return false;
  }
}

export function isModelCached(): boolean {
  return getCachedModelPath() !== null;
}
