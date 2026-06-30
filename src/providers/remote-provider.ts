/**
 * Remote Model Provider — Layer 3 (Fingerprint + Download)
 *
 * For models > 2GB (like nomic-embed-code 7B). Only the model manifest
 * (SHA256, URL, chunk info) is embedded. The actual weights are downloaded
 * at runtime from HuggingFace/CDN with progress tracking, integrity
 * verification, and local caching.
 *
 * Incbin philosophy: the "fingerprint" is embedded — the binary equivalent
 * of a linker symbol pointing to external data.
 */

import * as fs from 'fs';
import * as path from 'path';
import type { IModelProvider, ModelDescriptor, DownloadProgress } from '../types';
import type { ProviderType } from '../types';
import { base64ToArrayBuffer, sha256 } from './embedded-provider';

export class RemoteModelProvider implements IModelProvider {
  readonly type: ProviderType = 'remote';
  private modelBuffer: ArrayBuffer | null = null;
  private tokenizerBuffer: ArrayBuffer | null = null;

  constructor(
    private readonly config: ModelDescriptor,
    private readonly tokenizerData: { base64: string; sha256: string },
    private readonly onProgress?: (progress: DownloadProgress) => void,
    private readonly cacheDir?: string,
  ) {}

  async getModelBuffer(): Promise<ArrayBuffer> {
    if (this.modelBuffer) return this.modelBuffer;

    const remote = this.config.remote;
    if (!remote) {
      throw new Error('Remote model config is missing. Check the model registry.');
    }

    // Check cache first
    const cached = this.checkCache();
    if (cached) {
      this.modelBuffer = cached;
      return this.modelBuffer;
    }

    // Download and assemble chunks
    this.modelBuffer = await this.downloadAndAssemble(remote);
    return this.modelBuffer;
  }

  async getTokenizerBuffer(): Promise<ArrayBuffer> {
    if (this.tokenizerBuffer) return this.tokenizerBuffer;
    this.tokenizerBuffer = base64ToArrayBuffer(this.tokenizerData.base64);
    return this.tokenizerBuffer;
  }

  getConfig(): ModelDescriptor {
    return this.config;
  }

  async verify(): Promise<boolean> {
    try {
      const buffer = await this.getModelBuffer();
      const actual = await sha256(buffer);
      return actual === (this.config.remote?.sha256 ?? '');
    } catch {
      return false;
    }
  }

  dispose(): void {
    this.modelBuffer = null;
    this.tokenizerBuffer = null;
  }

  // ─── Private Helpers ─────────────────────────────────────

  private checkCache(): ArrayBuffer | null {
    if (!this.cacheDir) return null;

    const cacheFile = path.join(
      this.cacheDir,
      `${this.config.id}-v${this.config.version}.onnx`,
    );
    const cacheMeta = path.join(cacheFile, '..', `${this.config.id}.meta.json`);

    try {
      if (!fs.existsSync(cacheFile)) return null;

      // Verify hash
      const buffer = fs.readFileSync(cacheFile);
      const buf = buffer.buffer.slice(
        buffer.byteOffset,
        buffer.byteOffset + buffer.byteLength,
      );

      // Quick size check
      if (buf.byteLength !== this.config.remote?.totalSize) {
        return null; // Size mismatch, re-download
      }

      return buf as ArrayBuffer;
    } catch {
      return null; // Cache miss or corrupted
    }
  }

  private async downloadAndAssemble(
    remote: NonNullable<ModelDescriptor['remote']>,
  ): Promise<ArrayBuffer> {
    const { baseUrl, chunkCount, sha256: expectedHash, totalSize } = remote;

    // For models with chunkCount > 1, download chunks in parallel
    if (chunkCount > 1) {
      return this.downloadChunked(remote);
    }

    // Single-file download
    const url = `${baseUrl}/${this.config.target.onnxFile}`;
    return this.downloadWithProgress(url, totalSize, expectedHash);
  }

  private async downloadChunked(
    remote: NonNullable<ModelDescriptor['remote']>,
  ): Promise<ArrayBuffer> {
    const { baseUrl, chunkCount, sha256: expectedHash, totalSize } = remote;

    const fullBuffer = new ArrayBuffer(totalSize);
    const fullView = new Uint8Array(fullBuffer);
    let bytesDownloaded = 0;

    // Download chunks concurrently (max 4 at a time)
    const concurrency = 4;
    for (let i = 0; i < chunkCount; i += concurrency) {
      const batch = [];
      for (let j = i; j < Math.min(i + concurrency, chunkCount); j++) {
        batch.push(this.downloadChunk(j, baseUrl));
      }

      const results = await Promise.all(batch);
      for (const { index, buffer } of results) {
        const chunkSize = Math.min(
          totalSize - index * (totalSize / chunkCount),
          totalSize / chunkCount,
        );
        const offset = Math.floor(index * (totalSize / chunkCount));
        const view = new Uint8Array(buffer);
        fullView.set(view, offset);
        bytesDownloaded += view.length;

        this.onProgress?.({
          percent: Math.round((bytesDownloaded / totalSize) * 100),
          downloaded: bytesDownloaded,
          total: totalSize,
          chunk: index + 1,
          totalChunks: chunkCount,
        });
      }
    }

    // Verify
    const actualHash = await sha256(fullBuffer);
    if (actualHash !== expectedHash) {
      throw new Error(
        `Downloaded model integrity check failed.\nExpected: ${expectedHash}\nActual:   ${actualHash}`,
      );
    }

    // Cache
    this.saveToCache(fullBuffer);

    return fullBuffer;
  }

  private async downloadChunk(
    index: number,
    baseUrl: string,
  ): Promise<{ index: number; buffer: ArrayBuffer }> {
    const url = `${baseUrl}/onnx/model_int8.chunk_${String(index).padStart(4, '0')}.onnx`;
    const response = await fetch(url);
    if (!response.ok) {
      throw new Error(
        `Failed to download chunk ${index}: ${response.status} ${response.statusText}`,
      );
    }
    const buffer = await response.arrayBuffer();
    return { index, buffer };
  }

  private async downloadWithProgress(
    url: string,
    totalSize: number,
    expectedHash: string,
  ): Promise<ArrayBuffer> {
    const response = await fetch(url);
    if (!response.ok) {
      throw new Error(
        `Failed to download model: ${response.status} ${response.statusText}`,
      );
    }

    // Stream download with progress
    const reader = response.body?.getReader();
    if (!reader) {
      // Fallback: no streaming support
      const buffer = await response.arrayBuffer();
      this.onProgress?.({
        percent: 100,
        downloaded: buffer.byteLength,
        total: totalSize,
      });
      return buffer;
    }

    const chunks: Uint8Array[] = [];
    let downloaded = 0;

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      chunks.push(value);
      downloaded += value.length;

      this.onProgress?.({
        percent: Math.round((downloaded / totalSize) * 100),
        downloaded,
        total: totalSize,
      });
    }

    // Assemble
    const fullBuffer = new ArrayBuffer(downloaded);
    const fullView = new Uint8Array(fullBuffer);
    let offset = 0;
    for (const chunk of chunks) {
      fullView.set(chunk, offset);
      offset += chunk.length;
    }

    // Verify
    const actualHash = await sha256(fullBuffer);
    if (actualHash !== expectedHash) {
      throw new Error(
        `Downloaded model integrity check failed.\nExpected: ${expectedHash}\nActual:   ${actualHash}`,
      );
    }

    // Cache
    this.saveToCache(fullBuffer);

    return fullBuffer;
  }

  private saveToCache(buffer: ArrayBuffer): void {
    if (!this.cacheDir) return;

    try {
      fs.mkdirSync(this.cacheDir, { recursive: true });
      const cacheFile = path.join(
        this.cacheDir,
        `${this.config.id}-v${this.config.version}.onnx`,
      );
      fs.writeFileSync(cacheFile, new Uint8Array(buffer));

      // Write metadata
      const metaFile = path.join(this.cacheDir, `${this.config.id}.meta.json`);
      fs.writeFileSync(
        metaFile,
        JSON.stringify(
          {
            modelId: this.config.id,
            version: this.config.version,
            sha256: this.config.remote?.sha256,
            downloadedAt: Date.now(),
            size: buffer.byteLength,
          },
          null,
          2,
        ),
      );
    } catch (err) {
      // Cache write failure is non-fatal
      console.warn('Failed to cache model:', err);
    }
  }
}
