/**
 * Chunked Model Provider — Layer 2 (Chunked Embedded)
 *
 * For models 200MB - 2GB. The model is split into 50MB base64 chunks at build time.
 * Each chunk is a separate TypeScript module, lazy-loaded at runtime.
 *
 * Incbin philosophy: binary data IS code, just split across multiple modules
 * for TypeScript JIT performance.
 */

import type { IModelProvider, ModelDescriptor, DownloadProgress } from '../types';
import type { ProviderType } from '../types';
import { base64ToArrayBuffer, sha256 } from './embedded-provider';

/** Chunk manifest */
interface ChunkManifest {
  totalSize: number;
  sha256: string;
  chunkSize: number;
  chunkCount: number;
  chunks: Array<{
    index: number;
    offset: number;
    size: number;
    sha256: string;
  }>;
}

/**
 * Dynamic import function type for chunk modules.
 * Each chunk module exports: { CHUNK_BASE64: string, CHUNK_SHA256: string, CHUNK_INDEX: number }
 */
type ChunkImporter = (index: number) => Promise<{ default: { CHUNK_BASE64: string; CHUNK_SHA256: string; CHUNK_INDEX: number } }>;

export class ChunkedModelProvider implements IModelProvider {
  readonly type: ProviderType = 'chunked';
  private modelBuffer: ArrayBuffer | null = null;
  private tokenizerBuffer: ArrayBuffer | null = null;

  constructor(
    private readonly config: ModelDescriptor,
    private readonly manifest: ChunkManifest,
    private readonly loadChunk: ChunkImporter,
    private readonly tokenizerData: { base64: string; sha256: string },
    private readonly onProgress?: (progress: DownloadProgress) => void,
  ) {}

  async getModelBuffer(): Promise<ArrayBuffer> {
    if (this.modelBuffer) return this.modelBuffer;

    const { totalSize, sha256: expectedHash, chunkCount } = this.manifest;

    // Allocate the final buffer
    const fullBuffer = new ArrayBuffer(totalSize);
    const fullView = new Uint8Array(fullBuffer);
    let bytesAssembled = 0;

    // Load chunks sequentially (to keep memory pressure low)
    for (let i = 0; i < chunkCount; i++) {
      const chunk = await this.loadChunk(i);
      const chunkBuffer = base64ToArrayBuffer(chunk.default.CHUNK_BASE64);

      // Verify chunk integrity
      const chunkHash = await sha256(chunkBuffer);
      if (chunkHash !== chunk.default.CHUNK_SHA256) {
        throw new Error(`Chunk ${i} integrity check failed.`);
      }

      // Copy into final buffer at correct offset
      const chunkView = new Uint8Array(chunkBuffer);
      const offset = this.manifest.chunks[i]!.offset;
      fullView.set(chunkView, offset);
      bytesAssembled += chunkView.length;

      // Report progress
      this.onProgress?.({
        percent: Math.round((bytesAssembled / totalSize) * 100),
        downloaded: bytesAssembled,
        total: totalSize,
        chunk: i + 1,
        totalChunks: chunkCount,
      });
    }

    // Final integrity check
    const actualHash = await sha256(fullBuffer);
    if (actualHash !== expectedHash) {
      throw new Error(
        `Assembled model integrity check failed.\nExpected: ${expectedHash}\nActual:   ${actualHash}`,
      );
    }

    this.modelBuffer = fullBuffer;
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
      return actual === this.manifest.sha256;
    } catch {
      return false;
    }
  }

  dispose(): void {
    this.modelBuffer = null;
    this.tokenizerBuffer = null;
  }
}
