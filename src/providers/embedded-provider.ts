/**
 * Embedded Model Provider — Layer 1 (Fully Embedded)
 *
 * Incbin philosophy: weights are compiled INTO the TypeScript bundle as base64 constants.
 * Zero network dependency at runtime. Works offline, in air-gapped environments.
 *
 * Suitable for models < 200MB (e.g., tokenizer, config, small distilled models).
 */

import type { IModelProvider, ModelDescriptor } from '../types';
import type { ProviderType } from '../types';

/** Interface for embedded weight modules */
interface EmbeddedWeightModule {
  WEIGHT_BASE64: string;
  WEIGHT_SHA256: string;
  WEIGHT_SIZE: number;
}

export class EmbeddedModelProvider implements IModelProvider {
  readonly type: ProviderType = 'embedded';
  private modelBuffer: ArrayBuffer | null = null;
  private tokenizerBuffer: ArrayBuffer | null = null;

  constructor(
    private readonly config: ModelDescriptor,
    private readonly weights: EmbeddedWeightModule,
    private readonly tokenizerData: EmbeddedWeightModule,
  ) {}

  async getModelBuffer(): Promise<ArrayBuffer> {
    if (this.modelBuffer) return this.modelBuffer;

    // Decode base64 → ArrayBuffer (the incbin decode step)
    this.modelBuffer = base64ToArrayBuffer(this.weights.WEIGHT_BASE64);

    // Verify integrity
    const actual = await sha256(this.modelBuffer);
    if (actual !== this.weights.WEIGHT_SHA256) {
      throw new Error(
        `Embedded model integrity check failed.\nExpected: ${this.weights.WEIGHT_SHA256}\nActual:   ${actual}`,
      );
    }

    return this.modelBuffer;
  }

  async getTokenizerBuffer(): Promise<ArrayBuffer> {
    if (this.tokenizerBuffer) return this.tokenizerBuffer;
    this.tokenizerBuffer = base64ToArrayBuffer(this.tokenizerData.WEIGHT_BASE64);
    return this.tokenizerBuffer;
  }

  getConfig(): ModelDescriptor {
    return this.config;
  }

  async verify(): Promise<boolean> {
    try {
      const buffer = await this.getModelBuffer();
      const actual = await sha256(buffer);
      return actual === this.weights.WEIGHT_SHA256;
    } catch {
      return false;
    }
  }

  dispose(): void {
    this.modelBuffer = null;
    this.tokenizerBuffer = null;
  }
}

// ─── Utilities ─────────────────────────────────────────────

/**
 * Convert base64 string to ArrayBuffer.
 * Optimized for both Node.js (Buffer) and browser (atob).
 */
export function base64ToArrayBuffer(base64: string): ArrayBuffer {
  // Node.js fast path: Buffer.from
  if (typeof Buffer !== 'undefined') {
    const buffer = Buffer.from(base64, 'base64');
    // Detach the underlying ArrayBuffer without copying
    return buffer.buffer.slice(
      buffer.byteOffset,
      buffer.byteOffset + buffer.byteLength,
    );
  }

  // Browser path: atob + manual decode
  const binary = atob(base64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) {
    bytes[i] = binary.charCodeAt(i);
  }
  return bytes.buffer;
}

/**
 * Convert ArrayBuffer to base64 string.
 */
export function arrayBufferToBase64(buffer: ArrayBuffer): string {
  const bytes = new Uint8Array(buffer);
  if (typeof Buffer !== 'undefined') {
    return Buffer.from(bytes).toString('base64');
  }
  let binary = '';
  for (let i = 0; i < bytes.length; i++) {
    binary += String.fromCharCode(bytes[i]!);
  }
  return btoa(binary);
}

/**
 * Compute SHA256 of an ArrayBuffer.
 * Uses Web Crypto API (browser) or Node.js crypto.
 */
export async function sha256(buffer: ArrayBuffer): Promise<string> {
  // Node.js
  if (typeof process !== 'undefined' && process.versions?.node) {
    const { createHash } = await import('crypto');
    return createHash('sha256').update(new Uint8Array(buffer)).digest('hex');
  }

  // Browser / Web Crypto
  const hashBuffer = await crypto.subtle.digest('SHA-256', buffer);
  const hashArray = Array.from(new Uint8Array(hashBuffer));
  return hashArray.map((b) => b.toString(16).padStart(2, '0')).join('');
}
