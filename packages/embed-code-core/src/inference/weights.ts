/**
 * WeightBuffer — parses the embed-code-ts binary weight format.
 *
 * The `.weights.int8.bin` file layout:
 *   [Header: 256 bytes] → magic, version, architecture params
 *   [Offset Table]       → per-tensor name, offset, size, dims, scale_offset
 *   [Payload]            → int8 weight data + float32 scale arrays
 *
 * Provides O(1) random access to any named tensor via DataView slicing.
 */

/** Metadata for a single weight tensor */
export interface TensorMeta {
  readonly offset: number;
  readonly size: number;
  readonly dims: readonly number[];
  readonly scaleOffset: number;
}

/** Architecture parameters parsed from the binary header */
export interface ArchParams {
  readonly numLayers: number;
  readonly hiddenSize: number;
  readonly intermediateSize: number;
  readonly vocabSize: number;
  readonly maxPositions: number;
  readonly numHeads: number;
  readonly headDim: number;
}

// Binary format constants
const HEADER_SIZE = 256;
const MAGIC = 'EMBCODE1';
const ENTRY_NAME_LEN = 64;
const ENTRY_SIZE = 114; // 64(name) + 8(offset) + 8(size) + 2(rank) + 8(dims) + 8(scale_off) + 16(reserved)

export class WeightBuffer {
  readonly archParams: ArchParams;
  private readonly buffer: ArrayBuffer;
  private readonly dataView: DataView;
  private readonly tensorMap: Map<string, TensorMeta>;

  /**
   * Create a WeightBuffer from raw binary data (ArrayBuffer).
   * Parses the header and offset table on construction.
   */
  constructor(data: ArrayBuffer) {
    this.buffer = data;
    this.dataView = new DataView(data);
    this.tensorMap = new Map();

    // Parse header
    this.validateMagic();
    this.archParams = this.parseHeader();

    // Parse offset table
    this.parseOffsetTable();
  }

  /** Check if a tensor exists */
  has(name: string): boolean {
    return this.tensorMap.has(name);
  }

  /** Get tensor metadata */
  meta(name: string): TensorMeta | undefined {
    return this.tensorMap.get(name);
  }

  /** Get all tensor names */
  get names(): string[] {
    return Array.from(this.tensorMap.keys());
  }

  /** Get Int8Array view of a weight tensor */
  getInt8(name: string): Int8Array {
    const meta = this.tensorMap.get(name);
    if (!meta) throw new Error(`Tensor "${name}" not found in WeightBuffer`);
    return new Int8Array(this.buffer, meta.offset, meta.size);
  }

  /** Get Float32Array view of a scale tensor */
  getFloat32(name: string): Float32Array {
    const meta = this.tensorMap.get(name);
    if (!meta) throw new Error(`Tensor "${name}" not found in WeightBuffer`);
    return new Float32Array(this.buffer, meta.scaleOffset, meta.dims[0]!);
  }

  /** Get the entire raw ArrayBuffer */
  get rawBuffer(): ArrayBuffer {
    return this.buffer;
  }

  // ─── Private ──────────────────────────────────────────────

  private validateMagic(): void {
    for (let i = 0; i < MAGIC.length; i++) {
      if (this.dataView.getUint8(i) !== MAGIC.charCodeAt(i)) {
        throw new Error('Invalid weight file: bad magic number');
      }
    }
  }

  private parseHeader(): ArchParams {
    const dv = this.dataView;
    let off = 8; // after magic

    const version = dv.getUint32(off, true);
    off += 4;
    if (version !== 1) {
      throw new Error(`Unsupported weight file version: ${version} (expected 1)`);
    }

    const numLayers = dv.getUint16(off, true);
    off += 2;
    const hiddenSize = dv.getUint16(off, true);
    off += 2;
    const intermediateSize = dv.getUint32(off, true);
    off += 4;
    const vocabSize = dv.getUint32(off, true);
    off += 4;
    const maxPositions = dv.getUint32(off, true);
    off += 4;
    const numHeads = dv.getUint16(off, true);
    off += 2;
    const headDim = dv.getUint16(off, true);
    // remaining is reserved

    return {
      numLayers,
      hiddenSize,
      intermediateSize,
      vocabSize,
      maxPositions,
      numHeads,
      headDim,
    };
  }

  private parseOffsetTable(): void {
    const dv = this.dataView;

    // Read total size from header (byte 32, after all arch fields)
    const totalSize = Number(dv.getBigUint64(32, true));

    // Offset table starts right after header
    let off = HEADER_SIZE;

    while (off < totalSize) {
      // Read entry name (64 bytes, null-padded ASCII)
      let name = '';
      for (let i = 0; i < ENTRY_NAME_LEN; i++) {
        const ch = dv.getUint8(off + i);
        if (ch === 0) break;
        name += String.fromCharCode(ch);
      }

      if (name === '') break; // end of table

      off += ENTRY_NAME_LEN;

      const tensorOffset = Number(dv.getBigUint64(off, true));
      off += 8;
      const size = Number(dv.getBigUint64(off, true));
      off += 8;
      const rank = dv.getUint16(off, true);
      off += 2;
      const dims: number[] = [];
      for (let i = 0; i < 4; i++) {
        dims.push(dv.getUint16(off, true));
        off += 2;
      }
      const actualDims = dims.slice(0, rank);
      const scaleOffset = Number(dv.getBigUint64(off, true));
      off += 8;
      // Skip reserved (16 bytes)
      off += 16;

      this.tensorMap.set(name, {
        offset: tensorOffset,
        size,
        dims: actualDims,
        scaleOffset,
      });
    }
  }

  /**
   * Factory: create from a Node.js Buffer or Uint8Array.
   */
  static fromBuffer(buf: Uint8Array): WeightBuffer {
    // Copy into a standalone ArrayBuffer for independent lifecycle
    const ab = new ArrayBuffer(buf.byteLength);
    new Uint8Array(ab).set(buf);
    return new WeightBuffer(ab);
  }

  /**
   * Factory: create from a file path (Node.js).
   */
  static async fromFile(filePath: string): Promise<WeightBuffer> {
    const fs = await import('node:fs/promises');
    const data = await fs.readFile(filePath);
    return WeightBuffer.fromBuffer(new Uint8Array(data.buffer));
  }
}
