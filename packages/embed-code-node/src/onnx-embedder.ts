/**
 * NodeEmbedder — ONNX Runtime Node.js code embedding.
 *
 * Implements IEmbedder using onnxruntime-node (AVX2/AVX-512 native).
 */
import type {
  IEmbedder,
  BatchOptions,
  ModelInfo,
  TokenizedInput,
} from '@agentix-e/embed-code-core';
import { WordPieceTokenizer, meanPool, l2Normalize } from '@agentix-e/embed-code-core';
import { NodeOrtBackend } from './ort-backend.js';
import * as path from 'node:path';
import * as fs from 'node:fs';

export interface NodeEmbedderOptions {
  modelPath: string;
  tokenizerPath?: string;
}

export class NodeEmbedder implements IEmbedder {
  private readonly backend = new NodeOrtBackend();
  private session: Awaited<ReturnType<NodeOrtBackend['createSession']>> | null = null;
  readonly dimensions: number;
  readonly maxSequenceLength: number;
  readonly modelInfo: ModelInfo;
  private tokenizer: WordPieceTokenizer;
  private _modelPath: string;

  private constructor(options: NodeEmbedderOptions, tokenizer: WordPieceTokenizer) {
    this._modelPath = options.modelPath;
    this.tokenizer = tokenizer;
    this.dimensions = 768;
    this.maxSequenceLength = 512;
    this.modelInfo = {
      name: 'nomic-embed-code',
      version: 'v1.5',
      dimensions: 768,
      maxSequenceLength: 512,
      vocabSize: tokenizer.vocabSize,
      quantization: 'int8',
    };
  }

  /** Create from a model file. The tokenizer.json is expected alongside the model. */
  static async create(options: NodeEmbedderOptions): Promise<NodeEmbedder> {
    const tokenizerPath =
      options.tokenizerPath ?? path.join(path.dirname(options.modelPath), 'tokenizer.json');
    if (!fs.existsSync(tokenizerPath)) {
      throw new Error(`Tokenizer not found: ${tokenizerPath}`);
    }
    const tokenizer = WordPieceTokenizer.fromFile(tokenizerPath, 512);
    const embedder = new NodeEmbedder(options, tokenizer);
    embedder.session = await embedder.backend.createSession(options.modelPath);
    return embedder;
  }

  /** Create from the embedded model in this package's models/ directory. */
  static async createFromPackage(): Promise<NodeEmbedder> {
    const __dirname = path.dirname(new URL(import.meta.url).pathname);
    const modelPath = path.join(__dirname, '..', 'models', 'nomic-embed-code-v1.5.int8.onnx');
    return NodeEmbedder.create({ modelPath });
  }

  async embed(text: string): Promise<Float32Array> {
    if (!this.session) throw new Error('Session not initialized');
    const tokens = this.tokenizer.tokenize(text);
    const feeds = this.buildFeeds(tokens, 1);

    const outputs = await this.session.run(feeds);
    const hidden = outputs.last_hidden_state.data as Float32Array;

    // Mean pool + L2 normalize
    const pooled = meanPool(
      hidden,
      tokens.attentionMask,
      1,
      this.maxSequenceLength,
      this.dimensions,
    );
    l2Normalize(pooled, 1, this.dimensions);
    return pooled;
  }

  async embedBatch(texts: string[], options?: BatchOptions): Promise<Float32Array[]> {
    const results: Float32Array[] = [];
    const concurrency = options?.concurrency ?? 4;
    let completed = 0;

    const processChunk = async (start: number, end: number) => {
      for (let i = start; i < end; i++) {
        const r = await this.embed(texts[i]!);
        results[i] = r;
        completed++;
        options?.onProgress?.(completed, texts.length);
      }
    };

    const chunkSize = Math.ceil(texts.length / concurrency);
    const workers: Promise<void>[] = [];
    for (let i = 0; i < concurrency; i++) {
      const start = i * chunkSize;
      if (start >= texts.length) break;
      const end = Math.min(start + chunkSize, texts.length);
      workers.push(processChunk(start, end));
    }
    await Promise.all(workers);
    return results;
  }

  async dispose(): Promise<void> {
    this.session?.release();
    this.session = null;
  }

  private buildFeeds(tokens: TokenizedInput, batch: number) {
    return {
      input_ids: this.backend.createTensor('int64', tokens.inputIds, [
        batch,
        this.maxSequenceLength,
      ]),
      attention_mask: this.backend.createTensor('int64', tokens.attentionMask, [
        batch,
        this.maxSequenceLength,
      ]),
      token_type_ids: this.backend.createTensor('int64', tokens.tokenTypeIds, [
        batch,
        this.maxSequenceLength,
      ]),
    };
  }
}
