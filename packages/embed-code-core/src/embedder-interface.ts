/**
 * IEmbedder — code embedding abstraction.
 *
 * All platforms (Node/Web) implement this interface.
 * Consumers depend only on IEmbedder, never on platform specifics.
 */
export interface IEmbedder {
  /** Embed a single text → 768d Float32Array */
  embed(text: string): Promise<Float32Array>;

  /** Embed a batch with concurrency control + progress + backpressure */
  embedBatch(texts: string[], options?: BatchOptions): Promise<Float32Array[]>;

  /** Embedding dimension (nomic-embed-code: 768) */
  readonly dimensions: number;

  /** Maximum sequence length (nomic-embed-code: 512) */
  readonly maxSequenceLength: number;

  /** Model identifier */
  readonly modelInfo: ModelInfo;

  /** Release ONNX session / buffers */
  dispose(): Promise<void>;
}

export interface BatchOptions {
  /** Concurrency limit (default: os.cpus().length / 2) */
  concurrency?: number;
  /** Progress callback */
  onProgress?: (completed: number, total: number) => void;
  /** Per-item timeout in ms (default: 30000) */
  timeout?: number;
}

export interface ModelInfo {
  readonly name: string;
  readonly version: string;
  readonly dimensions: number;
  readonly maxSequenceLength: number;
  readonly vocabSize: number;
  readonly quantization: 'int8' | 'float32';
}
