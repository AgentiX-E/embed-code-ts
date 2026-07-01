/**
 * IOrtBackend — ONNX Runtime abstraction.
 *
 * onnxruntime-node and onnxruntime-web share identical
 * InferenceSession/Tensor APIs at the JavaScript level.
 * This interface encodes the minimal surface needed by IEmbedder.
 */
export interface IOrtBackend {
  createSession(modelPath: string): Promise<IOrtSession>;
  createTensor(
    type: 'int64' | 'float32',
    data: number[] | Int32Array | Float32Array | BigInt64Array,
    dims: number[],
  ): IOrtTensor;
}

export interface IOrtSession {
  run(feeds: Record<string, IOrtTensor>): Promise<Record<string, IOrtTensor>>;
  release(): void;
}

export interface IOrtTensor {
  readonly type: string;
  readonly dims: readonly number[];
  readonly data: Float32Array | Int32Array | BigInt64Array;
  dispose(): void;
}
