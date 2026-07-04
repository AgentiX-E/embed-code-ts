/**
 * ORT backend adapter for onnxruntime-node.
 */
import type { IOrtBackend, IOrtSession, IOrtTensor } from '@agentix-e/embed-code-core';
import { int32ToBigInt64 } from '@agentix-e/embed-code-core';
import * as ort from 'onnxruntime-node';

export class NodeOrtBackend implements IOrtBackend {
  async createSession(modelPath: string): Promise<IOrtSession> {
    const session = await ort.InferenceSession.create(modelPath, {
      executionProviders: ['cpu'],
      graphOptimizationLevel: 'all',
      enableCpuMemArena: true,
      enableMemPattern: true,
      logSeverityLevel: 3,
    });
    return new NodeOrtSession(session);
  }

  createTensor(
    type: 'int64' | 'float32',
    data: number[] | Int32Array | Float32Array | BigInt64Array,
    dims: number[],
  ): IOrtTensor {
    let tensorData: any;
    if (type === 'int64') {
      tensorData = int32ToBigInt64(data as Int32Array | number[]);
    } else {
      tensorData = data;
    }
    return new NodeOrtTensor(new ort.Tensor(type, tensorData, dims));
  }
}

class NodeOrtSession implements IOrtSession {
  constructor(private session: ort.InferenceSession) {}
  async run(feeds: Record<string, IOrtTensor>): Promise<Record<string, IOrtTensor>> {
    const feedMap: Record<string, any> = {};
    for (const [k, v] of Object.entries(feeds)) {
      feedMap[k] = (v as NodeOrtTensor).tensor;
    }
    const results = await this.session.run(feedMap);
    const out: Record<string, IOrtTensor> = {};
    for (const [k, v] of Object.entries(results)) {
      out[k] = new NodeOrtTensor(v);
    }
    return out;
  }
  release(): void {
    this.session.release();
  }
}

class NodeOrtTensor implements IOrtTensor {
  constructor(public readonly tensor: ort.Tensor) {}
  get type(): string {
    return this.tensor.type;
  }
  get dims(): readonly number[] {
    return this.tensor.dims;
  }
  get data(): Float32Array | Int32Array | BigInt64Array {
    return this.tensor.data as Float32Array | Int32Array | BigInt64Array;
  }
  dispose(): void {
    if (this.tensor && typeof (this.tensor as any).dispose === 'function') {
      (this.tensor as any).dispose();
    }
  }
}
