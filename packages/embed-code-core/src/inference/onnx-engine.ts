/**
 * ONNX Inference Engine — wraps onnxruntime-node for embed-code-ts.
 *
 * Pattern: IInferenceEngine interface enables testability and
 * future platform expansion (browser, CUDA, DirectML).
 *
 * The engine manages ONNX session lifecycle:
 *   load(modelPath) → run(feeds) → dispose()
 */

import type { IInferenceEngine, OrtTensor } from '../types';
import { InferenceError } from '../errors';

let ortNode: any = null;

async function getOrt(): Promise<any> {
  if (ortNode) return ortNode;
  try {
    ortNode = await import('onnxruntime-node');
    return ortNode;
  } catch {
    throw new InferenceError(
      'onnxruntime-node is not installed. Install it with:\n  npm install onnxruntime-node',
    );
  }
}

export interface EngineOptions {
  executionProvider?: string;
  intraOpNumThreads?: number;
}

export class EmbedCodeInferenceEngine implements IInferenceEngine {
  private session: any = null;
  private _loaded = false;
  private readonly options: EngineOptions;

  constructor(options: EngineOptions = {}) {
    this.options = options;
  }

  isLoaded(): boolean {
    return this._loaded;
  }

  async load(modelPath: string, options?: { skipWarmup?: boolean }): Promise<void> {
    const ort = await getOrt();

    const ep = this.options.executionProvider ?? 'cpu';
    const sessionOptions: any = {
      executionProviders: [ep],
      graphOptimizationLevel: 'all',
      enableCpuMemArena: true,
      enableMemPattern: true,
      logSeverityLevel: 3, // Error only
    };

    if (this.options.intraOpNumThreads && ep === 'cpu') {
      sessionOptions.intraOpNumThreads = this.options.intraOpNumThreads;
    }

    this.session = await ort.InferenceSession.create(modelPath, sessionOptions);

    // Warmup: run a dummy inference to prime JIT/caches
    if (!options?.skipWarmup) {
      await this.warmup();
    }

    this._loaded = true;
  }

  async run(feeds: Record<string, OrtTensor>): Promise<Record<string, OrtTensor>> {
    const ort = await getOrt();
    if (!this.session) {
      throw new InferenceError('Engine not loaded. Call load() first.');
    }

    const feedMap: Record<string, any> = {};
    for (const [name, tensor] of Object.entries(feeds)) {
      let data: any;
      if (tensor.type === 'int64') {
        data = BigInt64Array.from(Array.from(tensor.data as Int32Array).map((v) => BigInt(v)));
      } else {
        data = tensor.data;
      }
      feedMap[name] = new ort.Tensor(tensor.type, data, tensor.dims);
    }

    const results: Record<string, any> = await this.session.run(feedMap);

    const output: Record<string, OrtTensor> = {};
    const resultKeys = Object.keys(results);
    for (const name of resultKeys) {
      const tensor = results[name];
      output[name] = {
        data: new Float32Array(tensor.data),
        dims: tensor.dims as number[],
        type: tensor.type as string,
      };
    }

    return output;
  }

  dispose(): Promise<void> {
    if (this.session) {
      this.session.release();
      this.session = null;
    }
    this._loaded = false;
    return Promise.resolve();
  }

  private async warmup(): Promise<void> {
    // Run a minimal dummy input to warm up the ONNX session
    try {
      await this.run({
        input_ids: {
          data: new Int32Array(512).fill(0),
          dims: [1, 512],
          type: 'int64',
        },
        attention_mask: {
          data: new Int32Array(512).fill(0),
          dims: [1, 512],
          type: 'int64',
        },
        token_type_ids: {
          data: new Int32Array(512).fill(0),
          dims: [1, 512],
          type: 'int64',
        },
      });
    } catch {
      // Warmup failure is non-fatal — model may use different input names
    }
  }
}
