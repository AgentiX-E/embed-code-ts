/**
 * Node.js Platform Adapter
 *
 * Uses onnxruntime-node for high-performance ONNX inference.
 * The onnxruntime-node native binding is an optional peer dependency —
 * users who only need the web platform don't have to install it.
 */

import type { IInferenceSession, OrtTensor } from '../types';

let ortNode: any = null;
let ortAvailable = false;

async function ensureOrt(): Promise<any> {
  if (ortNode) return ortNode;
  if (ortAvailable === false) {
    throw new Error(
      'onnxruntime-node is not installed. Install it with:\n' +
        '  npm install onnxruntime-node',
    );
  }
  try {
    ortNode = await import('onnxruntime-node');
    ortAvailable = true;
    return ortNode;
  } catch {
    ortAvailable = false;
    throw new Error(
      'onnxruntime-node is not installed. Install it with:\n' +
        '  npm install onnxruntime-node',
    );
  }
}

/**
 * Create an ONNX inference session in Node.js.
 */
export async function createInferenceSession(
  modelBuffer: ArrayBuffer,
): Promise<IInferenceSession> {
  const ort = await ensureOrt();

  // The buffer must NOT be detached — copy if needed
  const safeBuffer = modelBuffer.slice(0) as ArrayBuffer;

  const session = await ort.InferenceSession.create(safeBuffer, {
    executionProviders: ['cpu'],
    graphOptimizationLevel: 'all',
    enableCpuMemArena: true,
    enableMemPattern: true,
    logSeverityLevel: 3, // Error only
  });

  return {
    async run(
      feeds: Record<string, OrtTensor>,
    ): Promise<Record<string, OrtTensor>> {
      const feedMap: Record<string, any> = {};

      for (const [name, tensor] of Object.entries(feeds)) {
        const TypedTensor = getOrtTensorType(tensor.type);

        let data: any;
        if (tensor.type === 'int64') {
          // ONNX Runtime requires BigInt64Array for int64
          data = BigInt64Array.from(
            Array.from(tensor.data as Int32Array).map((v) => BigInt(v)),
          );
        } else {
          data = tensor.data;
        }

        feedMap[name] = new TypedTensor(tensor.type, data, tensor.dims);
      }

      const results = await session.run(feedMap);

      const output: Record<string, OrtTensor> = {};
      for (const [name, tensor] of Object.entries(results) as [string, any][]) {
        output[name] = {
          data: new Float32Array(tensor.data as ArrayBuffer),
          dims: tensor.dims as number[],
          type: tensor.type as string,
        };
      }

      return output;
    },

    release(): void {
      session.release();
    },
  };
}

function getOrtTensorType(type: string): any {
  const ort = ortNode;
  if (!ort?.Tensor) {
    throw new Error('ONNX Runtime Tensor class not available');
  }
  return ort.Tensor;
}

/**
 * Check if onnxruntime-node is available.
 */
export async function isNodePlatformAvailable(): Promise<boolean> {
  try {
    await ensureOrt();
    return true;
  } catch {
    return false;
  }
}
