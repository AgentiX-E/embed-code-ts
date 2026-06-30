/**
 * Web/Browser Platform Adapter
 *
 * Uses onnxruntime-web for browser-based ONNX inference.
 * Model weights are loaded from embedded base64 data (Layer 1/2)
 * or downloaded via CDN (Layer 3).
 */

import type { IInferenceSession, OrtTensor } from '../types';

let ortWeb: any = null;
let ortAvailable: boolean | null = null;

async function ensureOrt(): Promise<any> {
  if (ortWeb) return ortWeb;
  if (ortAvailable === false) {
    throw new Error(
      'onnxruntime-web is not installed. Install it with:\n' +
        '  npm install onnxruntime-web',
    );
  }
  try {
    ortWeb = await import('onnxruntime-web');
    ortAvailable = true;
    return ortWeb;
  } catch {
    ortAvailable = false;
    throw new Error(
      'onnxruntime-web is not installed. Install it with:\n' +
        '  npm install onnxruntime-web',
    );
  }
}

/**
 * Create an ONNX inference session in the browser.
 */
export async function createInferenceSession(
  modelBuffer: ArrayBuffer,
): Promise<IInferenceSession> {
  const ort = await ensureOrt();

  const session = await ort.InferenceSession.create(modelBuffer, {
    executionProviders: ['wasm'],
    graphOptimizationLevel: 'all',
    enableCpuMemArena: true,
    enableMemPattern: true,
  });

  return {
    async run(
      feeds: Record<string, OrtTensor>,
    ): Promise<Record<string, OrtTensor>> {
      const feedMap: Record<string, any> = {};

      for (const [name, tensor] of Object.entries(feeds)) {
        // onnxruntime-web uses 'tensor()' helper
        feedMap[name] = new ort.Tensor(
          tensor.type,
          tensor.data,
          tensor.dims,
        );
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

/**
 * Check if onnxruntime-web is available (browser environment).
 */
export async function isWebPlatformAvailable(): Promise<boolean> {
  try {
    await ensureOrt();
    return true;
  } catch {
    return false;
  }
}
