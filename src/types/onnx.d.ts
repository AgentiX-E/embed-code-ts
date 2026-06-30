/**
 * Type declarations for optional peer dependencies.
 * These modules are dynamically imported and may not be installed.
 */

declare module 'onnxruntime-node' {
  export class InferenceSession {
    static create(
      buffer: ArrayBuffer,
      options?: InferenceSessionOptions,
    ): Promise<InferenceSession>;
    run(feeds: Record<string, Tensor>): Promise<Record<string, Tensor>>;
    release(): void;
  }

  export class Tensor {
    constructor(type: string, data: any, dims: number[]);
    type: string;
    data: any;
    dims: number[];
  }

  export interface InferenceSessionOptions {
    executionProviders?: string[];
    graphOptimizationLevel?: 'disabled' | 'basic' | 'extended' | 'all';
    enableCpuMemArena?: boolean;
    enableMemPattern?: boolean;
    logSeverityLevel?: number;
  }
}

declare module 'onnxruntime-web' {
  export class InferenceSession {
    static create(
      buffer: ArrayBuffer,
      options?: InferenceSessionOptions,
    ): Promise<InferenceSession>;
    run(feeds: Record<string, Tensor>): Promise<Record<string, Tensor>>;
    release(): void;
  }

  export class Tensor {
    constructor(type: string, data: any, dims: number[]);
    type: string;
    data: any;
    dims: number[];
  }

  export interface InferenceSessionOptions {
    executionProviders?: string[];
    graphOptimizationLevel?: 'disabled' | 'basic' | 'extended' | 'all';
    enableCpuMemArena?: boolean;
    enableMemPattern?: boolean;
  }
}

declare module 'onnxruntime-common' {
  export class InferenceSession {
    static create(
      buffer: ArrayBuffer,
      options?: any,
    ): Promise<InferenceSession>;
    run(feeds: Record<string, Tensor>): Promise<Record<string, Tensor>>;
    release(): void;
  }
  export class Tensor {
    constructor(type: string, data: any, dims: number[]);
    type: string;
    data: any;
    dims: number[];
  }
}
