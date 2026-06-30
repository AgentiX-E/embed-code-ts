// Type declarations for onnxruntime-node (peer dependency)
declare module 'onnxruntime-node' {
  export class InferenceSession {
    static create(pathOrBuffer: string | ArrayBuffer, options?: any): Promise<InferenceSession>;
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
