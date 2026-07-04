/**
 * Error classes for embed-code-ts.
 *
 * Provides a typed error hierarchy so consumers can distinguish
 * between different failure modes at runtime.
 */

export class EmbedCodeError extends Error {
  constructor(message: string, options?: { cause?: Error }) {
    super(message, options);
    this.name = 'EmbedCodeError';
  }
}

export class ModelNotFoundError extends EmbedCodeError {
  constructor(message: string) {
    super(message);
    this.name = 'ModelNotFoundError';
  }
}

export class DownloadError extends EmbedCodeError {
  constructor(
    message: string,
    public readonly statusCode?: number,
  ) {
    super(message);
    this.name = 'DownloadError';
  }
}

export class ChecksumMismatchError extends EmbedCodeError {
  constructor(message: string) {
    super(message);
    this.name = 'ChecksumMismatchError';
  }
}

export class ProxyAuthError extends EmbedCodeError {
  constructor(
    message: string,
    public readonly statusCode: number,
  ) {
    super(message);
    this.name = 'ProxyAuthError';
  }
}

export class InferenceError extends EmbedCodeError {
  constructor(message: string, cause?: Error) {
    super(message, { cause });
    this.name = 'InferenceError';
  }
}

export class TokenizationError extends EmbedCodeError {
  constructor(message: string, options?: { cause?: Error }) {
    super(message, options);
    this.name = 'TokenizationError';
  }
}

export class ModelNotCompiledError extends EmbedCodeError {
  constructor(message: string) {
    super(message);
    this.name = 'ModelNotCompiledError';
  }
}
