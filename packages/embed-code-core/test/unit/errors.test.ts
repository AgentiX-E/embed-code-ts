/**
 * Unit tests for error classes.
 */
import { describe, it, expect } from 'vitest';
import {
  EmbedCodeError,
  ModelNotFoundError,
  DownloadError,
  ChecksumMismatchError,
  ProxyAuthError,
  InferenceError,
  TokenizationError,
  ModelNotCompiledError,
} from '../../src/errors';

describe('Error classes', () => {
  it('EmbedCodeError has correct name', () => {
    const err = new EmbedCodeError('test');
    expect(err.name).toBe('EmbedCodeError');
    expect(err.message).toBe('test');
    expect(err).toBeInstanceOf(Error);
  });

  it('ModelNotFoundError inherits from EmbedCodeError', () => {
    const err = new ModelNotFoundError('model missing');
    expect(err).toBeInstanceOf(EmbedCodeError);
    expect(err).toBeInstanceOf(Error);
    expect(err.name).toBe('ModelNotFoundError');
  });

  it('DownloadError stores status code', () => {
    const err = new DownloadError('fail', 404);
    expect(err.statusCode).toBe(404);
  });

  it('DownloadError status code is optional', () => {
    const err = new DownloadError('fail');
    expect(err.statusCode).toBeUndefined();
  });

  it('ChecksumMismatchError is an EmbedCodeError', () => {
    const err = new ChecksumMismatchError('hash mismatch');
    expect(err).toBeInstanceOf(EmbedCodeError);
  });

  it('ProxyAuthError stores status code', () => {
    const err = new ProxyAuthError('auth required', 407);
    expect(err.statusCode).toBe(407);
  });

  it('InferenceError supports cause chaining', () => {
    const cause = new Error('inner');
    const err = new InferenceError('inference failed', cause);
    expect(err.cause).toBe(cause);
  });

  it('TokenizationError has correct name', () => {
    const err = new TokenizationError('bad token');
    expect(err.name).toBe('TokenizationError');
  });

  it('ModelNotCompiledError has correct name', () => {
    const err = new ModelNotCompiledError('not compiled');
    expect(err.name).toBe('ModelNotCompiledError');
  });

  it('EmbedCodeError supports cause chaining', () => {
    const inner = new Error('inner error');
    const err = new EmbedCodeError('outer error', { cause: inner });
    expect(err.cause).toBe(inner);
    expect(err.message).toBe('outer error');
  });

  it('DownloadError with status code provides useful info', () => {
    const err = new DownloadError('not found', 404);
    expect(err.message).toContain('not found');
    expect(err.statusCode).toBe(404);
    expect(err).toBeInstanceOf(EmbedCodeError);
  });

  it('ProxyAuthError is distinguishable from DownloadError', () => {
    const proxyErr = new ProxyAuthError('proxy auth required', 407);
    const downloadErr = new DownloadError('download failed', 500);
    expect(proxyErr).toBeInstanceOf(EmbedCodeError);
    expect(proxyErr).not.toBeInstanceOf(DownloadError);
    expect(downloadErr).toBeInstanceOf(EmbedCodeError);
  });

  it('all error classes have distinct names for runtime discrimination', () => {
    const errors = [
      new EmbedCodeError(''),
      new ModelNotFoundError(''),
      new DownloadError(''),
      new ChecksumMismatchError(''),
      new ProxyAuthError('', 407),
      new InferenceError(''),
      new TokenizationError(''),
      new ModelNotCompiledError(''),
    ];
    const names = new Set(errors.map((e) => e.name));
    expect(names.size).toBe(errors.length); // all distinct
  });
});
