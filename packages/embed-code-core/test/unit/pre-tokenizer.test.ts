/**
 * Unit tests for pre-tokenizer.
 */
import { describe, it, expect } from 'vitest';
import { preTokenize, isPunctuation } from '../../src/tokenizer/pre-tokenizer';

describe('preTokenizer', () => {
  it('splits on whitespace', () => {
    const r = preTokenize('hello world');
    expect(r).toEqual(['hello', 'world']);
  });

  it('lowercases input', () => {
    const r = preTokenize('Hello World');
    expect(r).toEqual(['hello', 'world']);
  });

  it('strips accents via NFKD', () => {
    const r = preTokenize('café');
    expect(r).toEqual(['cafe']);
  });

  it('splits CJK characters', () => {
    const r = preTokenize('你好世界');
    expect(r).toEqual(['你', '好', '世', '界']);
  });

  it('splits punctuation from words', () => {
    const r = preTokenize('hello.world');
    expect(r).toEqual(['hello', '.', 'world']);
  });

  it('handles empty input', () => {
    const r = preTokenize('');
    expect(r).toEqual([]);
  });
});

describe('isPunctuation', () => {
  it('identifies periods as punctuation', () => {
    expect(isPunctuation('.')).toBe(true);
  });

  it('identifies commas as punctuation', () => {
    expect(isPunctuation(',')).toBe(true);
  });

  it('range 33-47: exclamation mark', () => {
    expect(isPunctuation('!')).toBe(true);
  });

  it('range 58-64: colon', () => {
    expect(isPunctuation(':')).toBe(true);
  });

  it('range 91-96: backslash', () => {
    expect(isPunctuation('\\')).toBe(true);
  });

  it('range 123-126: tilde', () => {
    expect(isPunctuation('~')).toBe(true);
  });

  it('returns false for letters', () => {
    expect(isPunctuation('a')).toBe(false);
    expect(isPunctuation('Z')).toBe(false);
  });

  it('returns false for digits', () => {
    expect(isPunctuation('5')).toBe(false);
  });
});
