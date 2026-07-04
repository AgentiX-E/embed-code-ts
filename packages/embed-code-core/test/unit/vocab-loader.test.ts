/**
 * Unit tests for vocab-loader.
 */
import { describe, it, expect } from 'vitest';
import { loadVocab } from '../../src/tokenizer/vocab-loader';

describe('loadVocab', () => {
  it('loads basic vocab with special tokens', () => {
    const json = {
      model: {
        vocab: {
          '[PAD]': 0,
          '[UNK]': 1,
          '[CLS]': 2,
          '[SEP]': 3,
          '[MASK]': 4,
          hello: 5,
          world: 6,
          '##ing': 7,
        },
      },
      added_tokens: [
        { content: '[PAD]', id: 0, special: true },
        { content: '[UNK]', id: 1, special: true },
      ],
    };
    const info = loadVocab(json);
    expect(info.padTokenId).toBe(0);
    expect(info.clsTokenId).toBe(2);
    expect(info.sepTokenId).toBe(3);
    expect(info.unkTokenId).toBe(1);
    expect(info.isContinuation(7)).toBe(true);
    expect(info.isContinuation(5)).toBe(false);
  });

  it('throws when critical special tokens are missing', () => {
    const json = { model: { vocab: { hello: 5 } } };
    expect(() => loadVocab(json)).toThrow(/Missing special token/i);
  });

  it('recognizes continuation subwords from added_tokens', () => {
    const json = {
      model: {
        vocab: {
          '[PAD]': 0,
          '[UNK]': 1,
          '[CLS]': 2,
          '[SEP]': 3,
          '[MASK]': 4,
          hello: 5,
          world: 6,
        },
      },
      added_tokens: [
        { content: '##ing', id: 7, special: false },
        { content: '##ly', id: 8, special: false },
      ],
    };
    const info = loadVocab(json);
    expect(info.isContinuation(7)).toBe(true);
    expect(info.isContinuation(8)).toBe(true);
    expect(info.isContinuation(5)).toBe(false);
  });

  it('throws on empty vocabulary without special tokens', () => {
    expect(() => loadVocab({ model: { vocab: {} } })).toThrow(/Missing special token/i);
  });
});
