/**
 * Unit tests for vocab-loader.
 */
import { describe, it, expect } from 'vitest';
import { loadVocab } from '../../src/tokenizer/vocab-loader';

describe('loadVocab', () => {
  it('loads basic vocab with special tokens', () => {
    const json = {
      model: {
        vocab: { '[PAD]': 0, '[UNK]': 1, '[CLS]': 2, '[SEP]': 3, hello: 4, world: 5, '##ing': 6 },
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
    expect(info.isContinuation(6)).toBe(true);
    expect(info.isContinuation(4)).toBe(false);
  });

  it('handles missing special tokens gracefully', () => {
    const json = { model: { vocab: { hello: 5 } } };
    const info = loadVocab(json);
    expect(info.padTokenId).toBe(0); // default fallback
  });
});
