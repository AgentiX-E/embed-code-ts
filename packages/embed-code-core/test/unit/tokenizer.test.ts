/**
 * Unit tests for the tokenizer.
 */
import { describe, it, expect, beforeAll } from 'vitest';
import { Tokenizer } from '../../src/tokenizer';

// Build a minimal tokenizer with byte-level vocabulary entries.
// Since BPE starts from individual bytes, each byte char must be in the vocab.
function buildMinimalVocab(): string {
  const vocab: Record<string, number> = {};
  // Add byte-level entries for common ASCII chars
  const chars = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ';
  for (let i = 0; i < chars.length; i++) {
    vocab[chars[i]!] = i + 1;
  }
  vocab['<|endoftext|>'] = 0; // pad token

  return JSON.stringify({
    model: {
      type: 'bpe',
      vocab,
      merges: [], // No merges → stays at byte level
    },
    pad_token: '<|endoftext|>',
  });
}

describe('Tokenizer', () => {
  let tokenizer: Tokenizer;

  beforeAll(() => {
    tokenizer = new Tokenizer();
    tokenizer.loadFromBuffer(new TextEncoder().encode(buildMinimalVocab()).buffer);
  });

  it('loads vocabulary from tokenizer JSON', () => {
    expect(tokenizer.vocabSize).toBeGreaterThan(10);
  });

  it('encodes text to token IDs (byte-level)', () => {
    const result = tokenizer.encode(['hello'], 10);
    // 'hello' → 5 bytes → 5 token IDs
    expect(result.attentionMask[0]).toBe(1);
    expect(result.attentionMask[4]).toBe(1);
    // Padding starts after 5 tokens
    expect(result.attentionMask[5]).toBe(0);
  });

  it('pads shorter sequences with pad_token_id', () => {
    const result = tokenizer.encode(['hi'], 10);
    // 'hi' → 2 tokens
    expect(result.attentionMask[0]).toBe(1);
    expect(result.attentionMask[1]).toBe(1);
    expect(result.attentionMask[2]).toBe(0); // padding
    expect(result.inputIds[2]).toBe(0); // pad_token_id
  });

  it('encodes batch of texts', () => {
    const result = tokenizer.encode(['hi', 'ok'], 5);
    // Batch 0: 'hi' (2 bytes)
    expect(result.attentionMask[0]).toBe(1);
    expect(result.attentionMask[1]).toBe(1);
    // Batch 1: 'ok' (2 bytes) — offset by maxLength=5
    expect(result.attentionMask[5]).toBe(1);
    expect(result.attentionMask[6]).toBe(1);
  });

  it('truncates sequences exceeding max length', () => {
    const longText = 'abcdefghijklmnop'; // 16 chars → many more after byte expansion
    const result = tokenizer.encode([longText], 3);
    expect(result.attentionMask[0]).toBe(1);
    expect(result.attentionMask[1]).toBe(1);
    expect(result.attentionMask[2]).toBe(1);
  });

  it('returns correct array lengths', () => {
    const maxLen = 8;
    const result = tokenizer.encode(['test'], maxLen);
    expect(result.inputIds.length).toBe(maxLen);
    expect(result.attentionMask.length).toBe(maxLen);
  });
});
