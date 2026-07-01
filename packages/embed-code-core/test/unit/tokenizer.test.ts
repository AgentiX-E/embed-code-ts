/**
 * Unit tests for the tokenizer.
 *
 * Tests both byte-level (no merges) and BPE merge-aware tokenization
 * using realistic vocabulary structures derived from Qwen2 tokenizer patterns.
 */
import { describe, it, expect, beforeAll, beforeEach } from 'vitest';
import { Tokenizer } from '../../src/tokenizer';

// ─── Fixture: byte-level vocabulary (no merges) ─────────────────────────

function buildByteLevelVocab(): string {
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
      merges: [],
    },
    pad_token: '<|endoftext|>',
  });
}

// ─── Fixture: BPE merge-aware vocabulary ────────────────────────────────
//
// Simulates a subset of a real BPE tokenizer (Qwen2-style).
// Individual bytes map to IDs 1-256, merged tokens get higher IDs.

function buildBpeMergeVocab(): string {
  const vocab: Record<string, number> = {};

  // Individual byte characters (IDs 1-256)
  for (let i = 0; i < 256; i++) {
    vocab[String.fromCharCode(i)] = i + 1;
  }

  // Multi-byte merged tokens (higher IDs)
  vocab['he'] = 300;
  vocab['ll'] = 301;
  vocab['lo'] = 302;
  vocab['th'] = 303;
  vocab['er'] = 304;
  vocab['in'] = 305;
  vocab['an'] = 306;
  vocab['on'] = 307;
  vocab['re'] = 308;
  vocab['de'] = 309;
  vocab['def'] = 310;
  vocab['return'] = 311;
  vocab['function'] = 312;

  vocab['<|endoftext|>'] = 0; // pad token

  // Merges in priority order (lower index = higher priority)
  const merges: string[] = [
    'd e', // → 309
    'e f', // → 310 (in combination with d e + f → def)
    'd ef', // → 310
    't h', // → 303
    'e r', // → 304
    'h e', // → 300
    'l l', // → 301
    'l o', // → 302
    'i n', // → 305
    'a n', // → 306
    'o n', // → 307
    'r e', // → 308
  ];

  return JSON.stringify({
    model: {
      type: 'bpe',
      vocab,
      merges,
    },
    pad_token: '<|endoftext|>',
    eos_token: '<|endoftext|>',
    unk_token: null,
  });
}

describe('Tokenizer — byte-level (no merges)', () => {
  let tokenizer: Tokenizer;

  beforeAll(() => {
    tokenizer = new Tokenizer();
    tokenizer.loadFromBuffer(new TextEncoder().encode(buildByteLevelVocab()).buffer);
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
    const longText = 'abcdefghijklmnop';
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

  it('throws when encoding without initialization', () => {
    const uninit = new Tokenizer();
    expect(() => uninit.encode(['hello'], 10)).toThrow();
  });

  it('has correct accessor defaults before loading', () => {
    const uninit = new Tokenizer();
    expect(uninit.vocabSize).toBe(0);
    expect(uninit.maxLength).toBe(32768);
    expect(uninit.padTokenId).toBe(0);
  });
});

// ─── BPE Merge-Aware Tests ───────────────────────────────────────────────

describe('Tokenizer — BPE merge-aware', () => {
  let tokenizer: Tokenizer;

  beforeEach(() => {
    tokenizer = new Tokenizer();
    tokenizer.loadFromBuffer(new TextEncoder().encode(buildBpeMergeVocab()).buffer);
  });

  it('merges common bigrams according to merge priority', () => {
    // 'hello' bytes: h(104) e(101) l(108) l(108) o(111)
    // Merges: 'he'(300) then 'll'(301), remaining 'o' → [300, 301, o]
    const result = tokenizer.encode(['hello'], 10);

    // Should merge 'he' → 300, 'll' → 301, 'o' → 112 (byte 111+1)
    expect(result.inputIds[0]).toBe(300); // merged 'he'
    expect(result.inputIds[1]).toBe(301); // merged 'll'
    expect(result.inputIds[2]).toBe(112); // 'o' byte
    // Remaining padded
    expect(result.attentionMask[0]).toBe(1);
    expect(result.attentionMask[1]).toBe(1);
    expect(result.attentionMask[2]).toBe(1);
    expect(result.attentionMask[3]).toBe(0);
  });

  it('merges multiple levels (th → e → the pattern)', () => {
    // 'the' bytes: t(116) h(104) e(101)
    // Merge order: 'th'(303) then h(104)+e(101) can't merge because 'h' already merged
    // So 'th' merges first, leaving 'the' → ['th'(303), 'e'(102)]
    const result = tokenizer.encode(['the'], 10);

    expect(result.inputIds[0]).toBe(303); // merged 'th'
    expect(result.inputIds[1]).toBe(102); // 'e' byte
    expect(result.attentionMask[2]).toBe(0);
  });

  it('does not merge when vocab lacks the merged token', () => {
    // 'xy' – neither 'x' nor 'y' have merge pairs in our vocab
    // x=120, y=121 (bytes)
    const result = tokenizer.encode(['xy'], 10);

    expect(result.inputIds[0]).toBe(121); // 'x' byte + 1
    expect(result.inputIds[1]).toBe(122); // 'y' byte + 1
    expect(result.attentionMask[2]).toBe(0);
  });

  it('handles empty string input', () => {
    const result = tokenizer.encode([''], 5);
    // No tokens → all padding
    for (let i = 0; i < 5; i++) {
      expect(result.attentionMask[i]).toBe(0);
      expect(result.inputIds[i]).toBe(0); // pad_token_id
    }
  });

  it('handles unicode characters as bytes', () => {
    // 'é' is U+00E9 → UTF-8 bytes: 0xC3 0xA9 → 195+1=196, 169+1=170
    const result = tokenizer.encode(['\u00E9'], 5);

    expect(result.attentionMask[0]).toBe(1);
    expect(result.attentionMask[1]).toBe(1);
    expect(result.attentionMask[2]).toBe(0);
  });

  it('has correct accessor values after BPE loading', () => {
    expect(tokenizer.vocabSize).toBeGreaterThan(250);
    expect(tokenizer.maxLength).toBe(32768);
    expect(tokenizer.padTokenId).toBe(0);
    expect(tokenizer.eosTokenId).toBe(0); // <|endoftext|> is both pad and eos
  });
});
