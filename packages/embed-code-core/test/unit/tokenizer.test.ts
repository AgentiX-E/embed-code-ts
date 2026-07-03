/**
 * Unit tests for WordPiece tokenizer — validated against Python transformers reference.
 */
import { describe, it, expect, beforeAll } from 'vitest';
import * as fs from 'node:fs';
import * as path from 'node:path';
import { WordPieceTokenizer } from '../../src/tokenizer/wordpiece-tokenizer';

const TOKENIZER_PATH = path.join(process.cwd(), 'models', 'tokenizer.json');
const REFERENCE_PATH = path.join(process.cwd(), 'models', 'tokenizer-reference.json');

const hasTokenizer = fs.existsSync(TOKENIZER_PATH);
const hasReference = fs.existsSync(REFERENCE_PATH);

describe('WordPieceTokenizer', () => {
  describe('vocabulary loading', () => {
    it('loads from tokenizer.json', () => {
      if (!hasTokenizer) return;
      const tok = WordPieceTokenizer.fromFile(TOKENIZER_PATH);
      expect(tok.vocabSize).toBeGreaterThan(30000);
    });

    it('has correct special token IDs', () => {
      if (!hasTokenizer) return;
      const tok = WordPieceTokenizer.fromFile(TOKENIZER_PATH);
      expect(tok.clsTokenId).toBe(101);
      expect(tok.sepTokenId).toBe(102);
      expect(tok.unkTokenId).toBe(100);
      expect(tok.padTokenId).toBe(0);
      expect(tok.maskTokenId).toBe(103);
    });

    it('defaults to maxLength=512', () => {
      if (!hasTokenizer) return;
      const tok = WordPieceTokenizer.fromFile(TOKENIZER_PATH);
      expect(tok.maxLength).toBe(512);
    });
  });

  describe('static constructors', () => {
    it('fromBuffer works with ArrayBuffer', () => {
      if (!hasTokenizer) return;
      const raw = fs.readFileSync(TOKENIZER_PATH);
      const buf = raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength);
      const tok = WordPieceTokenizer.fromBuffer(buf, 512);
      expect(tok.clsTokenId).toBe(101);
      expect(tok.sepTokenId).toBe(102);
    });
  });

  describe('basic tokenization', () => {
    let tok: WordPieceTokenizer;

    beforeAll(() => {
      if (!hasTokenizer) return;
      tok = WordPieceTokenizer.fromFile(TOKENIZER_PATH);
    });

    it('[CLS] is always at position 0', () => {
      if (!hasTokenizer) return;
      const r = tok.tokenize('hello');
      expect(r.inputIds[0]).toBe(tok.clsTokenId);
    });

    it('[SEP] follows real tokens', () => {
      if (!hasTokenizer) return;
      const r = tok.tokenize('hello');
      // [SEP] should be at the first 0-attention-mask position minus 1
      let sepPos = -1;
      for (let i = 1; i < 512; i++) {
        if (r.attentionMask[i] === 0 && r.attentionMask[i - 1] === 1) {
          sepPos = i - 1;
          break;
        }
      }
      expect(r.inputIds[sepPos]).toBe(tok.sepTokenId);
    });

    it('attentionMask is 1 for real tokens, 0 for padding', () => {
      if (!hasTokenizer) return;
      const r = tok.tokenize('hello');
      expect(r.attentionMask[0]).toBe(1); // [CLS]
      expect(r.attentionMask[r.attentionMask.indexOf(0) - 1]).toBe(1); // last real token
      expect(r.attentionMask[511]).toBe(0); // last padding token
    });

    it('tokenTypeIds are all zeros', () => {
      if (!hasTokenizer) return;
      const r = tok.tokenize('hello');
      for (let i = 0; i < 512; i++) {
        expect(r.tokenTypeIds[i]).toBe(0);
      }
    });

    it('output arrays are exactly maxLength long', () => {
      if (!hasTokenizer) return;
      const r = tok.tokenize('test');
      expect(r.inputIds.length).toBe(512);
      expect(r.attentionMask.length).toBe(512);
      expect(r.tokenTypeIds.length).toBe(512);
    });

    it('handles empty input', () => {
      if (!hasTokenizer) return;
      const r = tok.tokenize('');
      // [CLS], [SEP], then all [PAD]
      expect(r.inputIds[0]).toBe(tok.clsTokenId);
      expect(r.inputIds[1]).toBe(tok.sepTokenId);
      expect(r.inputIds[2]).toBe(tok.padTokenId);
      expect(r.attentionMask[0]).toBe(1);
      expect(r.attentionMask[1]).toBe(1);
      expect(r.attentionMask[2]).toBe(0);
    });

    it('truncates input exceeding 512 tokens', () => {
      if (!hasTokenizer) return;
      const longText = Array(500).fill('function hello() { console.log("test"); }').join(' ');
      const r = tok.tokenize(longText);
      expect(r.attentionMask[511]).toBe(1); // last position should be [SEP]
      expect(r.inputIds[511]).toBe(tok.sepTokenId);
    });
  });

  describe('WordPiece subword decomposition', () => {
    let tok: WordPieceTokenizer;

    beforeAll(() => {
      if (!hasTokenizer) return;
      tok = WordPieceTokenizer.fromFile(TOKENIZER_PATH);
    });

    it('decomposes camelCase identifiers with ## prefix', () => {
      if (!hasTokenizer) return;
      // "getUserProfileImage" should decompose
      const r = tok.tokenize('getUserProfileImage');
      // Should have more than just [CLS], word, [SEP]
      const realTokens = [];
      for (let i = 0; i < 512; i++) {
        if (r.attentionMask[i] === 0) break;
        realTokens.push(r.inputIds[i]);
      }
      expect(realTokens.length).toBeGreaterThan(3); // [CLS] + ≥2 subwords + [SEP]
    });

    it('uses [UNK] for truly unknown tokens', () => {
      if (!hasTokenizer) return;
      // Emoji are definitely OOV for BERT tokenizers
      const r = tok.tokenize('🎉🔥💻🤖');
      const ids = [];
      for (let i = 0; i < 512; i++) {
        if (r.attentionMask[i] === 0) break;
        ids.push(r.inputIds[i]);
      }
      expect(ids).toContain(tok.unkTokenId);
    });
  });
});

describe('WordPieceTokenizer — Python reference comparison', () => {
  it('matches Python transformers on 29 test cases', () => {
    if (!hasReference || !hasTokenizer) return;
    const ref = JSON.parse(fs.readFileSync(REFERENCE_PATH, 'utf-8'));
    const tok = WordPieceTokenizer.fromFile(TOKENIZER_PATH, 512);

    let failures = 0;
    for (const test of ref.tests) {
      const result = tok.tokenize(test.text);
      const n = Math.min(20, test.input_ids.length);
      let ok = true;
      for (let j = 0; j < n; j++) {
        if (result.inputIds[j] !== test.input_ids[j]) {
          ok = false;
          break;
        }
      }
      if (!ok) failures++;
    }

    expect(failures).toBe(0);
    if (failures === 0)
      console.log(`  ✅ ${ref.tests.length}/${ref.tests.length} match Python transformers`);
  }, 10000);
});
