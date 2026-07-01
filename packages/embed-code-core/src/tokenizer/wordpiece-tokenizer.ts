/**
 * WordPiece Tokenizer — BERT-standard implementation.
 *
 * Algorithm:
 * 1. Pre-tokenize via BertPreTokenizer (whitespace + punctuation split)
 * 2. For each word, apply WordPiece subword decomposition:
 *    - Find longest matching subword in vocab
 *    - For non-first subwords, try with "##" prefix
 *    - If no match, use [UNK]
 * 3. Post-process: inject [CLS] at start, [SEP] at end
 * 4. Pad/truncate to max_length
 * 5. Return inputIds + attentionMask + tokenTypeIds
 *
 * Matches Python transformers.AutoTokenizer output.
 */

import { loadVocab, type VocabInfo } from './vocab-loader';
import { preTokenize } from './pre-tokenizer';

export interface TokenizedInput {
  inputIds: Int32Array; // [CLS] + tokens + [SEP] + [PAD]*, length = maxLength
  attentionMask: Int32Array; // 1 for real, 0 for padding
  tokenTypeIds: Int32Array; // all 0 (single segment)
}

export class WordPieceTokenizer {
  private readonly vocab: VocabInfo;
  readonly maxLength: number;

  constructor(vocab: VocabInfo, maxLength: number = 512) {
    this.vocab = vocab;
    this.maxLength = maxLength;
  }

  /** Load from a parsed tokenizer.json object */
  static fromJSON(json: Record<string, any>, maxLength: number = 512): WordPieceTokenizer {
    return new WordPieceTokenizer(loadVocab(json), maxLength);
  }

  /** Load from a tokenizer.json file path (Node.js) */
  static fromFile(path: string, maxLength: number = 512): WordPieceTokenizer {
    const fs = require('node:fs');
    const json = JSON.parse(fs.readFileSync(path, 'utf-8'));
    return WordPieceTokenizer.fromJSON(json, maxLength);
  }

  /** Load from a buffer (embeddable scenarios) */
  static fromBuffer(buf: ArrayBuffer, maxLength: number = 512): WordPieceTokenizer {
    const decoder = new TextDecoder();
    const json = JSON.parse(decoder.decode(buf));
    return WordPieceTokenizer.fromJSON(json, maxLength);
  }

  get padTokenId(): number {
    return this.vocab.padTokenId;
  }
  get clsTokenId(): number {
    return this.vocab.clsTokenId;
  }
  get sepTokenId(): number {
    return this.vocab.sepTokenId;
  }
  get unkTokenId(): number {
    return this.vocab.unkTokenId;
  }
  get maskTokenId(): number {
    return this.vocab.maskTokenId;
  }
  get vocabSize(): number {
    return this.vocab.size;
  }

  /**
   * Tokenize a single text.
   * Returns inputIds, attentionMask, tokenTypeIds as Int32Array(length=maxLength).
   */
  tokenize(text: string): TokenizedInput {
    return this.tokenizeBatch([text], this.maxLength);
  }

  /**
   * Tokenize a batch of texts with padding to maxLength.
   */
  tokenizeBatch(texts: string[], maxLength?: number): TokenizedInput {
    const mLen = maxLength ?? this.maxLength;
    const B = texts.length;
    const inputIds = new Int32Array(B * mLen);
    const attentionMask = new Int32Array(B * mLen);
    const tokenTypeIds = new Int32Array(B * mLen);

    for (let b = 0; b < B; b++) {
      const tokens = this._tokenizeSingle(texts[b]!, mLen);
      const off = b * mLen;
      for (let i = 0; i < mLen; i++) {
        if (i < tokens.length) {
          inputIds[off + i] = tokens[i]!;
          attentionMask[off + i] = 1;
        } else {
          inputIds[off + i] = this.vocab.padTokenId;
          attentionMask[off + i] = 0;
        }
        tokenTypeIds[off + i] = 0; // single segment
      }
    }

    return { inputIds, attentionMask, tokenTypeIds };
  }

  // ─── Private ──────────────────────────────────────────────

  /**
   * Tokenize a single text: [CLS] + wordpieces + [SEP], truncated to maxLength.
   */
  private _tokenizeSingle(text: string, maxLength: number): number[] {
    // 1. Pre-tokenize
    const words = preTokenize(text);

    // 2. WordPiece decomposition for each word
    const subwords: number[] = [];

    for (const word of words) {
      const ids = this._wordpiece(word);
      for (const id of ids) {
        if (subwords.length >= maxLength - 2) break; // leave room for [CLS] and [SEP]
        subwords.push(id);
      }
      if (subwords.length >= maxLength - 2) break;
    }

    // 3. Post-process: [CLS] + subwords + [SEP]
    const result: number[] = [this.vocab.clsTokenId];
    for (const id of subwords) {
      if (result.length >= maxLength - 1) break;
      result.push(id);
    }
    result.push(this.vocab.sepTokenId);

    // Truncate if too long (shouldn't happen since we limited above)
    return result.slice(0, maxLength);
  }

  /**
   * WordPiece decomposition for a single pre-tokenized word.
   *
   * Greedy longest-match from left to right.
   * First subword: match directly in vocab.
   * Subsequent subwords: prepend "##" before lookup.
   * If no match at all: use [UNK].
   */
  private _wordpiece(word: string): number[] {
    const ids: number[] = [];
    const { vocab, isContinuation } = this.vocab;

    // If word is unreasonably long or all-same-char, use [UNK]
    // (matches Python tokenizer's max_input_chars_per_word behavior)
    if (word.length > 100) {
      return [this.vocab.unkTokenId];
    }

    // Try the word as-is first
    const directId = vocab.get(word);
    if (directId !== undefined && !isContinuation(directId)) {
      ids.push(directId);
      return ids;
    }

    // WordPiece decomposition
    let start = 0;
    let isFirst = true;

    while (start < word.length) {
      let end = word.length;
      let foundId: number | undefined;

      while (start < end) {
        const sub = isFirst ? word.slice(start, end) : '##' + word.slice(start, end);
        const id = vocab.get(sub);
        if (id !== undefined && (isFirst || isContinuation(id))) {
          foundId = id;
          break;
        }
        end--;
      }

      if (foundId !== undefined) {
        ids.push(foundId);
        start = end;
        isFirst = false;
      } else {
        // No match found — use [UNK]
        ids.push(this.vocab.unkTokenId);
        break; // skip rest of the word
      }
    }

    return ids;
  }
}
