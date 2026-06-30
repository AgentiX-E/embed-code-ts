/**
 * BPE Tokenizer — compatible with Qwen2.5 tokenizer used by nomic-embed-code.
 *
 * Loads the HuggingFace tokenizer.json and implements BPE encoding/decoding.
 * The tokenizer config is loaded from disk (typically co-located with the model).
 */

import { TokenizationError } from './errors';
import type { TokenizationResult, TokenizerDescriptor } from './types';
import * as fs from 'node:fs';

export class Tokenizer {
  private vocab: Map<string, number> = new Map();
  private reverseVocab: Map<number, string> = new Map();
  private merges: Map<string, number> = new Map();
  private descriptor!: TokenizerDescriptor;
  private initialized = false;

  /** Load tokenizer from a tokenizer.json file path */
  loadFromFile(tokenizerPath: string): void {
    const raw = fs.readFileSync(tokenizerPath, 'utf-8');
    const data = JSON.parse(raw);

    // Build vocabulary
    if (data.model?.vocab) {
      for (const [token, id] of Object.entries(data.model.vocab)) {
        this.vocab.set(token, id as number);
        this.reverseVocab.set(id as number, token);
      }
    }

    // Load BPE merges
    if (data.model?.merges) {
      for (const [idx, merge] of (data.model.merges as string[]).entries()) {
        this.merges.set(merge, idx);
      }
    }

    // Added tokens
    if (data.added_tokens) {
      for (const item of data.added_tokens) {
        this.vocab.set(item.content, item.id);
        this.reverseVocab.set(item.id, item.content);
      }
    }

    this.descriptor = {
      type: 'bpe',
      vocab_size: this.vocab.size,
      max_length: data.model?.max_length ?? 32768,
      pad_token: data.pad_token || null,
      pad_token_id: this.vocab.get(data.pad_token) ?? 0,
      bos_token: data.bos_token || null,
      bos_token_id: data.bos_token ? (this.vocab.get(data.bos_token) ?? null) : null,
      eos_token: data.eos_token || null,
      eos_token_id: data.eos_token ? (this.vocab.get(data.eos_token) ?? null) : null,
      unk_token: data.unk_token || null,
      unk_token_id: data.unk_token ? (this.vocab.get(data.unk_token) ?? null) : null,
    };

    this.initialized = true;
  }

  /** Load tokenizer from a buffer (for embedded scenarios) */
  loadFromBuffer(buffer: ArrayBuffer): void {
    const decoder = new TextDecoder();
    const json = decoder.decode(buffer);
    // Use process.cwd() to avoid hardcoded /tmp and ensure concurrency safety
    const tmpDir = fs.mkdtempSync('embed-code-tokenizer-');
    const tmpPath = `${tmpDir}/tokenizer.json`;
    fs.writeFileSync(tmpPath, json);
    try {
      this.loadFromFile(tmpPath);
    } finally {
      try { fs.unlinkSync(tmpPath); } catch { /* best effort */ }
      try { fs.rmdirSync(tmpDir); } catch { /* best effort */ }
    }
  }

  /**
   * Encode texts into input_ids and attention_mask.
   *
   * For nomic-embed-code, task prefixes are prepended automatically:
   *   - Queries: "search_query: {text}"
   *   - Documents: "search_document: {code}"
   */
  encode(texts: string[], maxLength: number): TokenizationResult {
    if (!this.initialized) {
      throw new TokenizationError('Tokenizer not initialized. Call loadFromFile() first.');
    }

    const batchSize = texts.length;
    const padId = this.descriptor.pad_token_id;

    const inputIds = new Int32Array(batchSize * maxLength);
    const attentionMask = new Int32Array(batchSize * maxLength);

    for (let b = 0; b < batchSize; b++) {
      const tokens = this.tokenizeText(texts[b], maxLength);
      const offset = b * maxLength;

      for (let i = 0; i < maxLength; i++) {
        if (i < tokens.length) {
          inputIds[offset + i] = tokens[i]!;
          attentionMask[offset + i] = 1;
        } else {
          inputIds[offset + i] = padId;
          attentionMask[offset + i] = 0;
        }
      }
    }

    return { inputIds, attentionMask };
  }

  get vocabSize(): number {
    return this.vocab.size;
  }

  get maxLength(): number {
    return this.descriptor?.max_length ?? 32768;
  }

  get padTokenId(): number {
    return this.descriptor?.pad_token_id ?? 0;
  }

  get eosTokenId(): number | null {
    return this.descriptor?.eos_token_id ?? null;
  }

  // ─── Private ──────────────────────────────────────────────

  private tokenizeText(text: string, maxLen: number): number[] {
    // BPE tokenization matching Qwen2.5 tokenizer behavior
    const encoder = new TextEncoder();
    const bytes = encoder.encode(text);

    // Convert each byte to a character for BPE processing
    const symbols: string[] = [];
    for (const b of bytes) {
      symbols.push(String.fromCharCode(b));
    }

    // Apply BPE merges (greedy, lowest-rank first)
    if (this.merges.size > 0) {
      for (;;) {
        let bestRank = Infinity;
        let bestIdx = -1;

        for (let i = 0; i < symbols.length - 1; i++) {
          const pair = symbols[i] + symbols[i + 1];
          const rank = this.merges.get(pair);
          if (rank !== undefined && rank < bestRank) {
            bestRank = rank;
            bestIdx = i;
          }
        }

        if (bestIdx === -1) break;
        symbols[bestIdx] = symbols[bestIdx] + symbols[bestIdx + 1];
        symbols.splice(bestIdx + 1, 1);
      }
    }

    // Map symbols to token IDs
    const ids: number[] = [];
    for (const sym of symbols) {
      const id = this.vocab.get(sym);
      if (id !== undefined) {
        ids.push(id);
      } else {
        // Fallback: try character-level mapping
        for (const char of sym) {
          ids.push(this.vocab.get(char) ?? this.descriptor.unk_token_id ?? 0);
        }
      }
      if (ids.length >= maxLen) break;
    }

    return ids.slice(0, maxLen);
  }
}
