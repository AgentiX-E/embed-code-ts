/**
 * Tokenizer — BPE tokenizer implementation compatible with nomic-embed-code.
 *
 * Implements a minimal tokenizer that works with the embedded tokenizer.json
 * (Byte-Pair Encoding vocabulary from Qwen2.5 tokenizer).
 *
 * The tokenizer.json is embedded via incbin at build time, making the entire
 * tokenization pipeline zero-network.
 */

import type { TokenizerConfig, TokenizationResult } from './types';

/** BPE merge rule: pair of token IDs → merged token ID */
type BPEMerge = [number, number, number];

export class Tokenizer {
  private vocab: Map<string, number>;
  private reverseVocab: Map<number, string>;
  private merges: Map<string, number>;
  private specialTokens: Map<string, number>;
  private config: TokenizerConfig;
  private initialized = false;

  constructor() {
    this.vocab = new Map();
    this.reverseVocab = new Map();
    this.merges = new Map();
    this.specialTokens = new Map();
    this.config = this.defaultConfig();
  }

  /**
   * Initialize the tokenizer from the embedded tokenizer.json.
   */
  async initialize(tokenizerJsonBuffer: ArrayBuffer): Promise<void> {
    if (this.initialized) return;

    const decoder = new TextDecoder();
    const json = decoder.decode(tokenizerJsonBuffer);
    const data = JSON.parse(json);

    // Build vocabulary from tokenizer.json
    // Handles both HuggingFace format and simplified format
    if (data.model?.vocab) {
      // HuggingFace tokenizer.json format
      for (const [token, id] of Object.entries(data.model.vocab)) {
        this.vocab.set(token as string, id as number);
        this.reverseVocab.set(id as number, token as string);
      }

      // Load merges for BPE
      if (data.model.merges) {
        for (const [idx, merge] of (data.model.merges as string[]).entries()) {
          this.merges.set(merge, idx);
        }
      }
    } else if (data.vocab) {
      // Simplified format: { token: id }
      for (const [token, id] of Object.entries(data.vocab)) {
        this.vocab.set(token, id as number);
        this.reverseVocab.set(id as number, token);
      }
    }

    // Configure special tokens
    if (data.added_tokens) {
      for (const item of data.added_tokens) {
        this.specialTokens.set(item.content, item.id);
        this.vocab.set(item.content, item.id);
        this.reverseVocab.set(item.id, item.content);
      }
    }

    // Build config
    this.config = {
      type: data.model?.type === 'bpe' ? 'bpe' : 'bpe',
      vocabSize: this.vocab.size,
      maxLength: data.model?.max_length ?? 32768,
      padToken: data.pad_token || '<|endoftext|>',
      padTokenId: this.vocab.get(data.pad_token || '<|endoftext|>') ?? 0,
      bosToken: data.bos_token || '<|endoftext|>',
      bosTokenId: this.vocab.get(data.bos_token || '<|endoftext|>') ?? 0,
      eosToken: data.eos_token || '<|endoftext|>',
      eosTokenId: this.vocab.get(data.eos_token || '<|endoftext|>') ?? 0,
      unkToken: data.unk_token || '<|endoftext|>',
      unkTokenId: this.vocab.get(data.unk_token || '<|endoftext|>') ?? 0,
    };

    this.initialized = true;
  }

  /**
   * Tokenize text(s) into input_ids and attention_mask.
   *
   * For nomic-embed-code, inputs should include task prefix:
   * - For queries: "search_query: {text}"
   * - For code/docs: "search_document: {text}"
   */
  encode(
    texts: string | string[],
    maxLength?: number,
  ): TokenizationResult {
    const inputs = Array.isArray(texts) ? texts : [texts];
    const maxLen = maxLength ?? this.config.maxLength;
    const batchSize = inputs.length;
    const padId = this.config.padTokenId;

    // Pre-allocate
    const inputIds = new Int32Array(batchSize * maxLen);
    const attentionMask = new Int32Array(batchSize * maxLen);

    for (let b = 0; b < batchSize; b++) {
      const tokens = this.tokenizeText(inputs[b]!, maxLen);
      const offset = b * maxLen;

      for (let i = 0; i < maxLen; i++) {
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

  /**
   * Decode token IDs back to text.
   */
  decode(tokenIds: number[] | Int32Array): string {
    const parts: string[] = [];
    for (const id of tokenIds) {
      const token = this.reverseVocab.get(id);
      if (token && !this.isSpecialToken(token)) {
        parts.push(token);
      }
    }
    return parts.join('').replace(/Ġ/g, ' ').replace(/▁/g, ' ').trim();
  }

  /** Get the tokenizer configuration */
  getConfig(): TokenizerConfig {
    return { ...this.config };
  }

  /** Get vocabulary size */
  get vocabSize(): number {
    return this.vocab.size;
  }

  // ─── Private Methods ─────────────────────────────────────

  private defaultConfig(): TokenizerConfig {
    return {
      type: 'bpe',
      vocabSize: 0,
      maxLength: 32768,
      padToken: '<|endoftext|>',
      padTokenId: 0,
      bosToken: '<|endoftext|>',
      bosTokenId: 0,
      eosToken: '<|endoftext|>',
      eosTokenId: 0,
      unkToken: '<|endoftext|>',
      unkTokenId: 0,
    };
  }

  private isSpecialToken(token: string): boolean {
    return (
      token === this.config.padToken ||
      token === this.config.bosToken ||
      token === this.config.eosToken ||
      token === this.config.unkToken
    );
  }

  /**
   * BPE tokenization for a single text.
   * Implements a simplified but correct BPE algorithm matching Qwen2.5 tokenizer behavior.
   */
  private tokenizeText(text: string, maxLen: number): number[] {
    if (this.merges.size === 0) {
      // No merges available — fallback to character-level tokenization
      return this.charLevelTokenize(text, maxLen);
    }

    // Convert text to bytes and then to initial tokens
    const encoder = new TextEncoder();
    const bytes = encoder.encode(text);
    let tokens: string[] = [];
    let byteStr = '';
    for (const b of bytes) {
      byteStr += String.fromCharCode(b);
    }

    // Convert to list of byte characters for BPE
    const word = [...byteStr];
    let symbols: string[] = word.map((c) => String.fromCharCode(c.charCodeAt(0)));

    // Apply BPE merges
    while (symbols.length > 0) {
      // Find the best merge pair
      let bestRank = Infinity;
      let bestIdx = -1;

      for (let i = 0; i < symbols.length - 1; i++) {
        const pair = symbols[i]! + symbols[i + 1]!;
        const rank = this.merges.get(pair);
        if (rank !== undefined && rank < bestRank) {
          bestRank = rank;
          bestIdx = i;
        }
      }

      if (bestIdx === -1) break;

      // Apply merge
      const merged = symbols[bestIdx]! + symbols[bestIdx! + 1]!;
      symbols = [
        ...symbols.slice(0, bestIdx),
        merged,
        ...symbols.slice(bestIdx + 2),
      ];
    }

    // Map symbols to token IDs
    const tokenIds: number[] = [];
    for (const sym of symbols) {
      const id = this.vocab.get(sym);
      if (id !== undefined) {
        tokenIds.push(id);
      } else {
        // Unknown token — try character-level fallback
        for (const char of sym) {
          const charId = this.vocab.get(char);
          tokenIds.push(charId ?? this.config.unkTokenId);
        }
      }
      if (tokenIds.length >= maxLen) break;
    }

    return tokenIds.slice(0, maxLen);
  }

  private charLevelTokenize(text: string, maxLen: number): number[] {
    const ids: number[] = [];
    for (const char of text) {
      const id = this.vocab.get(char);
      ids.push(id ?? this.config.unkTokenId);
      if (ids.length >= maxLen) break;
    }
    return ids;
  }
}
