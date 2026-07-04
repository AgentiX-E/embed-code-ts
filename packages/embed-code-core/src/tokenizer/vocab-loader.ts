/**
 * VocabLoader — loads a WordPiece vocabulary from tokenizer.json.
 *
 * The vocab maps token strings → integer IDs.
 * Special tokens [CLS], [SEP], [UNK], [PAD], [MASK] are extracted.
 * Provides O(1) lookup via Map<string, number>.
 */

export interface VocabInfo {
  /** Map from token string to ID */
  readonly vocab: Map<string, number>;
  /** Map from ID to token string */
  readonly reverseVocab: Map<number, string>;
  /** Vocabulary size */
  readonly size: number;
  /** Special token IDs */
  readonly padTokenId: number;
  readonly clsTokenId: number;
  readonly sepTokenId: number;
  readonly unkTokenId: number;
  readonly maskTokenId: number;
  /** Whether a token at a given ID is a continuation subword (starts with ##) */
  isContinuation(id: number): boolean;
}

/**
 * Load vocab from a parsed tokenizer.json object.
 *
 * Expected structure (HuggingFace tokenizer.json):
 *   model.vocab: { "token": id, ... }
 *   added_tokens: [{ "content": "...", "id": N, "special": true }, ...]
 *   unk_token: "[UNK]", pad_token: "[PAD]", etc.
 */
export function loadVocab(tokenizerJson: Record<string, any>): VocabInfo {
  const vocab = new Map<string, number>();
  const reverseVocab = new Map<number, string>();
  const continuationSet = new Set<number>();

  // Load model.vocab
  const modelVocab = tokenizerJson.model?.vocab as Record<string, number> | undefined;
  if (modelVocab) {
    for (const [token, id] of Object.entries(modelVocab)) {
      vocab.set(token, id as number);
      reverseVocab.set(id as number, token);
      if (token.startsWith('##')) {
        continuationSet.add(id as number);
      }
    }
  }

  // Overlay added_tokens (they take precedence)
  const addedTokens = tokenizerJson.added_tokens as
    | Array<{
        content: string;
        id: number;
        special?: boolean;
      }>
    | undefined;
  if (addedTokens) {
    for (const t of addedTokens) {
      vocab.set(t.content, t.id);
      reverseVocab.set(t.id, t.content);
      if (t.content.startsWith('##')) {
        continuationSet.add(t.id);
      }
    }
  }

  if (!modelVocab && (!addedTokens || addedTokens.length === 0)) {
    throw new Error('Vocabulary is empty: no model.vocab and no added_tokens found');
  }

  // Resolve special token IDs from the vocab map (most reliable)
  const findId = (label: string, ...candidates: string[]): number => {
    for (const c of candidates) {
      const id = vocab.get(c);
      if (id !== undefined) return id;
    }
    throw new Error(
      `Missing special token in vocabulary: ${label}. Candidates: [${candidates.join(', ')}]`,
    );
  };

  return {
    vocab,
    reverseVocab,
    size: vocab.size,
    padTokenId: findId('pad', '[PAD]', '<|endoftext|>'),
    clsTokenId: findId('cls', '[CLS]'),
    sepTokenId: findId('sep', '[SEP]'),
    unkTokenId: findId('unk', '[UNK]'),
    maskTokenId: findId('mask', '[MASK]'),
    isContinuation: (id: number) => continuationSet.has(id),
  };
}
