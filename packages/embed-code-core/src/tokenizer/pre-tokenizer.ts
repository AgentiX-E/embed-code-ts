/**
 * Pre-tokenizer for BERT WordPiece.
 *
 * BertPreTokenizer: splits input text into word-like units by
 * separating on whitespace and punctuation. This handles
 * Unicode normalization (NFKD), lowercase, and Chinese character
 * splitting (per the tokenizer.json BertNormalizer config).
 *
 * The output is a list of strings ready for WordPiece subword
 * decomposition.
 */

/**
 * Pre-tokenize text into word units (BertPreTokenizer).
 *
 * Steps:
 * 1. Lowercase (BertNormalizer with lowercase=true)
 * 2. Split on whitespace
 * 3. Split punctuation from word boundaries
 * 4. Handle Chinese/CJK characters (split each as own token)
 */
export function preTokenize(text: string): string[] {
  // 1. Unicode NFKD normalization (strips accents: café → cafe)
  let normalized = text.normalize('NFKD').replace(/[\u0300-\u036f]/g, '');

  // 2. Lowercase (BertNormalizer)
  normalized = normalized.toLowerCase();

  // 2. Handle Chinese characters (split each as separate token)
  // CJK Unified Ideographs range: U+4E00–U+9FFF
  // This mirrors BertNormalizer with handle_chinese_chars=true
  const cjkSplit: string[] = [];
  for (const ch of normalized) {
    const code = ch.codePointAt(0)!;
    if (
      (code >= 0x4e00 && code <= 0x9fff) || // CJK Unified
      (code >= 0x3400 && code <= 0x4dbf) || // CJK Extension A
      (code >= 0xf900 && code <= 0xfaff) || // CJK Compatibility
      (code >= 0x30a0 && code <= 0x30ff) || // Katakana
      (code >= 0x3040 && code <= 0x309f) || // Hiragana
      (code >= 0xac00 && code <= 0xd7af) || // Hangul Syllables
      (code >= 0x2f800 && code <= 0x2fa1f) // CJK Compatibility Supplement
    ) {
      cjkSplit.push(' ' + ch + ' ');
    } else {
      cjkSplit.push(ch);
    }
  }
  normalized = cjkSplit.join('');

  // 3. Split on whitespace
  const whitespaceTokens = normalized.split(/[\s]+/).filter((t) => t.length > 0);

  // 4. Split punctuation from word boundaries
  // BERT pre-tokenizer splits ALL punctuation (not just word-boundary)
  const result: string[] = [];
  for (const token of whitespaceTokens) {
    // Split punctuation: anything non-alphanumeric except ## (which is handled later)
    const subTokens = token
      .split(/([!"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~])/)
      .filter((t) => t.length > 0);
    for (const st of subTokens) result.push(st);
  }

  return result;
}

/**
 * Check if a character is a BERT punctuation character.
 */
export function isPunctuation(ch: string): boolean {
  const code = ch.charCodeAt(0);
  return (
    (code >= 33 && code <= 47) ||
    (code >= 58 && code <= 64) ||
    (code >= 91 && code <= 96) ||
    (code >= 123 && code <= 126)
  );
}
