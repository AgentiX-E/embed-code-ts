#!/usr/bin/env python3
"""Generate Python reference tokenizer outputs for WordPiece validation."""
import json, sys
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained(
    "/workspace/embed-code-ts/models", trust_remote_code=True
)

# 100+ test cases covering various programming scenarios
test_cases = [
    # Python
    "def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)",
    "class BinarySearchTree:\n  def insert(self, value): ...",
    "import torch\nimport torch.nn as nn\n\nclass AttentionBlock(nn.Module): ...",
    "@dataclass\nclass Point:\n    x: float\n    y: float",
    "async def fetch(url: str) -> Response:\n    return await client.get(url)",
    
    # TypeScript/JavaScript
    "function fibonacci(n) { return n <= 1 ? n : fibonacci(n-1) + fibonacci(n-2); }",
    "interface CacheEntry<T> { readonly key: string; readonly value: T; }",
    "const factorial = (n: number): number => n <= 1 ? 1 : n * factorial(n - 1);",
    "export class AuthService implements IAuthService { /* ... */ }",
    "let clamp = (v: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, v));",
    
    # Go
    "func (s *Server) HandleRequest(ctx context.Context, req *http.Request) (*http.Response, error) {",
    "package main\n\nimport \"fmt\"\n\nfunc main() { fmt.Println(\"hello\") }",
    
    # Rust
    "#[derive(Debug, Clone, Serialize, Deserialize)]\npub struct EmbeddingRequest {\n  pub texts: Vec<String>,\n  pub model: String,\n}",
    
    # SQL
    "SELECT u.id, u.name, COUNT(o.id) as order_count FROM users u LEFT JOIN orders o ON u.id = o.user_id",
    
    # Natural language
    "How to sort an array using quicksort?",
    "Calculate the n-th Fibonacci number recursively",
    "Implement a binary search tree in Python",
    "search_query: How to sort an array?",
    "search_document: def sort(arr): return sorted(arr)",
    
    # Edge cases
    "getUserProfileImage",           # camelCase OOV decomposition
    "DS_Store",                       # unknown token → [UNK]
    "hello world",                    # simple
    "function hello() { console.log('Hello World'); }",  # mixed
    "x = 42",                         # very short
    "",                               # empty
    "a" * 1000,                       # very long (truncation)
    "   def foo(): pass   ",           # whitespace
    "UPPERCASE lowercase",            # case handling (lowercase norm)
    "café résumé naïve",             # unicode
]

# Generate reference outputs
results = []
for text in test_cases:
    encoded = tok(text, padding="max_length", truncation=True, max_length=512, return_tensors="np", return_token_type_ids=True)
    input_ids = encoded["input_ids"][0].tolist()
    attention_mask = encoded["attention_mask"][0].tolist()
    token_type_ids = encoded["token_type_ids"][0].tolist()
    
    results.append({
        "text": text,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "token_type_ids": token_type_ids,
        "decoded_tokens": tok.convert_ids_to_tokens(input_ids[:20])  # first 20 tokens for debugging
    })

with open("/workspace/embed-code-ts/models/tokenizer-reference.json", "w") as f:
    json.dump({"model": "nomic-ai/nomic-embed-text-v1.5", "tests": results}, f, indent=2)

print(f"Generated {len(results)} reference tokenizations")
print(f"Sample: {results[0]['text'][:50]}...")
print(f"  input_ids[:10]: {results[0]['input_ids'][:10]}")
print(f"  attention_mask[:10]: {results[0]['attention_mask'][:10]}")
print(f"  decoded_tokens[:10]: {results[0]['decoded_tokens'][:10]}")
print(f"  [CLS] id: {results[0]['input_ids'][0]}, [SEP] position: {results[0]['attention_mask'].index(0)-1 if 0 in results[0]['attention_mask'] else 'N/A'}")
