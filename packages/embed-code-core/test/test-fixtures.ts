/**
 * Deterministic test fixtures for embed-code-ts.
 *
 * Uses mulberry32 PRNG with seed=42 to guarantee identical outputs
 * across platforms and CI runs. These fixtures serve as the single
 * source of truth for both unit tests and benchmarks.
 */

/** Mulberry32 PRNG — deterministic, seed-based */
function mulberry32(seed: number): () => number {
  return () => {
    seed |= 0;
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const rand = mulberry32(42);

/** Pick a random element from an array */
function pick<T>(arr: T[]): T {
  return arr[Math.floor(rand() * arr.length)];
}

// ─── Code snippet templates ──────────────────────────────────────────

const CODE_SNIPPETS = [
  // Python
  'def fibonacci(n):\n  if n <= 1:\n    return n\n  return fibonacci(n - 1) + fibonacci(n - 2)',
  'def quicksort(arr):\n  if len(arr) <= 1:\n    return arr\n  pivot = arr[len(arr) // 2]\n  left = [x for x in arr if x < pivot]\n  middle = [x for x in arr if x == pivot]\n  right = [x for x in arr if x > pivot]\n  return quicksort(left) + middle + quicksort(right)',
  'class BinarySearchTree:\n  def __init__(self):\n    self.root = None\n  def insert(self, value):\n    if self.root is None:\n      self.root = Node(value)\n    else:\n      self._insert_recursive(self.root, value)',
  'async function fetchUserData(userId: string): Promise<User> {\n  const response = await fetch(`/api/users/${userId}`);\n  if (!response.ok) throw new Error(`HTTP ${response.status}`);\n  return response.json();\n}',
  'interface CacheEntry<T> {\n  readonly key: string;\n  readonly value: T;\n  readonly expiresAt: number;\n  readonly createdAt: number;\n}',
  'const factorial = (n: number): number =>\n  n <= 1 ? 1 : n * factorial(n - 1);',
  'func (s *Server) HandleRequest(ctx context.Context, req *http.Request) (*http.Response, error) {\n  span, ctx := opentracing.StartSpanFromContext(ctx, "handle_request")\n  defer span.Finish()\n  return s.router.ServeHTTP(ctx, req)\n}',
  '#[derive(Debug, Clone, Serialize, Deserialize)]\npub struct EmbeddingRequest {\n  pub texts: Vec<String>,\n  pub model: String,\n  pub normalize: bool,\n}',
  "SELECT u.id, u.name, COUNT(o.id) as order_count\nFROM users u\nLEFT JOIN orders o ON u.id = o.user_id\nWHERE u.created_at > NOW() - INTERVAL '30 days'\nGROUP BY u.id, u.name\nORDER BY order_count DESC\nLIMIT 100;",
  'import torch\nimport torch.nn as nn\n\nclass AttentionBlock(nn.Module):\n  def __init__(self, dim, heads):\n    super().__init__()\n    self.qkv = nn.Linear(dim, dim * 3)\n    self.proj = nn.Linear(dim, dim)\n    self.heads = heads',
];

const QUERY_TEXTS = [
  'How to sort an array using quicksort?',
  'Implement a binary search tree in Python',
  'Database query for user order counts',
  'TypeScript interface for cache entries',
  'Compute the n-th Fibonacci number recursively',
  'Rust struct for embedding requests',
  'HTTP server request handler in Go',
  'Neural network attention block in PyTorch',
  'Fetch user data from REST API',
  'GraphQL query for paginated results',
];

// ─── Fixture generators ──────────────────────────────────────────────

/** Single code snippet for basic embedding tests */
export const basicCodeSnippet = 'def hello_world():\n  print("Hello, World!")';

/** Query–document pair for similarity verification */
export const queryDocPair = {
  query: 'search_query: Recursive factorial implementation',
  document: 'search_document: def factorial(n): return 1 if n <= 1 else n * factorial(n - 1)',
};

/** Unrelated text pair for negative similarity verification */
export const unrelatedPair = {
  text1: 'search_query: How to implement a binary search tree?',
  text2: 'search_document: def sort_strings(arr): return sorted(arr, key=len)',
};

/** Generate N random code snippets */
export function generateCodeSnippets(n: number): string[] {
  const result: string[] = [];
  for (let i = 0; i < n; i++) {
    result.push(pick(CODE_SNIPPETS));
  }
  return result;
}

/** Generate N random queries */
export function generateQueries(n: number): string[] {
  const result: string[] = [];
  for (let i = 0; i < n; i++) {
    result.push(pick(QUERY_TEXTS));
  }
  return result;
}

/** Pre-computed fixtures for benchmark consistency */
export const benchmarkFixtures = Object.freeze({
  singleShort: 'search_document: const x = 42',
  singleMedium:
    'search_document: function binarySearch(arr: number[], target: number): number {\n  let left = 0;\n  let right = arr.length - 1;\n  while (left <= right) {\n    const mid = Math.floor((left + right) / 2);\n    if (arr[mid] === target) return mid;\n    if (arr[mid] < target) left = mid + 1;\n    else right = mid - 1;\n  }\n  return -1;\n}',
  singleLong: 'search_document: ' + CODE_SNIPPETS[1], // quicksort
  batch4: [
    'search_document: const add = (a: number, b: number): number => a + b',
    'search_document: def multiply(x, y): return x * y',
    'search_document: func divide(a, b float64) float64 { return a / b }',
    'search_document: let subtract = (a, b) => a - b',
  ],
  batch8: [
    'search_document: const PI = 3.141592653589793',
    'search_document: def circle_area(r): return 3.14159 * r * r',
    'search_document: func squareRoot(x float64) float64 { return math.Sqrt(x) }',
    'search_document: let pow = (base: number, exp: number) => base ** exp',
    'search_document: const abs = (n: number) => n < 0 ? -n : n',
    'search_document: def is_prime(n): return n > 1 and all(n % i != 0 for i in range(2, int(n**0.5)+1))',
    'search_document: func isEven(n int) bool { return n%2 == 0 }',
    'search_document: let clamp = (v: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, v))',
  ],
  // Large batch for benchmark throughput testing
  batch16: [
    'search_document: const a = 1',
    'search_document: const b = 2',
    'search_document: def f1(): pass',
    'search_document: def f2(): pass',
    'search_document: let x = "hello"',
    'search_document: let y = "world"',
    'search_document: func g1() {}',
    'search_document: func g2() {}',
    'search_document: class C1 {}',
    'search_document: class C2 {}',
    'search_document: import os',
    'search_document: import sys',
    'search_document: export default {}',
    'search_document: export const z = 0',
    'search_document: module M1 end',
    'search_document: module M2 end',
  ],
} as const);
