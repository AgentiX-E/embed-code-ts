/**
 * Pure-TypeScript matrix multiplication (matmul / GEMM).
 *
 * Operates on Float32Array for maximum performance. All functions
 * are allocation-free: the caller must provide the output tensor.
 *
 *   C[m][n] = A[m][k] @ B[k][n]
 *
 * Row-major layout throughout.
 */

/**
 * Standard matrix multiply: C = A @ B.
 *
 *   A: Float32Array [m × k]  (row-major)
 *   B: Float32Array [k × n]  (row-major)
 *   C: Float32Array [m × n]  (row-major, caller-allocated)
 *
 * Uses loop ordering `i → j → p` to maximise L1 cache reuse
 * on A's rows (the innermost loop walks contiguous memory).
 */
export function matmul(
  a: Float32Array,
  b: Float32Array,
  c: Float32Array,
  m: number,
  k: number,
  n: number,
): void {
  // Block size tuned for L1 cache (32 KiB → 8k floats) on typical x86-64.
  const BLOCK = 32;

  for (let bi = 0; bi < m; bi += BLOCK) {
    const biEnd = Math.min(bi + BLOCK, m);
    for (let bj = 0; bj < n; bj += BLOCK) {
      const bjEnd = Math.min(bj + BLOCK, n);
      for (let bk = 0; bk < k; bk += BLOCK) {
        const bkEnd = Math.min(bk + BLOCK, k);

        for (let i = bi; i < biEnd; i++) {
          const aRowOff = i * k;
          const cRowOff = i * n;
          for (let j = bj; j < bjEnd; j++) {
            let acc = c[cRowOff + j]!;
            const bColOff = j;
            for (let p = bk; p < bkEnd; p++) {
              acc += a[aRowOff + p]! * b[p * n + bColOff]!;
            }
            c[cRowOff + j] = acc;
          }
        }
      }
    }
  }
}

/**
 * Matrix multiply with implicit bias addition: C = A @ B + bias.
 *
 * Bias is broadcast over rows: bias[j] is added to every row's column j.
 */
export function matmulBiased(
  a: Float32Array,
  b: Float32Array,
  bias: Float32Array,
  c: Float32Array,
  m: number,
  k: number,
  n: number,
): void {
  // Initialise C with bias
  for (let i = 0; i < m; i++) {
    const off = i * n;
    for (let j = 0; j < n; j++) {
      c[off + j] = bias[j]!;
    }
  }

  matmul(a, b, c, m, k, n);
}

/**
 * Matrix-vector product: y = A @ x.
 *
 *   A: Float32Array [m × k]  (row-major)
 *   x: Float32Array [k]
 *   y: Float32Array [m]       (caller-allocated)
 */
export function matvec(
  a: Float32Array,
  x: Float32Array,
  y: Float32Array,
  m: number,
  k: number,
): void {
  for (let i = 0; i < m; i++) {
    let acc = 0;
    const off = i * k;
    for (let j = 0; j < k; j++) {
      acc += a[off + j]! * x[j]!;
    }
    y[i] = acc;
  }
}

/**
 * Element-wise multiply and accumulate: C[i][j] += A[i][j] * B[i][j].
 */
export function hadamardAccumulate(
  a: Float32Array,
  b: Float32Array,
  c: Float32Array,
  len: number,
): void {
  for (let i = 0; i < len; i++) {
    c[i] += a[i]! * b[i]!;
  }
}

/**
 * Transpose a matrix in-place (square matrices only for performance).
 * For non-square, caller should provide a separate output array.
 */
export function transposeSquare(m: Float32Array, dim: number, out: Float32Array): void {
  for (let i = 0; i < dim; i++) {
    for (let j = 0; j < dim; j++) {
      out[j * dim + i] = m[i * dim + j]!;
    }
  }
}
