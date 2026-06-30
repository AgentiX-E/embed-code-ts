# @agentix-e/embed-code-ts — 技术设计方案

## 1. 概述

将 nomic-embed-code 模型的 int8 权重嵌入 npm 包，借鉴 C/C++ 中 `incbin` 的思路——在构建时将二进制数据直接编译进产物，实现零网络依赖的运行时加载。

## 2. 核心挑战

| 挑战 | 详情 |
|------|------|
| **模型规模** | nomic-embed-code 为 7B 参数模型，int8 ONNX 约 7-8GB |
| **npm 限制** | npm 单包上限约 500MB |
| **无现成 ONNX** | HF 仓库仅有 safetensors，需自行转换为 ONNX int8 |
| **TypeScript 约束** | 单文件 base64 字面量不能过大（JIT 解析限制） |

## 3. Incbin 分层嵌入策略（三层架构）

借鉴 incbin 的哲学——"编译时嵌入，运行时即用"，设计三层渐进式嵌入策略：

```
┌─────────────────────────────────────────────────────────────┐
│                    Embedding Strategy                        │
├─────────────┬─────────────────┬─────────────────────────────┤
│   Layer 1   │   Layer 2       │   Layer 3                   │
│  FULLY      │  CHUNKED        │  FINGERPRINT                │
│  EMBEDDED   │  EMBEDDED       │  + DOWNLOAD                 │
├─────────────┼─────────────────┼─────────────────────────────┤
│ < 200MB     │  200MB - 2GB    │  > 2GB                      │
│ Base64 常量  │  分片 Base64    │  嵌入 Hash + Metadata       │
│ 零网络依赖   │  懒加载重组      │  运行时 CDN 下载             │
├─────────────┼─────────────────┼─────────────────────────────┤
│ Tokenizer   │  Small Models   │  nomic-embed-code (7B)      │
│ Config      │  Distilled      │  Future: distilled version  │
│ Vocab       │  Future: MoE    │                             │
└─────────────┴─────────────────┴─────────────────────────────┘
```

### 3.1 Layer 1: Fully Embedded（完全嵌入）

```typescript
// 构建时生成: src/weights/tokenizer.inc.ts
// incbin 哲学：二进制直接作为代码常量编译进产物
export const TOKENIZER_BASE64 = "eyJ2ZXJzaW9uIjoxLjAsInRydW5jYXRpb24iOn...";
export const TOKENIZER_BYTES = /* base64 → Uint8Array at import time */;
```

适用：
- Tokenizer JSON (~11MB)
- Model Config JSON (~1KB)
- Vocabulary (~3MB)
- 小型蒸馏模型（未来）

### 3.2 Layer 2: Chunked Embedded（分片嵌入）

```typescript
// 构建时生成: src/weights/model.int8.chunk_0.ts ~ chunk_N.ts
export const CHUNK_0 = "AQIDBAUG..."; // 50MB base64 each
export const CHUNK_0_SHA256 = "abc123...";
```

ONNX 模型被分割为 50MB 的 base64 块。运行时按需懒加载并重组为完整 ArrayBuffer。

### 3.3 Layer 3: Fingerprint + Download（指纹+下载）

```typescript
// 构建时生成: src/weights/model.manifest.ts
export const MODEL_MANIFEST = {
  version: "nomic-embed-code-v1",
  url: "https://huggingface.co/AgentiX-E/nomic-embed-code/resolve/main/onnx/model_int8.onnx",
  sha256: "e3b0c44298fc1c149afbf4c8996fb924...",
  size: 7516192768, // 7.5GB
  chunks: 16,       // 分成16个500MB块
};
```

运行时自动下载、校验、缓存。API 与 Layer 1/2 完全一致。

## 4. 项目架构

```
embed-code-ts/
├── package.json
├── tsconfig.json
├── tsup.config.ts                    # 构建配置
│
├── src/
│   ├── index.ts                      # 入口：导出 EmbedCode 类
│   ├── embed-code.ts                 # 核心：嵌入推理引擎
│   ├── tokenizer.ts                  # Tokenizer（基于 tokenizer.json）
│   ├── pooling.ts                    # Last-token pooling
│   │
│   ├── registry.ts                   # 模型注册中心
│   ├── model-provider.ts             # 抽象模型提供者接口
│   ├── providers/
│   │   ├── embedded-provider.ts      # Layer 1: 完全嵌入提供者
│   │   ├── chunked-provider.ts       # Layer 2: 分片嵌入提供者
│   │   └── remote-provider.ts        # Layer 3: 远程下载提供者
│   │
│   ├── weights/                      # 构建时生成的权重文件
│   │   ├── index.ts                  # 自动生成
│   │   ├── tokenizer.inc.ts          # 嵌入的 tokenizer.json
│   │   ├── config.inc.ts             # 嵌入的模型配置
│   │   ├── vocab.inc.ts              # 嵌入的词汇表
│   │   └── model.manifest.ts         # 模型清单（含 sha256）
│   │
│   ├── platform/
│   │   ├── node.ts                   # Node.js ONNX Runtime 绑定
│   │   └── web.ts                    # 浏览器 ONNX Runtime Web
│   │
│   └── types.ts                      # TypeScript 类型定义
│
├── scripts/
│   ├── build.ts                      # 构建编排器
│   ├── incbin.ts                     # incbin 核心：二进制 → TypeScript 常量
│   ├── download-model.ts             # 从 HF 下载模型
│   ├── convert-onnx.ts              # PyTorch → ONNX int8 转换
│   └── generate-types.ts            # 自动生成类型定义
│
├── models/                           # .gitignore：下载的原始模型
│   └── nomic-embed-code/
│       ├── onnx/
│       │   └── model_int8.onnx
│       ├── tokenizer.json
│       └── ...
│
├── tests/
│   ├── embed.test.ts
│   ├── tokenizer.test.ts
│   └── incbin.test.ts
│
└── examples/
    ├── basic.ts
    └── advanced.ts
```

## 5. Incbin 核心技术实现

### 5.1 incbin 脚本核心逻辑

```typescript
// scripts/incbin.ts
// 将任意二进制文件转换为 TypeScript 模块

interface IncbinOptions {
  inputPath: string;        // 输入二进制文件路径
  outputPath: string;       // 输出 .ts 文件路径
  exportName: string;       // 导出的常量名
  chunkSize?: number;       // 分片大小（字节），默认 50MB
  compress?: boolean;       // 是否 gzip 压缩后 base64
}

async function incbin(options: IncbinOptions): Promise<void> {
  const buffer = await fs.readFile(options.inputPath);
  const hash = crypto.createHash('sha256').update(buffer).digest('hex');

  if (options.compress) {
    // gzip 压缩可减少 30-50% 体积（文本模型压缩率高）
    buffer = await gzip(buffer);
  }

  const base64 = buffer.toString('base64');

  if (base64.length < CHUNK_THRESHOLD) {
    // 单文件嵌入
    await generateSingleFile(options.outputPath, options.exportName, base64, hash);
  } else {
    // 分片嵌入
    await generateChunkedFiles(options.outputPath, options.exportName, base64, hash);
  }
}
```

### 5.2 运行时解码

```typescript
// 嵌入式权重 → ArrayBuffer（零拷贝）
function decodeEmbeddedWeights(base64: string): ArrayBuffer {
  // 使用 Buffer.from (Node.js) 或 atob + Uint8Array (Browser)
  if (typeof Buffer !== 'undefined') {
    return Buffer.from(base64, 'base64').buffer;
  }
  const binary = atob(base64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  return bytes.buffer;
}
```

### 5.3 ModelProvider 抽象

```typescript
interface ModelProvider {
  readonly type: 'embedded' | 'chunked' | 'remote';
  getModelBuffer(): Promise<ArrayBuffer>;
  getConfig(): ModelConfig;
  getTokenizer(): TokenizerData;
  verify(): Promise<boolean>;  // SHA256 校验
}
```

## 6. 模型注册中心（扩展性核心）

```typescript
// src/registry.ts
const MODEL_REGISTRY: Record<string, ModelDescriptor> = {
  'nomic-embed-code-v1': {
    provider: 'remote',           // 当前使用 Layer 3（7B 太大）
    manifestUrl: './weights/model.manifest',
    tokenizerProvider: 'embedded', // Tokenizer 始终嵌入式
    configProvider: 'embedded',    // Config 始终嵌入式
    target: {
      repository: 'nomic-ai/nomic-embed-code',
      onnxFile: 'onnx/model_int8.onnx',
    },
    runtime: {
      embeddingDim: 3584,
      maxTokens: 32768,
      poolingStrategy: 'last_token',
    },
  },
  'nomic-embed-text-v1.5': {
    provider: 'embedded',         // 137MB 完全适合 Layer 1
    target: {
      repository: 'nomic-ai/nomic-embed-text-v1.5',
      onnxFile: 'onnx/model_int8.onnx',
    },
    runtime: {
      embeddingDim: 768,
      maxTokens: 8192,
      poolingStrategy: 'mean',
    },
  },
};

// 添加新模型只需在此注册
```

## 7. API 设计

```typescript
import { EmbedCode } from '@agentix-e/embed-code-ts';

// 初始化（自动处理权重加载/下载/缓存）
const embedder = await EmbedCode.create({
  model: 'nomic-embed-code-v1',     // 模型选择
  provider: 'auto',                  // auto | embedded | chunked | remote
  cacheDir: './.cache/embed-code',   // 缓存目录
  onProgress: (pct) => console.log(`下载: ${pct}%`),
});

// 生成代码嵌入
const embeddings = await embedder.embed([
  'def factorial(n):\n  return 1 if n <= 1 else n * factorial(n-1)',
  'Calculate the n-th factorial of a number',
]);

// embeddings.shape → [2, 3584]
// 支持批量、流式、相似度计算
const similarity = embedder.similarity(embeddings[0], embeddings[1]);
```

## 8. 构建流程

```
┌─────────────┐   ┌──────────────┐   ┌──────────────┐   ┌───────────┐
│ Download    │──▶│ ONNX Convert │──▶│ Incbin       │──▶│ tsup      │
│ HF Model    │   │ + Int8 Quant │   │ Generate .ts │   │ Bundle    │
└─────────────┘   └──────────────┘   └──────────────┘   └───────────┘
     ↓                   ↓                  ↓                 ↓
  models/            models/            src/weights/       dist/
  safetensors        onnx/              *.inc.ts           *.js
```

## 9. 未来扩展路径

| 场景 | 扩展方式 |
|------|---------|
| nomic-embed-code-v2 蒸馏版 | 在 registry 添加新条目，使用 Layer 1/2 |
| 新的 Nomic 模型家族 | registry 添加条目 + 下载对应模型 |
| 自定义 ONNX 模型 | 通过 `EmbedCode.create({ customModel: buffer })` |
| 浏览器端部署 | platform/web.ts 使用 onnxruntime-web |
| Deno/Bun 运行时 | 添加 platform adapter |

## 10. 实施计划

1. ✅ 项目骨架搭建（package.json, tsconfig, tsup）
2. 实现 incbin 核心脚本
3. 实现 ModelProvider 三层架构
4. 实现 Tokenizer（基于 tokenizer.json incbin 嵌入）
5. 实现 EmbedCode 核心类
6. 实现 ONNX 运行时集成
7. 编写测试
8. 编写示例
9. npm 发布配置
