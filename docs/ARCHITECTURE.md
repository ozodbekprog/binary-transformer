# Arxitektura — Binar/Ternar Transformer Engine (Phase 1)

## 1. Nega ko'paytirishsiz?

Ternar qiymatlar {-1, 0, +1} uchun `a·b` ko'paytirish amali bit
operatsiyalariga aylanadi. Har bir qiymat ikki bit-plane bilan ifodalanadi:

```
+1 -> (pos=1, neg=0)
-1 -> (pos=0, neg=1)
 0 -> (pos=0, neg=0)
```

Unda skalyar ko'paytma:

```
a·b = popcount(pos_a & pos_b) + popcount(neg_a & neg_b)
    - popcount(pos_a & neg_b) - popcount(neg_a & pos_b)
```

Ya'ni: **AND/XOR + POPCNT** — FP ko'paytirish yo'q, energiya va kremniy tejaladi.
Binar {0,1} — xuddi shu sxemaning xususiy holi (klassik XNOR+POPCNT).

## 2. Modul A — SIMD Bitwise Engine (`src/core/ternary_simd.hpp`)

- **Packing:** `bitpack.hpp` — `uint64` (CPU) va `uint32` (GPU) so'zlar;
  `mmap` orqali og'irliklarni diskdan xotiraga ulash (`MappedFile`).
- **Yadrolar:**
  - `dot_scalar` — reference/fallback (`__builtin_popcountll`).
  - `dot_avx2` — 256-bit: 4 so'z/iteratsiya, nibble-LUT (`vpshufb`) popcount,
    `vpsadbw` bilan gorizontal yig'indi; manfiy baytlar +32 surish bilan
    to'g'rilanadi.
  - `dot_avx512` — `vpopcntq` (VPOPCNTDQ) bilan 8 so'z/iteratsiya.
  - `dot_neon` — ARM `VCNT` + `vpaddl`.
- **Dispatch:** `detect_best()` — runtime CPUID (`__builtin_cpu_supports`).
- **Tiling/blocking:** `TileConfig{k_block_words, n_block, m_block}` — M-bloklar
  thread'larga bo'linadi (atomic counter), K-bloklar L1/L2 do'stona ishlanadi.
  Branch miss yo'q (faqat tabiiy loop shartlari), cache miss kamayadi.
- **Parallelizm:** `ternary_matmul(..., threads)` — M bo'ylab dinamik blok
  taqsimoti (katta `d_model` da yaxshi scaling).

## 3. Modul B — Vulkan Compute (`src/gpu/`)

- `binary_matmul.comp`: har thread bitta output qator uchun `batch` ta ustunni
  hisoblaydi; `bitCount()` — native, `subgroupAdd`/`subgroupXor` — ixtiyoriy
  `BT_USE_SUBGROUP` rejimida (K ni subgroup bo'ylab bo'lish).
- `vulkan_wrapper.cpp`: **SDK'siz** — `libvulkan.so.1` / `vulkan-1.dll` /
  `libvulkan.dylib` `dlopen` qilinadi, funksiyalar `vkGetInstanceProcAddr` /
  `vkGetDeviceProcAddr` orqali olinadi (`VK_NO_PROTOTYPES`).
- Og'irliklar `DEVICE_LOCAL` xotiraga staging (16MB chunk) orqali yuklanadi;
  butun model bitta command buffer'da layerlar ketma-ketligi sifatida yoziladi
  (har token = bitta submit + fence wait).
- `--check` rejimida GPU natijasi CPU reference bilan **aynan** solishtiriladi.

## 4. Modul C — Pipeline RPC (`src/net/pipeline_rpc.hpp`)

- Header-only, non-blocking TCP (`O_NONBLOCK` + `select`), length-prefixed
  frame: `magic(2) | version(1) | flags(1) | seq(4) | dim(4) | float32[dim]`.
- Rollar: `first` (generator) → `middle` (qabul + transform + forward) →
  `last` (qabul + transform). Faqat `1×d_model` hidden-state uzatiladi —
  og'irliklar har node'da lokal qoladi.
- `pipeline_ping` — 4 node'li zanjir demosi (frames/s, MB/s hisoboti).

## 5. Benchmark metodikasi (`benchmark_engine`)

1. `--params` bo'yicha fake ternar og'irliklar generatsiya qilinadi
   (default 1B; `layers = params / d²`, `d=2048`).
2. CPU: har layer `d×d` ternar matmul + quantize/activation; `--tokens` marta
   o'lchanadi; `tok/s = tokens / vaqt`, `MAC/s = tok/s × params`.
3. GPU: butun model VRAM'da; har token butun zanjirni bitta submit bilan
   o'tadi (og'irliklar VRAM'dan oqib o'tadi — decode memory-bound).
4. `--check` — 61×97 kichik matritsa CPU reference == AVX2 == GPU.

Eslatma: bu **compute benchmark**, haqiqiy til modeli emas — attention/softmax
yo'q; maqsad ternar GEMM yadrolarining tezligini o'lchash.

## 6. Klaster taqsimoti (online 4 node)

```
[Ozodbeks-MBP]  tokenizer / orchestrator / pipeline 'first'
      |
      v  (pipeline_rpc: 1×d float32)
[gtx1650-pc]    Vulkan GPU worker (GTX 1650 4GB)
      |
      v
[ryzen-ai7]     Vulkan GPU worker (RTX 5060 8GB)
      |
      v
[asror-fedora]  AVX2 SIMD CPU worker
```

Phase 2 g'oyalari: haqiqiy attention yadrosi, KV-cache bit-packing,
int8→ternar kalibratsiya, layerlarni node'lar bo'ylab bo'lish (pipeline
parallelizmi), tail-latency o'lchovlari.
