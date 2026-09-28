# Binar/Ternar Transformer Engine — Phase 1

**Maqsad:** multiplication-less (ko'paytirishsiz) Binar/Ternar Transformer
yadrolarining prototipi: CPU SIMD (AVX2/AVX-512/NEON), GPU (Vulkan compute) va
node'lararo yengil pipeline transporti.

Tashqi og'ir dependency **yo'q**: faqat C++20 kompilyator + (ixtiyoriy) Vulkan
header + (ixtiyoriy) `glslc` shader kompilyatori.

---

## Tuzilma

```
binary-transformer/
├── CMakeLists.txt                  # Mac/Linux/Windows kross-platforma
├── scripts/build_simple.sh         # CMake'siz build (Mac va yengil node'lar)
├── src/core/bitpack.hpp            # bit-packing + mmap yuklash
├── src/core/ternary_simd.hpp/.cpp  # Modul A: SIMD bitwise engine
├── src/gpu/binary_matmul.comp      # Modul B: Vulkan compute shader
├── src/gpu/vulkan_wrapper.hpp/.cpp # Modul B: C++ binding (dlopen, SDK'siz)
├── src/net/pipeline_rpc.hpp        # Modul C: non-blocking pipeline RPC
├── src/cli/benchmark_engine.cpp    # 1B fake ternar weights benchmark
├── src/cli/pipeline_ping.cpp       # zanjir demosi (Node1->2->3->4)
├── tests/selftest.cpp              # packing/yadro to'g'riligi
└── docs/ARCHITECTURE.md            # arxitektura va klaster taqsimoti
```

## Build

### CMake bilan (tavsiya)
```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/bin/bt_selftest
```

### CMake'siz (Mac va minimal node'lar)
```bash
CXX=clang++ bash scripts/build_simple.sh
./bin/bt_selftest
```

Vulkan header bo'lmasa build CPU-only bo'ladi (GPU qismi stub — xato bermaydi).

## Benchmark

```bash
# 1B parametrli fake ternar model, auto backend (CPU + GPU taqqoslash)
./build/bin/benchmark_engine

# Tez sinov
./build/bin/benchmark_engine --params 1e8 --tokens 5 --check

# Faqat CPU / ma'lum yadro
./build/bin/benchmark_engine --backend cpu --kernel avx2

# Backendlar ro'yxati (shu mashinada)
./build/bin/benchmark_engine --list
```

Chiqish: `tok/s`, `G-MAC/s`, `pass ms` — har backend uchun, g'olib bilan.

## Pipeline demosi (4 node)

```bash
# Node 4:  pipeline_ping --role last   --listen 9004 --dim 512 --frames 5000
# Node 3:  pipeline_ping --role middle --listen 9003 --next HOST4:9004 --dim 512 --frames 5000
# Node 2:  pipeline_ping --role middle --listen 9002 --next HOST3:9003 --dim 512 --frames 5000
# Node 1:  pipeline_ping --role first  --next HOST2:9002 --dim 512 --frames 5000
```

## Klaster xaritasi (online 4 node)

| Node | Quvvat | Rol |
|---|---|---|
| Ozodbeks-MBP (controller) | Intel i9 (AVX2), 16 thread | tokenizer/orchestrator, pipeline boshi |
| gtx1650-pc | GTX 1650 4GB + AVX2 | Vulkan GPU worker |
| ryzen-ai7 | RTX 5060 8GB + AVX2 16c | Vulkan GPU worker |
| asror-fedora | CPU AVX2 4c | SIMD CPU worker |

Batafsil: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)

## Litsenziya

MIT (prototip).
