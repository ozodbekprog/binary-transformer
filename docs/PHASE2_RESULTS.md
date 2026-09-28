# Phase 2 — Distributed 70B Ternary Pipeline: Natijalar

**Sana:** 2026-09-28 · **Holat:** ISHLADI ✅ (4/4 node, real generatsiya)

## Topologiya (Pipeline Parallelism)

```
[Ozodbeks-MBP]          [ryzen-ai7]           [gtx1650-pc]         [asror-fedora]
 controller              worker1               worker2              worker3 (final)
 layers 0-11             layers 12-43          layers 44-63         layers 64-79 + head
 CPU (8.05B param)       RTX 5060 (21.5B)      GTX 1650 (13.4B)     CPU (10.7B)
       └──── frame (1×8192 float32) ────►──────────►─────────►────────┘
       ◄──────────────── token id (return) ────────────────────────────┘
```

## Model (sintetik 70B-class)
- 80 layer, d_model = 8192, har layerda 10 matritsa (4 attention + 2×3 FFN)
- **53.7B parametr @ 2 bit = 13.4 GB** — vazifadagi 12–14 GB budjetga mos
- Og'irliklar deterministik seed (42) bilan har node'da o'zi generatsiya qiladi

## Transport (real NAT sharoiti)
- Node↔node to'g'ridan-to'g'ri TCP **yopiq** (CGNAT/firewall — tekshirildi)
- Yechim: **markaz relay mailbox** — farm serverga `/api/relay/send|recv`
  (long-poll) qo'shildi; har node'da Python `relay_bridge.py` engine stdio'sini
  relay'ga ulaydi
- LAN/Tailscale uchun `--transport tcp` rejimi ham mavjud (kod tayyor)

## O'lchov natijalari

| Bosqich | Qurilma | Parametr | Weight gen | Stage vaqti (token) |
|---|---|---|---|---|
| Controller | Mac intel i9, CPU | 8.05B | 5.6 s | ~0.6–1.0 s |
| Worker1 | RTX 5060 8GB | 21.5B (5.37GB VRAM) | 14.9 s | **47–233 ms** |
| Worker2 | GTX 1650 4GB | 13.4B (3.36GB VRAM) | 15.3 s | **397–511 ms** |
| Worker3 | 4×CPU AVX-512 | 10.7B (RAM) | 24.5 s | **197–222 ms** |

**End-to-end (4 token, controller vaqti):**
```
token 1/4  latency=4029 ms
token 2/4  latency=4366 ms
token 3/4  latency=5036 ms
token 4/4  latency=3592 ms

PIPELINE-RESULT: tokens=4  total=17.03s  tok_per_s=0.235  avg_latency=4256 ms
```

**Xulosa:** hisoblash bosqichlari tez (jami ~1.0–1.5 s/token), asosiy xarajat —
NAT relay orqali 4 ta tarmoq hop'i (~3 s/token). LAN/Tailscale'da to'g'ridan-
to'g'ri TCP rejimida latency ~5–10x tushishi kutiladi.

## 10 ta opencode agent (taqsimlangan tekshiruv)
- Ozodbeks-MBP: 4 agent (controller, relay, resurs, natija)
- gtx1650-pc: 2 agent (frame oqimi, GPU)
- ryzen-ai7: 2 agent (worker holati, GPU/RAM)
- asror-fedora: 2 agent (final stage, CPU yuklama)
- 9/10 muvaffaqiyatli; 1 agentda opencode provider xatosi (tizimga aloqasiz)

## Fayllar
- `src/main_multi_node.cpp` — distributed engine (roles, layers, sampling)
- `src/net/pipeline_rpc.hpp` — stream/stdio frame transport, ring buffer
- `scripts/start_70b_cluster.sh` — start/stop/status/run orchestratsiya
- `scripts/relay_bridge.py` — relay HTTP ↔ engine stdio ko'prik
- Farm server patch: `/api/relay/send|recv` (backup: `server.py.bak-*`)

## Keyingi qadamlar (Phase 3)
1. LAN/tailnet'da `--transport tcp` bilan to'g'ridan-to'g'ri pipeline (relay'siz)
2. Ko'p streamli (batched) generatsiya — throughput uchun
3. Real BitNet b1.58 og'irliklarini yuklash (`--model-path`)
4. GPU'da X upload + per-layer dispatch batching (submit overheadini kamaytirish)
