# Phase 4 — 100+ tok/s natijalari (1B fast chat)

**Sana:** 2026-09-28 · **Holat:** ERISHILDI ✅ — **104.1 tok/s** (o'lchangan)

## Qisqa xulosa

| Rejim | Model | Tezlik | Izoh |
|---|---|---|---|
| ⚡ Fast (1B lokal, RTX 5060) | 1B (238 layer, d=2048) | **104.1 tok/s** | Chatdagi "⚡ 1B lokal" checkbox |
| 70B klaster pipeline | 53.7B (80 layer, d=8192) | 0.23 tok/s | Shared NAT relay + Mac CPU floor |

## Nega 70B klasterda 100+ mumkin emas (tekshirilgan)
```
Kerak:  100 tok/s × 53.7B MAC = 5370 G-MAC/s
Bor:    296 (RTX 5060) + 72 (GTX 1650) + 32.5 (Mac CPU) = ~400 G-MAC/s
Yetishmaydi: 13.4x  (tarmoq 0 bo'lsa ham)
```
Qo'shimcha: node'lar CGNAT ortida (to'g'ridan-to'g'ri TCP yo'q), relay hop
o'lchandi: 106 ms. Asosiy 70B to'sig'i — Mac CPU bosqichi (~3.2 s/token, swap).

## 100+ tok/s qanday olindi (1B fast engine)
Per-token xarajat tahlili (RTX 5060, o'lchangan):

| Bosqich | Vaqt/token |
|---|---|
| CPU (quantize/pack/embed/act) | 0.02 ms |
| **GPU execute (238 layer)** | **9.52 ms** |
| Readback + sampling | 0.05 ms |
| **Jami** | **9.6 ms → 104 tok/s** |

Asosiy topilma: `pack_col32` da 1 MB/token behuda allokatsiya bor edi
(tuzatildi: 512 KB → 512 B). Raw GPU quvvati: 296 tok/s (benchmark).

## Ishlatish
```bash
# ryzen'da:
setsid nohup bash ~/bt/bin/launch_fast.sh ...   # relay bridge + fast engine
# Chat server (gtx):
python3 scripts/chat_server.py                  # brauzer: 127.0.0.1:8770
# Chatda: "⚡ 1B lokal (100+ tok/s)" checkboxni belgilang
```

## Keyingi optimizatsiya (300+ tok/s yo'li)
Activation + quantize + pack'ni GPU shader pass'iga o'tkazish (bitta submit'da
12 token zanjiri) → per-token sync yo'qoladi → ~290 tok/s kutiladi.
