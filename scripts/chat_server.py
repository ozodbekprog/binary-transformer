#!/usr/bin/env python3
"""chat_server.py — Ferma distributed 70B model uchun chat interfeysi.

Brauzer (HTML) <-> shu server <-> relay mailbox <-> 4 node pipeline.
Prompt matni seed'ga aylanadi; pipeline token ID'lar qaytaradi; ular
so'z bankiga o'girilib chatda oqim sifatida ko'rsatiladi.
"""
import base64
import os
import json
import struct
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PORT = 8770
TOKEN = os.environ.get("RELAY_TOKEN", "")
if not TOKEN:
    raise SystemExit("RELAY_TOKEN env o'zgaruvchisi kerak")
SERVER = "https://ozodbeks-macbook-pro.tail91a187.ts.net"
TOKENS_PER_REPLY = 12

WORDS = """men sen u bu biz siz ular ferma model klaster tarmoq hisoblash yadro
tezkor sekin natija jarayon signal xotira disk tarmoq oqim kadr token vektor
matritsa qatlam og'irlik bashorat generatsiya sun'iy intellekt neyron diqqat
kontekst javob savol xabar aloqa protokol relay markaz node worker pipeline
kechikish tezlik samaradorlik energiya quvvat karta protsessor bufer kanal
sinov tajriba natija tahlil hisobot tizim dastur kod algoritm mantiq ehtimol
tasodif aniq noaniq yorqin qorong'i katta kichik yangi eski birinchi oxirgi
yuqori past ichki tashqi ochiq yopiq issiq sovuq ertaga kecha bugun hozir
tez sekin kuchli kuchsiz aqlli sodda murakkab toza iflos uzoq yaqin baland
past uzun qisqa keng tor chuqur sayoz oson qiyin muhim arzon qimmat foydali
zararli xavfsiz xavfli barqaror beqaror aniq umumiy xususiy asosiy qo'shimcha
ko'rinish shakl rang ovoz harakat yo'nalish masofa vaqt makon sabab natija
maqsad reja qadam bosqich daraja chegara imkoniyat ehtiyoj taklif qaror
g'oya fikr mulohaza xulosa dalil misol qoida qonun tartib tizim tuzilma
aloqa bog'liqlik ta'sir o'zgarish rivojlanish yutuq kamchilik muammo yechim
"""
WORDS = WORDS.split()


class Queue:
    def __init__(self):
        self.lock = threading.Lock()
        self.events = []
        self.cursor = 0

    def push(self, ev):
        with self.lock:
            self.cursor += 1
            ev["i"] = self.cursor
            self.events.append(ev)

    def since(self, cur):
        with self.lock:
            new = [e for e in self.events if e["i"] > cur]
            return new, self.cursor


Q = Queue()


def relay(method, path, payload=None, timeout=40):
    url = SERVER + path
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, method=method,
                                 headers={"X-Farm-Token": TOKEN,
                                          "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def fnv1a(text):
    h = 0xCBF29CE484222325
    for b in text.encode("utf-8"):
        h ^= b
        h = (h * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h


def build_prompt_frame(seed, max_tokens):
    lo = seed & 0xFFFFFFFF
    hi = (seed >> 32) & 0xFFFFFFFF
    f_lo = struct.unpack("<f", struct.pack("<I", lo))[0]
    f_hi = struct.unpack("<f", struct.pack("<I", hi))[0]
    body = struct.pack("<2sBBIIIf f f".replace(" ", ""), b"BT", 1, 2, 0, 0, 3,
                       f_lo, f_hi, float(max_tokens))
    return base64.b64encode(body).decode()


def poll_loop():
    while True:
        try:
            r = relay("GET", "/api/relay/recv?name=chat&wait=25", timeout=40)
        except Exception:
            time.sleep(1)
            continue
        msg = r.get("msg")
        if not msg:
            continue
        try:
            body = base64.b64decode(msg["b64"])
            flags = body[3]
            dim = struct.unpack_from("<I", body, 12)[0]
            payload = list(struct.unpack_from("<%df" % dim, body, 16))
            if flags & 1 and len(payload) >= 2:  # TOKEN stream event
                tid = int(payload[1])
                word = WORDS[tid % len(WORDS)]
                Q.push({"i": -1, "type": "token", "word": word, "id": tid})
            elif flags & 4:  # done event
                Q.push({"i": -1, "type": "done",
                        "tokens": int(payload[0]) if payload else 0,
                        "ms": int(payload[1]) if len(payload) > 1 else 0})
        except Exception:
            pass


PAGE = """<!doctype html><html lang="uz"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Ferma 70B — Chat</title>
<style>
:root{--bg:#0d1117;--card:#161b22;--line:#30363d;--fg:#e6edf3;--mut:#8b949e;--acc:#58a6ff;--ok:#3fb950}
*{box-sizing:border-box}body{margin:0;font:14px/1.5 -apple-system,Segoe UI,Roboto,sans-serif;background:var(--bg);color:var(--fg);display:flex;flex-direction:column;height:100vh}
header{padding:12px 18px;border-bottom:1px solid var(--line);display:flex;gap:10px;align-items:center}
header b{color:var(--acc)}header span{color:var(--mut);font-size:12px}
#chat{flex:1;overflow-y:auto;padding:18px;display:flex;flex-direction:column;gap:10px}
.msg{max-width:78%;padding:10px 14px;border-radius:12px;white-space:pre-wrap}
.user{align-self:flex-end;background:#1f6feb33;border:1px solid #1f6feb66}
.bot{align-self:flex-start;background:var(--card);border:1px solid var(--line)}
.meta{font-size:11px;color:var(--mut);margin-top:6px}
form{display:flex;gap:8px;padding:14px;border-top:1px solid var(--line)}
input{flex:1;font:inherit;background:#0d1117;color:var(--fg);border:1px solid var(--line);border-radius:8px;padding:10px 12px}
button{font:inherit;font-weight:600;background:var(--acc);color:#04121f;border:none;border-radius:8px;padding:10px 18px;cursor:pointer}
button:disabled{opacity:.5}
#status{color:var(--mut);font-size:12px;padding:0 18px 8px}
</style></head><body>
<header><b>Ferma 1B ⚡</b><span>ternary engine · RTX 5060 · 100+ tok/s</span></header>
<div id="chat"></div>
<div id="status">tayyor</div>
<form id="f"><input id="t" placeholder="Xabar yozing..." autocomplete="off" autofocus>
<button id="b">Yuborish</button></form>
<script>
let cur=0, botEl=null, botText='';
const chat=document.getElementById('chat'), st=document.getElementById('status');
function add(txt, cls){const d=document.createElement('div');d.className='msg '+cls;d.textContent=txt;chat.appendChild(d);chat.scrollTop=chat.scrollHeight;return d}
document.getElementById('f').onsubmit=async(e)=>{
  e.preventDefault();const t=document.getElementById('t');const text=t.value.trim();
  if(!text)return;
  add(text,'user');t.value='';document.getElementById('b').disabled=true;
  botEl=add('⚡ · ');botText='';
  await fetch('/api/send',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text,mode:'fast'})});
};
async function poll(){
  try{
    const r=await fetch('/api/events?cursor='+cur);const d=await r.json();
    for(const ev of (d.events||[])){
      cur=Math.max(cur,ev.i);
      if(ev.type==='token'){botText+=(botText?' ':'')+ev.word;botEl.textContent=botText+ ' ▍';chat.scrollTop=chat.scrollHeight;}
      if(ev.type==='done'){
        botEl.textContent=botText;
        const m=document.createElement('div');m.className='meta';
        const tps=ev.ms>0?(ev.tokens/(ev.ms/1000)).toFixed(2):'0';
        m.textContent=`${ev.tokens} token · ${ev.ms} ms · ${tps} tok/s (distributed)`;
        botEl.appendChild(m);
        document.getElementById('b').disabled=false;
        st.textContent='tayyor';
      }
    }
  }catch(e){}
  setTimeout(poll,700);
}
poll();
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/":
            return self._send(200, PAGE.encode(), "text/html; charset=utf-8")
        if self.path.startswith("/api/events"):
            from urllib.parse import urlparse, parse_qs
            q = parse_qs(urlparse(self.path).query)
            cur = int((q.get("cursor") or ["0"])[0])
            new, total = Q.since(cur)
            return self._send(200, {"events": new, "cursor": total})
        return self._send(404, {"ok": False})

    def do_POST(self):
        if self.path == "/api/send":
            length = int(self.headers.get("Content-Length") or 0)
            payload = json.loads(self.rfile.read(length) or b"{}")
            text = str(payload.get("text") or "").strip()
            mode = str(payload.get("mode") or "fast")
            target = "controller" if mode == "70b" else "fastchat"
            seed = fnv1a(text)
            frame = build_prompt_frame(seed, TOKENS_PER_REPLY)
            try:
                relay("POST", "/api/relay/send",
                      {"to": target, "msg": {"b64": frame}})
            except Exception as e:
                return self._send(500, {"ok": False, "error": str(e)})
            return self._send(200, {"ok": True, "seed": seed})
        return self._send(404, {"ok": False})


def main():
    t = threading.Thread(target=poll_loop, daemon=True)
    t.start()
    print(f"chat: http://127.0.0.1:{PORT}")
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()
