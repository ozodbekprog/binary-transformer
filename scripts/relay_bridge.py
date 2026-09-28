#!/usr/bin/env python3
"""relay_bridge.py — engine stdio <-> farm relay HTTP mailbox o'rtasidagi ko'prik.

Engine stdout'dagi kadrlarni keyingi node mailbox'iga yuboradi; o'z inbox'idan
kelgan kadrlarni engine stdin'iga uzatadi. Shu tarzda NAT ortidagi node'lar
ham markaz orqali pipeline tuzadi.
"""
import argparse
import base64
import json
import shlex
import subprocess
import sys
import threading
import time
import urllib.request


def http(url, token, method="GET", data=None, timeout=60):
    req = urllib.request.Request(
        url, data=data,
        headers={"X-Farm-Token": token, "Content-Type": "application/json"},
        method=method)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def read_exact(stream, n):
    buf = b""
    while len(buf) < n:
        chunk = stream.read(n - len(buf))
        if not chunk:
            return None
        buf += chunk
    return buf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="o'z inbox nomi")
    ap.add_argument("--next", required=True, help="keyingi node nomi (inbox)")
    ap.add_argument("--server",
                    default="https://ozodbeks-macbook-pro.tail91a187.ts.net")
    ap.add_argument("--token", required=True)
    ap.add_argument("--engine", required=True, help="engine binar yo'li")
    ap.add_argument("--engine-args", default="")
    ap.add_argument("--log", default="/tmp/d70b_engine.log")
    args = ap.parse_args()

    cmd = [args.engine] + shlex.split(args.engine_args)
    print(f"bridge: {args.name} -> {args.next} | {' '.join(cmd)}",
          file=sys.stderr, flush=True)
    logf = open(args.log, "ab")
    eng = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                           stderr=logf, bufsize=0)

    stop = threading.Event()

    def uplink():
        while not stop.is_set():
            hdr = read_exact(eng.stdout, 4)
            if hdr is None:
                break
            ln = int.from_bytes(hdr, "little")
            if ln < 16 or ln > (1 << 26):
                print(f"bridge: buzuq kadr len={ln}", file=sys.stderr, flush=True)
                break
            body = read_exact(eng.stdout, ln)
            if body is None:
                break
            msg = {"b64": base64.b64encode(body).decode()}
            flags = body[3] if len(body) > 3 else 0
            target = "chat" if (flags & 4) else args.next
            try:
                http(args.server + "/api/relay/send", args.token, "POST",
                     json.dumps({"to": target, "msg": msg}).encode())
            except Exception as e:
                print(f"bridge: uplink xato: {e}", file=sys.stderr, flush=True)
                time.sleep(0.5)
        stop.set()

    def downlink():
        while not stop.is_set():
            try:
                r = http(args.server +
                         f"/api/relay/recv?name={args.name}&wait=25",
                         args.token, timeout=40)
            except Exception:
                time.sleep(1)
                continue
            msg = r.get("msg")
            if not msg:
                continue
            body = base64.b64decode(msg["b64"])
            try:
                eng.stdin.write(len(body).to_bytes(4, "little") + body)
                eng.stdin.flush()
            except Exception:
                break
        stop.set()

    t1 = threading.Thread(target=uplink, daemon=True)
    t2 = threading.Thread(target=downlink, daemon=True)
    t1.start()
    t2.start()
    rc = eng.wait()
    stop.set()
    print(f"bridge: engine tugadi rc={rc}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
