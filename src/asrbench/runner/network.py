from __future__ import annotations

import socket
import ssl
import time


def tcp_rtt_ms(host: str, port: int = 443, n: int = 5, timeout: float = 3.0) -> dict:
    samples: list[float] = []
    err = None
    ctx = ssl.create_default_context()
    for _ in range(n):
        t0 = time.perf_counter()
        try:
            with socket.create_connection((host, port), timeout=timeout) as raw:
                with ctx.wrap_socket(raw, server_hostname=host):
                    samples.append((time.perf_counter() - t0) * 1000.0)
        except Exception as exc:  # noqa: BLE001
            err = repr(exc)
    if not samples:
        return {"host": host, "n": 0, "error": err}
    samples.sort()
    return {
        "host": host,
        "n": len(samples),
        "p50_ms": round(samples[len(samples) // 2], 1),
        "min_ms": round(samples[0], 1),
        "max_ms": round(samples[-1], 1),
        "error": err,
    }


def baseline_for_hosts(hosts: list[str]) -> list[dict]:
    seen, out = set(), []
    for h in hosts:
        if h and h not in seen:
            seen.add(h)
            out.append(tcp_rtt_ms(h))
    return out
