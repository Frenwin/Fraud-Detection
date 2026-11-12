"""E11 — API load test: throughput and tail latency vs. concurrency (client-side, incl. HTTP + JSON).

    FRAUDDET_SINK=off python -m frauddet.serving.api      # terminal 1
    python -m frauddet.loadtest                            # terminal 2
"""
from __future__ import annotations

import argparse
import asyncio
import time

import httpx
import numpy as np
import pandas as pd

from . import viz
from .config import get_paths, load_config


async def _run_level(client, payloads, concurrency: int, endpoint: str = "/score"):
    lat, it = [], iter(payloads)
    lock = asyncio.Lock()

    async def worker():
        while True:
            async with lock:
                p = next(it, None)
            if p is None:
                return
            t0 = time.perf_counter()
            r = await client.post(endpoint, json=p)
            r.raise_for_status()
            lat.append((time.perf_counter() - t0) * 1e3)

    t0 = time.perf_counter()
    await asyncio.gather(*[worker() for _ in range(concurrency)])
    wall = time.perf_counter() - t0
    a = np.array(lat)
    return {"endpoint": endpoint, "concurrency": concurrency, "requests": len(a), "throughput_rps": len(a) / wall,
            "p50_ms": float(np.percentile(a, 50)), "p95_ms": float(np.percentile(a, 95)),
            "p99_ms": float(np.percentile(a, 99))}


async def _main(api, levels, n):
    cfg = load_config()
    raw = pd.read_parquet(get_paths(cfg).raw / "transactions.parquet")
    df = raw[raw.day >= cfg["split"]["test"][0]].head(2 * n * len(levels) + 500)
    payloads = [{"tx_id": int(r.tx_id), "ts": r.ts.strftime("%Y-%m-%dT%H:%M:%S.%f"), "customer_id": int(r.customer_id),
                 "terminal_id": int(r.terminal_id), "device_id": int(r.device_id), "channel": r.channel,
                 "category": r.category, "amount": float(r.amount), "lat": float(r.tx_lat), "lon": float(r.tx_lon)}
                for r in df.itertuples(index=False)]
    rows = []
    limits = httpx.Limits(max_connections=max(levels), max_keepalive_connections=max(levels))
    async with httpx.AsyncClient(base_url=api, timeout=30, limits=limits) as client:
        await _run_level(client, payloads[:500], 4)  # warm-up
        i = 0
        for ep in ("/score", "/score/async"):
            for c in levels:
                await client.post("/metrics/reset")
                r = await _run_level(client, payloads[500 + i * n: 500 + (i + 1) * n], c, ep)
                r["server_p99_ms"] = (await client.get("/metrics")).json()["latency_ms"]["p99"]
                i += 1
                rows.append(r)
                print(f"[loadtest] {ep:<18} c={c:<3} {r['throughput_rps']:>7.0f} req/s | p50={r['p50_ms']:.2f} "
                      f"p95={r['p95_ms']:.2f} p99={r['p99_ms']:.2f} ms | server-side p99={r['server_p99_ms']:.2f} ms")
    return pd.DataFrame(rows)


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default=f"http://{cfg['serving']['host']}:{cfg['serving']['port']}")
    ap.add_argument("--levels", default="1,4,16,64")
    ap.add_argument("--n", type=int, default=3000, help="requests per concurrency level")
    a = ap.parse_args()
    res = asyncio.run(_main(a.api, [int(x) for x in a.levels.split(",")], a.n))
    P = get_paths(cfg)
    res.to_csv(P.results / "e11_loadtest.csv", index=False)
    fig, (a1, a2) = viz.plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, col, title in ((a1, "throughput_rps", "Throughput (req/s)"), (a2, "p99_ms", "Client p99 latency (ms, log)")):
        for k, (ep, lab) in enumerate((("/score", "sync def (threadpool)"), ("/score/async", "async def (event loop)"))):
            d = res[res.endpoint == ep]
            ax.plot(d.concurrency, d[col], marker="o", color=viz.SERIES[k], markeredgecolor=viz.SURFACE,
                    markeredgewidth=1.5, label=lab)
        ax.set_xscale("log", base=2); ax.set_xticks(levels_ := sorted(res.concurrency.unique()), levels_)
        ax.set_xlabel("concurrent clients"); ax.set_title(title)
    a2.set_yscale("log"); a1.legend(loc="lower left")
    fig.suptitle("E11 · API load test (client-side, incl. HTTP + JSON)", x=0.01, ha="left", fontsize=12,
                 fontweight="semibold")
    viz.save(fig, P.figures / "e11_loadtest.png")


if __name__ == "__main__":
    main()
