"""Re-embed every memory-tier vectorset with the embedder currently in use.

Why: VADD fixes a vectorset's dimension on first write. Changing the embedding
provider (768 nomic -> 1024 Qwen3) leaves facts/procedures/episodes at the old
width, and every new write then fails with "vector dimension mismatch" — silently
on /store, as a 500 on /admin/delete-facts-by-content (OPEN #109).

What it does, per tier: enumerate element ids (shadow index ∪ VSIM scan), dump
ids+attrs to a JSON file, delete the vectorset, re-embed each element's content,
re-add with the original attrs, and rebuild the shadow index and cached vectors.
The dump is written before anything is deleted; a crash mid-run is recoverable
from that file.

Usage:
    python3 scripts/reembed_vectorsets.py --dump-dir ~/tmp/attic/reembed-$(date +%F)
    python3 scripts/reembed_vectorsets.py --vsets mem:facts --dry-run
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import redis.asyncio as aioredis  # noqa: E402

from core import store  # noqa: E402
from core.embedder import DIMS, embed_signature  # noqa: E402
from core.search import encode_batch  # noqa: E402

DEFAULT_VSETS = [store.EPISODE_KEY, store.FACT_KEY, store.PROC_KEY]
# mem:tools keeps its vectorset but no shadow index or cached-vector hash: the
# capability plane writes attrs and reads them back from VSIM alone.
NO_SHADOW_VSETS = {"mem:tools"}


async def vsim_all(r, vset: str, dim: int, count: int) -> list[str]:
    """Enumerate through VSIM with a zero vector of the set's *own* width.

    core.search.vscan() builds that vector from the global DIMS, so after a
    provider change it cannot even scan the old set. Here the width comes from
    VDIM, which is the whole point: the old set must be readable before it is
    replaced.
    """
    import numpy as np
    zero = np.zeros(dim, dtype=np.float32).tobytes()
    try:
        res = await r.execute_command("VSIM", vset, "FP32", zero, "COUNT", count, "WITHATTRIBS")
    except Exception as exc:  # noqa: BLE001
        print(f"vsim scan failed on {vset}: {exc}", flush=True)
        return []
    ids = []
    for i in range(0, len(res), 2):
        ids.append(res[i] if isinstance(res[i], str) else res[i].decode())
    return ids


async def enumerate_ids(r, vset: str, dim: int, count: int) -> list[str]:
    ids = set(await r.smembers(store.idx_key(vset)))
    ids.update(await vsim_all(r, vset, dim, count))
    return sorted(ids)


async def fetch_attrs(r, vset: str, ids: list[str]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for i in range(0, len(ids), 256):
        chunk = ids[i:i + 256]
        pipe = r.pipeline(transaction=False)
        for eid in chunk:
            pipe.execute_command("VGETATTR", vset, eid)
        for eid, raw in zip(chunk, await pipe.execute()):
            if not raw:
                continue
            try:
                out[eid] = json.loads(raw)
            except Exception:  # noqa: BLE001 — unreadable attrs are dropped, not guessed
                continue
    return out


def embed_text(vset: str, attrs: dict) -> str:
    """The string a tier was originally embedded from, in priority order.

    mem:tools is embedded from "name: description" by /register-tools, so a
    generic content/task/procedure lookup would silently drop every tool.
    """
    if vset in NO_SHADOW_VSETS:
        name = (attrs.get("name") or "").strip()
        desc = (attrs.get("description") or "").strip()
        return f"{name}: {desc}" if (name or desc) else ""
    for field in ("content", "task", "procedure"):
        text = (attrs.get(field) or "").strip()
        if text:
            return text
    return ""


async def rebuild(r, vset: str, *, dump_dir: Path, dry: bool) -> dict:
    before_card = int(await r.execute_command("VCARD", vset) or 0)
    before_dim = int(await r.execute_command("VDIM", vset) or 0)
    ids = await enumerate_ids(r, vset, before_dim, max(before_card, 1000))
    attrs = await fetch_attrs(r, vset, ids)
    missing = len(ids) - len(attrs)

    dump = dump_dir / f"{vset.split(':', 1)[-1]}.json"
    dump.write_text(json.dumps({"vset": vset, "dim": before_dim, "card": before_card,
                                "embed_sig_before": "unknown", "elements": attrs},
                               ensure_ascii=False))
    report = {"vset": vset, "card_before": before_card, "dim_before": before_dim,
              "ids": len(ids), "with_attrs": len(attrs), "missing_attrs": missing,
              "dump": str(dump)}
    if dry:
        return report
    if not attrs:
        report["status"] = "skipped (no elements with attrs)"
        return report

    texts = [embed_text(vset, a) for a in attrs.values()]
    keep = [(eid, a) for (eid, a), t in zip(attrs.items(), texts) if t]
    dropped_empty = len(attrs) - len(keep)
    embs = encode_batch([embed_text(vset, a) for _, a in keep])

    await r.delete(vset)
    shadow = vset not in NO_SHADOW_VSETS
    if shadow:
        await r.delete(store.vec_key(vset))
        await r.delete(store.idx_key(vset))
    for i in range(0, len(keep), 64):
        pipe = r.pipeline(transaction=False)
        for (eid, a), emb in zip(keep[i:i + 64], embs[i:i + 64]):
            pipe.execute_command("VADD", vset, "FP32", store._blob(emb), eid,
                                 "SETATTR", json.dumps(a, ensure_ascii=False))
            if shadow:
                pipe.hset(store.vec_key(vset), eid, store._quantize_vec(emb))
                pipe.sadd(store.idx_key(vset), eid)
        await pipe.execute()

    report.update({
        "card_after": int(await r.execute_command("VCARD", vset) or 0),
        "dim_after": int(await r.execute_command("VDIM", vset) or 0),
        "dropped_no_text": dropped_empty,
        "status": "rebuilt",
    })
    return report


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vsets", default=",".join(DEFAULT_VSETS))
    ap.add_argument("--dump-dir", default=f"/Users/jace/tmp/attic/reembed-{time.strftime('%Y%m%d-%H%M')}")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    dump_dir = Path(os.path.expanduser(a.dump_dir))
    dump_dir.mkdir(parents=True, exist_ok=True)
    r = aioredis.from_url(store.REDIS_URL, decode_responses=True)
    print(f"embedder={embed_signature()} dims={DIMS} dry_run={a.dry_run}", flush=True)
    reports = []
    for vset in [v.strip() for v in a.vsets.split(",") if v.strip()]:
        t0 = time.time()
        rep = await rebuild(r, vset, dump_dir=dump_dir, dry=a.dry_run)
        rep["seconds"] = round(time.time() - t0, 1)
        reports.append(rep)
        print(json.dumps(rep, ensure_ascii=False), flush=True)
    await r.aclose()
    (dump_dir / "report.json").write_text(json.dumps(reports, ensure_ascii=False, indent=1))
    return 0 if all(x.get("status") in ("rebuilt", "skipped (no elements with attrs)") for x in reports) or a.dry_run else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
