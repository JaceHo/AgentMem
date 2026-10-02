"""core/kb.py — Knowledge-base plane: documents in, digests + contextual chunks out.

Progressive-disclosure tiers (SOTA: Anthropic contextual retrieval + RAPTOR +
agent-skills progressive disclosure):
  Tier 0  index    — one line per document, always-in-context cheap
  Tier 1  digest   — per-document summary + section map, loadable
  Tier 2  original — contextual chunks via hybrid vector+BM25 (wRRF), plus
                     the on-disk original path

Originals stay on disk; Redis holds only chunk text + embeddings + metadata,
keyed by content hash for incremental re-ingest.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .search import BM25Index, encode, encode_batch
from .store import knn_search

log = logging.getLogger("agentmem.kb")

KB_DOC_KEY    = "kb:docs"        # hash doc_id → meta json
KB_PATH_KEY   = "kb:paths"       # hash abs path → doc_id
KB_CHUNK_KEY  = "kb:chunks"      # vectorset of chunk embeddings
KB_DOC_CHUNKS = "kb:doc_chunks"  # hash doc_id → json [element ids]
KB_TEXT_KEY   = "kb:chunk_text"  # hash element id → chunk content (BM25 warm)

CHUNK_CHARS = 2400
CHUNK_OVERLAP = 300
ENRICH_MAX_CHUNKS = 40          # LLM enrichment cap per document (cost guard)
HOME = os.path.expanduser("~")
TEXT_EXTS = {".md", ".markdown", ".txt"}
EXT_PARSER_NEEDED = {".pdf": "pymupdf (fitz)", ".epub": "ebooklib"}


class KBError(Exception):
    pass


# ── Parsing ───────────────────────────────────────────────────────────────────


def _strip_html(raw: str) -> str:
    text = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", raw, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<h([1-6])[^>]*>(.*?)</h\1>", lambda m: "\n\n# " + re.sub(r"<[^>]+>", "", m.group(2)).strip() + "\n", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<[^>]+>", "\n", text)
    text = re.sub(r"&nbsp;|&amp;|&lt;|&gt;", lambda m: {"&nbsp;": " ", "&amp;": "&", "&lt;": "<", "&gt;": ">"}[m.group(0)], text)
    return re.sub(r"\n{3,}", "\n\n", text)


def parse_file(path: str) -> Dict[str, Any]:
    """Parse one file into {title, sections: [{title, text}]}. Raises KBError."""
    p = Path(path)
    ext = p.suffix.lower()
    try:
        raw = p.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise KBError(f"unreadable: {exc}") from exc

    if ext == ".pdf":
        try:
            import fitz  # pymupdf
        except ImportError as exc:
            raise KBError(f".pdf needs {EXT_PARSER_NEEDED['.pdf']}") from exc
        doc = fitz.open(path)
        raw = "\n\n".join(page.get_text() for page in doc)
        doc.close()
    elif ext == ".epub":
        try:
            import ebooklib
            from ebooklib import epub
            from bs4 import BeautifulSoup
        except ImportError as exc:
            raise KBError(f".epub needs {EXT_PARSER_NEEDED['.epub']}") from exc
        book = epub.read_epub(path)
        parts = []
        for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
            parts.append(_strip_html(item.get_content().decode("utf-8", errors="replace")))
        raw = "\n\n".join(parts)

    if ext in (".html", ".htm"):
        raw = _strip_html(raw)

    title = p.stem.replace("_", " ").replace("-", " ").strip()
    m = re.search(r"^#\s+(.+)$", raw, re.MULTILINE)
    if m:
        title = m.group(1).strip()

    sections: List[Dict[str, str]] = []
    if ext in TEXT_EXTS and raw.count("\n# ") + raw.startswith("# ") >= 1:
        cur_title, buf = "Overview", []
        for line in raw.split("\n"):
            hm = re.match(r"^(#{1,3})\s+(.+)$", line)
            if hm:
                if buf and any(b.strip() for b in buf):
                    sections.append({"title": cur_title, "text": "\n".join(buf).strip()})
                cur_title, buf = hm.group(2).strip(), []
            else:
                buf.append(line)
        if buf and any(b.strip() for b in buf):
            sections.append({"title": cur_title, "text": "\n".join(buf).strip()})
    if not sections:
        text = raw.strip()
        if text:
            sections = [{"title": title, "text": text}]
    if not sections:
        raise KBError("no extractable text")
    return {"title": title, "sections": sections}


def chunk_sections(sections: List[Dict[str, str]],
                   chunk_chars: int = CHUNK_CHARS,
                   overlap: int = CHUNK_OVERLAP) -> List[Dict[str, Any]]:
    """Structural chunking: respect section borders, slide a window inside long ones."""
    chunks: List[Dict[str, Any]] = []
    for sec in sections:
        text = sec["text"]
        start = 0
        while start < len(text):
            piece = text[start:start + chunk_chars].strip()
            if len(piece) > 40:
                chunks.append({
                    "seq": len(chunks),
                    "section": sec["title"],
                    "text": piece,
                })
            if start + chunk_chars >= len(text):
                break
            start += chunk_chars - overlap
    return chunks


# ── LLM helpers (aiserv; every call degrades gracefully) ─────────────────────


async def _llm(system: str, prompt: str, timeout: float = 45.0) -> str:
    """One aiserv chat call. Returns "" on any failure — callers fall back."""
    import httpx

    from .extractor import AISERV_BASE, _load_key

    try:
        async with httpx.AsyncClient(trust_env=False, timeout=timeout) as client:
            resp = await client.post(
                f"{AISERV_BASE}/v1/chat/completions",
                headers={"Authorization": f"Bearer {_load_key()}",
                         "x-api-key": _load_key(), "Content-Type": "application/json"},
                json={"model": "aiserv/fast", "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ], "max_tokens": 400, "temperature": 0.2},
            )
            resp.raise_for_status()
            return str(resp.json()["choices"][0]["message"]["content"] or "")
    except Exception as exc:  # noqa: BLE001 — LLM is an enhancement, never a dependency
        log.warning("kb llm call failed (%s) — using extractive fallback", exc)
        return ""


def _fallback_desc(title: str, sections: List[Dict[str, str]]) -> str:
    first = sections[0]["text"][:200].replace("\n", " ").strip()
    return f"{title}: {first[:140]}…"


def _fallback_digest(sections: List[Dict[str, str]]) -> str:
    lines = ["Section map + opening lines (LLM digest unavailable):"]
    for sec in sections[:12]:
        opening = " ".join(sec["text"].split())[:120]
        lines.append(f"- {sec['title']}: {opening}")
    return "\n".join(lines)


async def make_digest(title: str, sections: List[Dict[str, str]]) -> Tuple[str, str]:
    corpus = "\n\n".join(f"## {s['title']}\n{s['text']}" for s in sections)[:6000]
    out = await _llm(
        "You condense documents for an agent knowledge base. Reply with JSON only.",
        f'Document "{title}". Produce JSON: {{"description": "one sentence, <=140 chars", '
        f'"digest": "markdown summary, <=15 lines, keep key facts/numbers"}}.\n\n{corpus}',
    )
    if out:
        m = re.search(r"\{.*\}", out, re.DOTALL)
        if m:
            try:
                obj = json.loads(m.group(0))
                return str(obj.get("description", ""))[:200], str(obj.get("digest", ""))[:2400]
            except json.JSONDecodeError:
                pass
    return _fallback_desc(title, sections), _fallback_digest(sections)


async def enrich_chunk(title: str, section: str, text: str) -> str:
    """Contextual retrieval (Anthropic): situate the chunk within the document."""
    blurb = await _llm(
        "You write chunk context for retrieval. Reply with ONE sentence (<=40 words), "
        "no preamble, situating the chunk inside its document.",
        f'Document "{title}", section "{section}". Chunk:\n"""\n{text[:1200]}\n"""',
        timeout=30,
    )
    blurb = blurb.strip().strip('"')
    return f"[{title} › {section}] {blurb}\n\n{text}" if blurb else f"[{title} › {section}]\n\n{text}"


# ── Store ─────────────────────────────────────────────────────────────────────


def _blob(arr: np.ndarray) -> bytes:
    return np.asarray(arr, dtype=np.float32).tobytes()


class KBStore:
    """Document store over Redis vectorsets + its own BM25 index."""

    def __init__(self):
        self.bm25 = BM25Index()

    async def _seed(self, r) -> None:
        try:
            await r.execute_command("VADD", KB_CHUNK_KEY, "FP32",
                                    _blob(encode("__seed__")), "__seed__",
                                    "SETATTR", json.dumps({"_seed": True}))
        except Exception:
            pass  # already seeded (or dims differ) — non-fatal

    async def ingest_path(self, r, path: str, *, enrich: bool = True,
                          force: bool = False, library: str = "") -> Dict[str, Any]:
        target = Path(path).expanduser().resolve()
        if not str(target).startswith(HOME):
            raise KBError(f"path must be under {HOME}")
        if not target.exists():
            raise KBError(f"not found: {target}")

        files: List[Path] = []
        if target.is_dir():
            for p in sorted(target.rglob("*")):
                if p.is_file() and (p.suffix.lower() in TEXT_EXTS or p.suffix.lower() in EXT_PARSER_NEEDED):
                    files.append(p)
        else:
            files = [target]

        report: Dict[str, Any] = {"ingested": [], "skipped": [], "failed": []}
        for f in files:
            try:
                out = await self._ingest_file(r, f, enrich=enrich, force=force, library=library)
                (report["ingested"] if out.get("status") == "ingested" else report["skipped"]).append(out)
            except KBError as exc:
                report["failed"].append({"path": str(f), "error": str(exc)})
            except Exception as exc:  # noqa: BLE001
                log.exception("kb ingest crashed for %s", f)
                report["failed"].append({"path": str(f), "error": f"{type(exc).__name__}: {exc}"})
        return report

    async def _ingest_file(self, r, f: Path, *, enrich: bool, force: bool, library: str) -> Dict[str, Any]:
        abspath = str(f)
        raw_bytes = f.read_bytes()
        sha = hashlib.sha1(raw_bytes).hexdigest()

        old_doc_id = await r.hget(KB_PATH_KEY, abspath)
        if old_doc_id and not force:
            old_meta = await self._meta(r, old_doc_id)
            if old_meta and old_meta.get("sha") == sha:
                return {"path": abspath, "doc": old_doc_id, "status": "unchanged"}

        parsed = parse_file(abspath)
        chunks = chunk_sections(parsed["sections"])
        if not chunks:
            raise KBError("no chunks after chunking")
        title = parsed["title"]

        doc_id = old_doc_id or ("kb_" + hashlib.sha1(abspath.encode()).hexdigest()[:12])
        if old_doc_id:
            await self.remove_doc(r, old_doc_id)

        # Contextual enrichment — cap LLM calls per document (cost guard).
        # Per-chunk failures degrade to a plain header prefix; ingest never dies.
        enriched = 0
        if enrich:
            for c in chunks[:ENRICH_MAX_CHUNKS]:
                try:
                    c["text"] = await enrich_chunk(title, c["section"], c["text"])
                    enriched += 1
                except Exception as exc:  # noqa: BLE001
                    log.warning("enrich failed (%s › %s): %s", title, c["section"], exc)
                    c["text"] = f"[{title} › {c['section']}]\n\n{c['text']}"

        embeddings = encode_batch([c["text"] for c in chunks])
        await self._seed(r)
        elem_ids = []
        for c, emb in zip(chunks, embeddings):
            elem = f"{doc_id}:{c['seq']:04d}"
            attrs = {"content": c["text"], "doc": doc_id, "title": title, "seq": c["seq"],
                     "section": c["section"], "path": abspath, "library": library,
                     "enriched": c["seq"] < enriched}
            await r.execute_command("VADD", KB_CHUNK_KEY, "FP32", _blob(emb), elem,
                                    "SETATTR", json.dumps(attrs, ensure_ascii=False))
            await r.hset(KB_TEXT_KEY, elem, c["text"])
            await self.bm25.add(elem, c["text"], attrs)
            elem_ids.append(elem)

        try:
            desc, digest = await make_digest(title, parsed["sections"])
        except Exception as exc:  # noqa: BLE001 — digest is an enhancement
            log.warning("digest failed for %s: %s", title, exc)
            desc, digest = _fallback_desc(title, parsed["sections"]), _fallback_digest(parsed["sections"])
        meta = {
            "id": doc_id, "title": title, "path": abspath, "sha": sha,
            "size": len(raw_bytes), "mtime": int(f.stat().st_mtime),
            "description": desc, "digest": digest,
            "sections": [s["title"] for s in parsed["sections"]],
            "chunks": len(chunks), "enriched_chunks": enriched,
            "library": library, "ingested_at": time.time(),
        }
        await r.hset(KB_DOC_KEY, doc_id, json.dumps(meta, ensure_ascii=False))
        await r.hset(KB_PATH_KEY, abspath, doc_id)
        await r.hset(KB_DOC_CHUNKS, doc_id, json.dumps(elem_ids))
        return {"path": abspath, "doc": doc_id, "status": "ingested",
                "title": title, "chunks": len(chunks), "enriched": enriched}

    async def _meta(self, r, doc_id: str) -> Optional[Dict[str, Any]]:
        raw = await r.hget(KB_DOC_KEY, doc_id)
        return json.loads(raw) if raw else None

    async def remove_doc(self, r, doc_id: str) -> bool:
        elems_raw = await r.hget(KB_DOC_CHUNKS, doc_id)
        if elems_raw:
            for elem in json.loads(elems_raw):
                try:
                    await r.execute_command("VREM", KB_CHUNK_KEY, elem)
                except Exception:
                    pass
                await r.hdel(KB_TEXT_KEY, elem)
            await self.bm25.remove(set(json.loads(elems_raw)))
        meta = await self._meta(r, doc_id)
        existed = meta is not None or elems_raw is not None
        await r.hdel(KB_DOC_KEY, doc_id)
        await r.hdel(KB_DOC_CHUNKS, doc_id)
        if meta and meta.get("path"):
            await r.hdel(KB_PATH_KEY, meta["path"])
        return existed

    async def index(self, r) -> List[Dict[str, Any]]:
        """Tier 0 — one line per document."""
        metas = [json.loads(v) for v in (await r.hvals(KB_DOC_KEY))]
        metas.sort(key=lambda m: m.get("ingested_at", 0), reverse=True)
        return [{"id": m["id"], "title": m["title"], "description": m.get("description", ""),
                 "sections": len(m.get("sections", [])), "chunks": m.get("chunks", 0),
                 "path": m.get("path", ""), "library": m.get("library", "")} for m in metas]

    async def digest(self, r, doc_id: str) -> Optional[Dict[str, Any]]:
        meta = await self._meta(r, doc_id)
        return meta

    async def search(self, r, query: str, k: int = 6, library: str = "") -> List[Dict[str, Any]]:
        """Tier 2 — hybrid vector + BM25, fused with weighted RRF."""
        k = max(1, min(k, 25))
        vec = encode(query)
        filter_expr = f'.library == "{library}"' if library else None
        vres = await knn_search(r, KB_CHUNK_KEY, vec, k * 4, filter_expr=filter_expr)
        bres = await self.bm25.search(query, k * 4)

        fused: Dict[str, Dict[str, Any]] = {}
        for rank, item in enumerate(vres):
            elem = item["_element"]
            fused[elem] = {"attrs": item["attrs"], "score": 1.0 / (60 + rank)}
        for rank, item in enumerate(bres):
            elem = item.get("uid") or (item.get("attrs") or {}).get("_uid", "")
            if not elem:
                continue
            entry = fused.setdefault(elem, {"attrs": item.get("attrs") or {"content": item.get("content", "")}, "score": 0.0})
            entry["score"] += 0.7 / (60 + rank)

        out = []
        for elem, entry in sorted(fused.items(), key=lambda kv: kv[1]["score"], reverse=True)[:k]:
            a = entry["attrs"]
            out.append({"doc": a.get("doc", ""), "title": a.get("title", ""),
                        "section": a.get("section", ""), "seq": a.get("seq", 0),
                        "path": a.get("path", ""), "text": a.get("content", ""),
                        "enriched": a.get("enriched", False), "score": round(entry["score"], 5)})
        return out

    async def warm(self, r) -> int:
        """Rebuild the in-RAM BM25 index from Redis after a service restart."""
        elems = await r.hgetall(KB_TEXT_KEY)
        n = 0
        for elem, content in elems.items():
            attrs_raw = await r.hget(KB_DOC_KEY, elem.split(":")[0])
            meta = json.loads(attrs_raw) if attrs_raw else {}
            await self.bm25.add(elem, content, {"content": content, "doc": elem.split(":")[0],
                                                "title": meta.get("title", ""), "path": meta.get("path", "")})
            n += 1
        return n


_kb: Optional[KBStore] = None


def get_kb() -> KBStore:
    global _kb
    if _kb is None:
        _kb = KBStore()
    return _kb
