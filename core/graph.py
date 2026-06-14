"""
Knowledge Graph Layer — v1.1 (LLM Wiki v2-aligned)

Stores entity relationships as Redis Hash adjacency lists alongside facts.
When a fact mentions entities X and Y together, both X→Y and Y→X edges
are recorded so graph traversal is O(1) per hop.

LLM Wiki v2 additions (v1.1):
  - Typed relationships: edges carry a type (uses, depends_on, contradicts,
    caused, fixed, supersedes, related_to) with confidence and source count.
  - Graph traversal: walk outward through typed edges for impact analysis.
  - Redis key pattern changed from Set to Hash for typed edges:
    mem:graph:{entity_slug}  →  Redis Hash {neighbour_slug: edge_json}
  - Legacy co-occurrence edges preserved as type="related_to".

Slugs are lowercased, whitespace→underscore, non-alnum stripped.
This ensures consistent lookup regardless of capitalisation.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from collections import deque

import redis.asyncio as aioredis
from redis.exceptions import ResponseError

from .utils import decode_json, decode_bytes, force_str

log = logging.getLogger(__name__)

GRAPH_PREFIX = "mem:graph:"

# LLM Wiki v2 typed relationship vocabulary
RELATIONSHIP_TYPES = frozenset({
    "uses",           # A uses B (e.g., "project uses Redis")
    "depends_on",     # A depends on B (stronger than uses)
    "contradicts",    # A contradicts B (opposing claims)
    "caused",         # A caused B (causal link)
    "fixed",          # A fixed B (bug fix)
    "supersedes",     # A supersedes B (newer version replaces older)
    "related_to",     # Default co-occurrence (legacy)
})

# ── Atomic edge upsert via Lua script ─────────────────────────────────────────
# Race condition fix: the old Python read-modify-write pattern lost source_count
# increments under concurrent access (two coroutines both read source_count=N,
# both write N+1 instead of N+2). This Lua script does the read+merge+write
# atomically inside Redis, which is single-threaded for Lua execution.

_UPSERT_EDGE_LUA = """
local key = KEYS[1]
local field = ARGV[1]
local edge_type = ARGV[2]
local confidence = tonumber(ARGV[3])
local now_ms = tonumber(ARGV[4])
local source = ARGV[5]

-- Migrate legacy Set keys to Hash: if key is a Set, convert members to Hash fields
local key_type = redis.call('TYPE', key)
if key_type == 'set' then
    local members = redis.call('SMEMBERS', key)
    redis.call('DEL', key)
    for _, m in ipairs(members) do
        local legacy_edge = {type='related_to', confidence=0.5, source_count=1, last_seen=now_ms, sources={}}
        redis.call('HSET', key, m, cjson.encode(legacy_edge))
    end
end

local raw = redis.call('HGET', key, field)
if raw then
    local ok, edge = pcall(cjson.decode, raw)
    if ok and type(edge) == 'table' then
        edge['source_count'] = (edge['source_count'] or 1) + 1
        edge['confidence'] = math.max(edge['confidence'] or 0.5, confidence)
        edge['last_seen'] = now_ms
        if source and source ~= '' then
            local sources = edge['sources'] or {}
            local found = false
            for i, s in ipairs(sources) do
                if s == source then found = true; break end
            end
            if not found then
                table.insert(sources, source)
                if #sources > 5 then
                    local trimmed = {}
                    for i = #sources - 4, #sources do
                        trimmed[#trimmed+1] = sources[i]
                    end
                    sources = trimmed
                end
                edge['sources'] = sources
            end
        end
        redis.call('HSET', key, field, cjson.encode(edge))
        return 1
    end
end

-- New edge
local new_edge = {
    type = edge_type,
    confidence = confidence,
    source_count = 1,
    last_seen = now_ms,
    sources = {}
}
if source and source ~= '' then
    new_edge['sources'] = {source}
end
redis.call('HSET', key, field, cjson.encode(new_edge))
return 0
"""

_edge_script = None  # cached Script object
_edge_script_lock = asyncio.Lock()  # protects _edge_script initialization


def _slug(entity: str) -> str:
    """Normalise an entity name to a stable Redis key segment.

    Preserves CJK characters (U+4E00-U+9FFF) which were previously stripped
    by ``[^\\w]`` since ``\\w`` does not include Unicode CJK in Python's
    default ASCII mode.
    """
    s = entity.lower().strip()
    s = re.sub(r"\s+", "_", s)
    # Keep ASCII word chars, CJK unified ideographs, and underscores
    s = re.sub(r"[^\w\u4e00-\u9fff\u3400-\u4dbf\u20000-\u2a6df]", "", s)
    return s


def _key(entity: str) -> str:
    return f"{GRAPH_PREFIX}{_slug(entity)}"


async def record_entities(
    r: aioredis.Redis,
    entities: list[str],
    persons: list[str],
    relationship_type: str = "related_to",
    confidence: float = 0.8,
    source: str = "",
) -> None:
    """
    Record co-occurrence edges for all pairs in entities + persons.

    For N items this creates N*(N-1) directed edges (stored as undirected
    by writing both directions). Uses atomic Lua scripts for each edge
    to prevent race conditions during concurrent updates.

    LLM Wiki v2 (v1.1): edges are now typed with confidence and source_count.
    Stored as Redis Hash: {neighbour_slug: edge_json} where edge_json contains
    {type, confidence, source_count, last_seen, sources}.

    If an edge already exists, increments source_count and updates confidence
    to max(existing, new) — same merge logic as add_typed_edge().

    Args:
        r:                Redis client.
        entities:         Entity names extracted from a fact (e.g. ["Redis", "FastAPI"]).
        persons:          Person names extracted from a fact (e.g. ["Bob", "Alice"]).
        relationship_type: One of RELATIONSHIP_TYPES (default: "related_to").
        confidence:       Confidence score for this edge (0-1).
        source:           Source identifier for this edge.
    """
    if relationship_type not in RELATIONSHIP_TYPES:
        relationship_type = "related_to"

    all_ents = list(dict.fromkeys(entities + persons))  # deduplicated, order-preserved
    if len(all_ents) < 2:
        return

    global _edge_script
    if _edge_script is None:
        async with _edge_script_lock:
            if _edge_script is None:
                _edge_script = r.register_script(_UPSERT_EDGE_LUA)

    now_ms = int(time.time() * 1000)

    for i, a in enumerate(all_ents):
        for b in all_ents[i + 1:]:
            slug_a = _slug(a)
            slug_b = _slug(b)
            if slug_a == slug_b:
                continue
            # Atomic upsert for both directions using Lua script
            for key, field in [(_key(a), slug_b), (_key(b), slug_a)]:
                try:
                    await _edge_script(
                        keys=[key],
                        args=[field, relationship_type, str(confidence), str(now_ms), source or ""],
                    )
                except Exception as e:
                    log.warning("record_entities Lua failed for %s/%s: %s", key, field, e)


async def add_typed_edge(
    r: aioredis.Redis,
    source_entity: str,
    target_entity: str,
    relationship_type: str,
    confidence: float = 0.8,
    source: str = "",
    bidirectional: bool = False,
) -> None:
    """
    Add a typed relationship edge between two entities.

    LLM Wiki v2: "Not all connections are equal. 'uses', 'depends on',
    'contradicts', 'caused', 'fixed', 'supersedes' carry different semantic weight."

    If an edge already exists between the same pair, increments source_count
    and updates confidence to max(existing, new).

    Uses atomic Lua script to prevent race conditions during concurrent
    edge updates (same pattern as record_entities).

    Args:
        r:                Redis client.
        source_entity:    Source entity name.
        target_entity:    Target entity name.
        relationship_type: One of RELATIONSHIP_TYPES.
        confidence:       Confidence score for this edge (0-1).
        source:           Source identifier.
        bidirectional:    If True, also add the reverse edge.
    """
    if relationship_type not in RELATIONSHIP_TYPES:
        relationship_type = "related_to"

    slug_s = _slug(source_entity)
    slug_t = _slug(target_entity)
    if slug_s == slug_t:
        return

    global _edge_script
    if _edge_script is None:
        async with _edge_script_lock:
            if _edge_script is None:
                _edge_script = r.register_script(_UPSERT_EDGE_LUA)

    now_ms = int(time.time() * 1000)

    async def _upsert_edge(key: str, field: str) -> None:
        try:
            await _edge_script(
                keys=[key],
                args=[field, relationship_type, str(confidence), str(now_ms), source or ""],
            )
        except Exception as e:
            log.warning("add_typed_edge Lua failed for %s/%s: %s", key, field, e)

    await _upsert_edge(_key(source_entity), slug_t)
    if bidirectional:
        await _upsert_edge(_key(target_entity), slug_s)


async def traverse(
    r: aioredis.Redis,
    entity: str,
    relationship_types: list[str] | None = None,
    max_depth: int = 2,
    max_nodes: int = 50,
) -> list[dict]:
    """
    Walk outward through typed edges for impact analysis.

    LLM Wiki v2: "When someone asks 'what's the impact of upgrading Redis?',
    the LLM shouldn't just keyword-search. It should start at the Redis node,
    walk outward through 'depends on' and 'uses' edges, and find everything
    downstream."

    Args:
        r:                Redis client.
        entity:           Starting entity name.
        relationship_types: Filter to only these edge types (None = all).
        max_depth:        Maximum traversal depth (1-3).
        max_nodes:        Maximum nodes to return.

    Returns:
        List of {entity, depth, edge_type, confidence, path} dicts.
    """
    type_filter = set(relationship_types) if relationship_types else None
    visited: dict[str, dict] = {}  # slug → {depth, edge_type, confidence, path}
    frontier: deque[tuple[str, int, str, float, str]] = deque([(_slug(entity), 0, "start", 1.0, entity)])

    while frontier and len(visited) < max_nodes:
        slug, depth, edge_type, conf, path = frontier.popleft()
        if slug in visited:
            continue
        visited[slug] = {
            "entity": slug,
            "depth": depth,
            "edge_type": edge_type,
            "confidence": conf,
            "path": path,
        }
        if depth >= max_depth:
            continue

        # Get neighbours from hash; legacy keys are Sets (hgetall → WRONGTYPE).
        # Synthesise empty edge payloads so they decode to the default edge type.
        try:
            raw_edges = await r.hgetall(_key_from_slug(slug))
        except ResponseError:
            members_raw = await r.smembers(_key_from_slug(slug))
            raw_edges = {m: b"" for m in members_raw}
        for neighbour_raw, edge_raw in raw_edges.items():
            n_slug = decode_bytes(neighbour_raw)
            if n_slug in visited:
                continue
            try:
                edge = decode_json(edge_raw)
            except Exception:
                edge = {"type": "related_to", "confidence": 0.5}

            e_type = edge.get("type", "related_to")
            if type_filter and e_type not in type_filter:
                continue

            frontier.append((
                n_slug, depth + 1, e_type,
                min(conf, edge.get("confidence", 0.5)),
                f"{path} → {e_type} → {n_slug}",
            ))

    # Remove the starting entity from results
    start_slug = _slug(entity)
    visited.pop(start_slug, None)
    return list(visited.values())


async def get_related(
    r: aioredis.Redis,
    entity: str,
    depth: int = 1,
) -> list[str]:
    """
    Return entity slugs related to *entity* up to *depth* hops.

    Works with both new Hash-based typed edges and legacy Set-based edges.
    depth=1 → direct neighbours only.
    depth=2 → neighbours + their neighbours (one extra hop).

    Returns a flat, deduplicated list of slug strings.
    """
    visited: set[str] = set()
    frontier: set[str] = {_slug(entity)}

    for _ in range(depth):
        next_frontier: set[str] = set()
        for slug in frontier:
            # Try Hash-based typed edges first (v1.1); legacy keys are Sets,
            # and hgetall raises WRONGTYPE on a Set rather than returning {}.
            members: set[str] = set()
            try:
                raw_edges = await r.hgetall(_key_from_slug(slug))
                if raw_edges:
                    members = {decode_bytes(k) for k in raw_edges.keys()}
            except ResponseError:
                # WRONGTYPE: key is a legacy Set, fall back to smembers
                try:
                    members_raw = await r.smembers(_key_from_slug(slug))
                    members = {decode_bytes(m) for m in members_raw}
                except Exception:
                    pass
            except Exception:
                pass
            next_frontier |= members - visited - frontier
        visited |= frontier
        frontier = next_frontier

    # Remove the original entity slug from results
    result = visited | frontier
    result.discard(_slug(entity))
    return sorted(result)


def _key_from_slug(slug: str) -> str:
    return f"{GRAPH_PREFIX}{slug}"


async def entity_recall(
    r: aioredis.Redis,
    entity: str,
    emb,
    k: int = 5,
) -> list[dict]:
    """
    Retrieve facts from the graph neighbourhood of *entity*.

    Steps:
      1. Get related entity slugs (depth=1).
      2. VSIM facts vectorset with the entity embedding.
      3. Post-filter: keep facts that mention any neighbour slug in content.

    Returns up to *k* fact dicts with {content, category, score, attrs}.
    """
    from . import store as mem_store

    try:
        related = await get_related(r, entity, depth=1)
    except Exception as e:
        log.warning("entity_recall: get_related failed for %s: %s", entity, e)
        return []
    if not related:
        return []

    # Broad semantic search
    candidates = await mem_store.knn_search(r, mem_store.FACT_KEY, emb, k=k * 3)

    # Filter to facts that mention at least one related entity in content or attrs
    relevant: list[dict] = []
    for fact in candidates:
        content_lower = fact.get("content", "").lower()
        attrs = fact.get("attrs", {})
        fact_persons  = [p.lower() for p in (attrs.get("persons") or [])]
        fact_entities = [e.lower() for e in (attrs.get("entities") or [])]
        all_fact_ents = set(fact_persons + fact_entities + content_lower.split())

        if any(rel in all_fact_ents for rel in related):
            relevant.append(fact)
            if len(relevant) >= k:
                break

    return relevant


async def graph_stats(r: aioredis.Redis) -> dict:
    """
    Count nodes and edges in the knowledge graph.

    A node = any mem:graph:* key (one per distinct entity slug).
    An edge = one hash field or set member (each real edge stored as two directed members).
    Handles both Hash-based typed edges (v1.1) and legacy Set-based edges (v0.9.0).
    Uses pipeline for batched size lookups (avoids N+1 queries).
    """
    node_keys = []
    async for key in r.scan_iter(f"{GRAPH_PREFIX}*"):
        node_keys.append(key)

    node_count = len(node_keys)
    if not node_keys:
        return {"node_count": 0, "edge_count_directed": 0, "edge_count_undirected": 0}

    # Batch size lookups via pipeline — raise_on_error=False so WRONGTYPE errors
    # on legacy Set keys come back as ResponseError objects instead of raising.
    key_strs = [decode_bytes(key) for key in node_keys]
    pipe = r.pipeline(transaction=False)
    for ks in key_strs:
        pipe.hlen(ks)
    hlens = await pipe.execute(raise_on_error=False)

    edge_count = 0
    fallback_keys = []
    for ks, hlen_val in zip(key_strs, hlens):
        if isinstance(hlen_val, int) and hlen_val > 0:
            edge_count += hlen_val
        else:
            # WRONGTYPE error (Set key) or 0 → fall back to SCARD
            fallback_keys.append(ks)

    if fallback_keys:
        pipe = r.pipeline(transaction=False)
        for ks in fallback_keys:
            pipe.scard(ks)
        scards = await pipe.execute(raise_on_error=False)
        for sc in scards:
            if isinstance(sc, int):
                edge_count += sc

    directed = edge_count
    undirected = directed // 2
    return {
        "node_count": node_count,
        "edge_count_directed": directed,
        "edge_count_undirected": undirected,
        # Dashboard-friendly aliases
        "nodes": node_count,
        "edges": undirected,
        "total_nodes": node_count,
        "total_edges": undirected,
    }


async def find_bridge_nodes(
    r: aioredis.Redis,
    terminal_facts: list[dict],
    query_emb,
    k: int = 3,
) -> list[dict]:
    """
    AriadneMem Steiner tree approximation: find bridge nodes that connect
    pairs of disconnected terminal facts for multi-hop retrieval.

    Algorithm (arXiv:2603.03290 Phase II §3.3):
      1. Collect entity+person sets from each terminal fact.
      2. For each terminal entity, fetch its graph neighbours.
      3. Find candidate facts whose content/attrs overlap with ≥2 different
         terminal entity sets — these are structural bridge nodes.
      4. Return up to k bridge facts, deduplicated against terminal_facts.

    This is deterministic (no LLM call) — replaces iterative LLM planning
    for multi-hop questions with one structural graph pass.

    Returns bridge fact dicts (same shape as knn_search results) annotated
    with {"_bridge": True}.
    """
    from . import store as mem_store

    if len(terminal_facts) < 2:
        return []

    # Build entity sets per terminal fact (up to 5 facts, cap entities per fact)
    terminal_entity_sets: list[set[str]] = []
    all_terminal_entities: set[str] = set()
    for f in terminal_facts[:5]:
        attrs = f.get("attrs", {})
        ents = set(e.lower() for e in (attrs.get("entities") or []))
        pers = set(p.lower() for p in (attrs.get("persons") or []))
        # Also pull significant words from content (4+ char tokens, not stopwords)
        _STOP = {"the", "and", "for", "with", "from", "that", "this", "have",
                 "they", "were", "been", "their", "what", "when", "where"}
        content_tokens = {
            w for w in f.get("content", "").lower().split()
            if len(w) >= 4 and w not in _STOP
        }
        combined = ents | pers | content_tokens
        terminal_entity_sets.append(combined)
        all_terminal_entities |= ents | pers

    if not all_terminal_entities:
        return []

    # Expand via graph neighbourhood: collect neighbour slugs for all entities
    neighbour_tasks = [get_related(r, ent, depth=1) for ent in list(all_terminal_entities)[:6]]
    try:
        neighbour_batches = await asyncio.gather(*neighbour_tasks, return_exceptions=True)
    except Exception:
        return []

    graph_neighbours: set[str] = set()
    for batch in neighbour_batches:
        if isinstance(batch, list):
            graph_neighbours.update(batch)

    # Broad semantic search for bridge candidates
    try:
        candidates = await mem_store.knn_search(r, mem_store.FACT_KEY, query_emb, k=k * 8)
    except Exception:
        return []

    terminal_contents = {f.get("content", "") for f in terminal_facts}
    bridges: list[dict] = []

    for cand in candidates:
        if cand.get("content", "") in terminal_contents:
            continue  # already a terminal — not a bridge

        attrs = cand.get("attrs", {})
        cand_ents  = set(e.lower() for e in (attrs.get("entities") or []))
        cand_pers  = set(p.lower() for p in (attrs.get("persons") or []))
        cand_words = set(cand.get("content", "").lower().split())
        cand_all   = cand_ents | cand_pers | cand_words

        # Graph neighbour boost: entity appeared in neighbourhood expansion
        graph_match = bool(cand_ents & graph_neighbours or cand_pers & graph_neighbours)

        # Count how many distinct terminal entity sets this candidate bridges
        terminal_sets_covered = sum(
            1 for ts in terminal_entity_sets
            if ts & cand_all  # non-empty intersection with this terminal's entities
        )

        # Bridge criterion: connects ≥2 terminal entity sets OR is in graph neighbourhood
        if terminal_sets_covered >= 2 or (graph_match and terminal_sets_covered >= 1):
            bridged = dict(cand)
            bridged["_bridge"] = True
            bridged["_bridge_coverage"] = terminal_sets_covered
            bridges.append(bridged)
            if len(bridges) >= k:
                break

    return bridges


async def get_entity_neighbors_with_counts(
    r: aioredis.Redis,
    entity: str,
) -> list[dict]:
    """
    Return related entities with their edge type and connection counts.

    Works with both Hash-based typed edges (v1.1) and legacy Set-based edges.
    Uses pipeline for batched connection count lookups (avoids N+1 queries).
    """
    # Try Hash-based typed edges first (v1.1); fall back on WRONGTYPE (legacy Sets)
    try:
        raw_edges = await r.hgetall(_key(entity))
    except Exception:
        raw_edges = {}
    if raw_edges:
        # Parse edges first
        parsed: list[tuple[str, dict]] = []
        for neighbour_raw, edge_raw in sorted(raw_edges.items(), key=lambda x: x[0]):
            slug = decode_bytes(neighbour_raw)
            try:
                edge = decode_json(edge_raw)
            except (json.JSONDecodeError, UnicodeDecodeError):
                edge = {"type": "related_to", "confidence": 0.5, "source_count": 1}
            parsed.append((slug, edge))

        # Batch connection count lookups via pipeline. A neighbour key may be a
        # legacy Set, so hlen raises WRONGTYPE; raise_on_error=False keeps the
        # exception in the results list instead of aborting the whole batch.
        pipe = r.pipeline(transaction=False)
        for slug, _ in parsed:
            pipe.hlen(_key_from_slug(slug))
        counts = await pipe.execute(raise_on_error=False)

        results = []
        for (slug, edge), n_edges in zip(parsed, counts):
            if isinstance(n_edges, Exception) or not n_edges:
                # Fallback: try legacy Set-based count
                n_edges = await r.scard(_key_from_slug(slug))
            results.append({
                "entity": slug,
                "connection_count": int(n_edges or 0),
                "edge_type": edge.get("type", "related_to"),
                "confidence": edge.get("confidence", 0.5),
                "source_count": edge.get("source_count", 1),
            })
        return results

    # Fallback: legacy Set-based edges (v0.9.0)
    try:
        related_slugs_raw = await r.smembers(_key(entity))
    except ResponseError:
        related_slugs_raw = set()
    related_slugs = [
        decode_bytes(m) for m in related_slugs_raw
    ]
    # Batch connection count lookups via pipeline
    if related_slugs:
        pipe = r.pipeline(transaction=False)
        for slug in related_slugs:
            pipe.scard(_key_from_slug(slug))
        counts = await pipe.execute()
    else:
        counts = []
    results = []
    for slug, card in zip(sorted(related_slugs), counts):
        results.append({
            "entity": slug,
            "connection_count": int(card or 0),
            "edge_type": "related_to",
            "confidence": 0.5,
            "source_count": 1,
        })
    return results
