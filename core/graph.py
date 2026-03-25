"""
Knowledge Graph Layer — v0.9.0

Stores entity relationships as Redis Set adjacency lists alongside facts.
When a fact mentions entities X and Y together, both X→Y and Y→X edges
are recorded so graph traversal is O(1) per hop.

Redis key pattern:
    mem:graph:{entity_slug}  →  Redis Set of related entity slugs

Slugs are lowercased, whitespace→underscore, non-alnum stripped.
This ensures consistent lookup regardless of capitalisation.
"""

from __future__ import annotations

import re

import redis.asyncio as aioredis

GRAPH_PREFIX = "mem:graph:"


def _slug(entity: str) -> str:
    """Normalise an entity name to a stable Redis key segment.

    Preserves CJK characters (U+4E00–U+9FFF) which were previously stripped
    by [^\w] since \w does not include Unicode CJK in Python's default ASCII mode.
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
) -> None:
    """
    Record co-occurrence edges for all pairs in entities + persons.

    For N items this creates N*(N-1) directed edges (stored as undirected
    by writing both directions). Uses a Redis pipeline for efficiency.

    Args:
        r:        Redis client.
        entities: Entity names extracted from a fact (e.g. ["Redis", "FastAPI"]).
        persons:  Person names extracted from a fact (e.g. ["Bob", "Alice"]).
    """
    all_ents = list(dict.fromkeys(entities + persons))  # deduplicated, order-preserved
    if len(all_ents) < 2:
        return

    pipe = r.pipeline(transaction=False)
    for i, a in enumerate(all_ents):
        for b in all_ents[i + 1:]:
            slug_a = _slug(a)
            slug_b = _slug(b)
            if slug_a == slug_b:
                continue
            # Bidirectional edges
            pipe.sadd(_key(a), slug_b)
            pipe.sadd(_key(b), slug_a)
    await pipe.execute()


async def get_related(
    r: aioredis.Redis,
    entity: str,
    depth: int = 1,
) -> list[str]:
    """
    Return entity slugs related to *entity* up to *depth* hops.

    depth=1 → direct neighbours only (SMEMBERS).
    depth=2 → neighbours + their neighbours (one extra hop).

    Returns a flat, deduplicated list of slug strings.
    """
    visited: set[str] = set()
    frontier: set[str] = {_slug(entity)}

    for _ in range(depth):
        next_frontier: set[str] = set()
        for slug in frontier:
            members_raw = await r.smembers(_key_from_slug(slug))
            members = {m.decode() if isinstance(m, bytes) else m for m in members_raw}
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
    import store as mem_store

    related = await get_related(r, entity, depth=1)
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
    An edge = one set member (each real edge stored as two directed members).
    """
    node_keys = []
    async for key in r.scan_iter(f"{GRAPH_PREFIX}*"):
        node_keys.append(key)

    node_count = len(node_keys)
    edge_count = 0
    for key in node_keys:
        card = await r.scard(key)
        edge_count += card

    # edge_count is sum of directed edges; undirected count = edge_count // 2
    return {
        "node_count": node_count,
        "edge_count_directed": edge_count,
        "edge_count_undirected": edge_count // 2,
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
    import store as mem_store

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
    Return related entities with their fact counts (for /graph/{entity} endpoint).

    Each result: {"entity": slug, "fact_count": N}
    Fact count is estimated by SMEMBERS of that entity's graph key.
    """
    import store as mem_store

    related_slugs_raw = await r.smembers(_key(entity))
    related_slugs = [
        (m.decode() if isinstance(m, bytes) else m) for m in related_slugs_raw
    ]

    results = []
    for slug in sorted(related_slugs):
        # Estimate fact count: how many entities this slug is related to
        # (a rough proxy — the actual count would require a full scan)
        card = await r.scard(_key_from_slug(slug))
        results.append({"entity": slug, "connection_count": int(card or 0)})

    return results
