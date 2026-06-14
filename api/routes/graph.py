"""Graph routes — knowledge graph, typed edges, traversal, confidence."""

import logging

from fastapi import APIRouter, HTTPException

from api import state
from api.schemas.graph import (
    GraphRecallRequest, TypedEdgeRequest, TraverseRequest, ReinforceRequest,
)
from core import graph as graph_mod
from core import store as mem_store
from core.search import encode
from core.utils import decode_bytes, decode_attrs

log = logging.getLogger("mem")
router = APIRouter(tags=["graph"])


@router.get("/graph/stats")
async def graph_stats_endpoint():
    """Return knowledge graph statistics: node count and edge count."""
    r = state.redis
    if r is None:
        raise HTTPException(status_code=503, detail="Redis not ready")
    try:
        return await graph_mod.graph_stats(r)
    except Exception as e:
        log.error(f"[graph/stats] {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/graph/nodes")
async def graph_nodes_endpoint(limit: int = 60):
    """Return top N most-connected entities with their edges for graph visualization.
    Handles both Hash-based typed edges (v1.1) and legacy Set-based edges (v0.9.0)."""
    limit = max(1, min(limit, 200))
    r = state.redis
    try:
        prefix = graph_mod.GRAPH_PREFIX
        all_keys = []
        async for key in r.scan_iter(f"{prefix}*", count=500):
            all_keys.append(decode_bytes(key))

        if not all_keys:
            return {"nodes": [], "edges": []}

        # Get sizes — try hlen first (v1.1 Hash), fall back to scard (legacy Set)
        pipe = r.pipeline(transaction=False)
        for k in all_keys:
            pipe.hlen(k)
        hlens = await pipe.execute(raise_on_error=False)

        # For any key where hlen failed (WRONGTYPE) or returned 0, try scard
        fallback_idx = [i for i, v in enumerate(hlens) if not isinstance(v, int) or v == 0]
        if fallback_idx:
            pipe = r.pipeline(transaction=False)
            for i in fallback_idx:
                pipe.scard(all_keys[i])
            scards = await pipe.execute(raise_on_error=False)
            for i, sc in zip(fallback_idx, scards):
                if isinstance(sc, int) and sc > 0:
                    hlens[i] = sc

        ranked = sorted(
            [(k, int(v) if isinstance(v, int) else 0) for k, v in zip(all_keys, hlens)],
            key=lambda x: x[1], reverse=True
        )[:limit]

        top_keys = [k for k, _ in ranked]

        # Fetch edges — try hgetall (v1.1 Hash), fall back to smembers (legacy Set)
        pipe = r.pipeline(transaction=False)
        for k in top_keys:
            pipe.hgetall(k)
        hgetall_results = await pipe.execute(raise_on_error=False)

        # For any key where hgetall failed, try smembers
        fallback_idx2 = [i for i, v in enumerate(hgetall_results) if not isinstance(v, (dict, set, list))]
        if fallback_idx2:
            pipe = r.pipeline(transaction=False)
            for i in fallback_idx2:
                pipe.smembers(top_keys[i])
            smembers_results = await pipe.execute(raise_on_error=False)
            for i, sm in zip(fallback_idx2, smembers_results):
                hgetall_results[i] = sm

        import json as _json

        nodes_out = []
        edges_out = []
        seen_edges: set[tuple] = set()

        for (k, conn_count), raw_edges in zip(ranked, hgetall_results):
            slug = k[len(prefix):]
            label = slug.replace("_", " ")
            nodes_out.append({"id": slug, "label": label, "connections": conn_count})

            if isinstance(raw_edges, dict):
                # Hash-based typed edges (v1.1)
                for neighbour_raw, edge_raw in raw_edges.items():
                    nb = decode_bytes(neighbour_raw)
                    try:
                        edge_data = _json.loads(decode_bytes(edge_raw))
                        edge_type = edge_data.get("type", "related_to")
                    except (ValueError, TypeError):
                        edge_type = "related_to"
                    pair = (min(slug, nb), max(slug, nb))
                    if pair not in seen_edges:
                        seen_edges.add(pair)
                        edges_out.append({"source": slug, "target": nb, "type": edge_type})
            elif isinstance(raw_edges, (set, list)):
                # Legacy Set-based edges (v0.9.0)
                for m in raw_edges:
                    nb = decode_bytes(m)
                    pair = (min(slug, nb), max(slug, nb))
                    if pair not in seen_edges:
                        seen_edges.add(pair)
                        edges_out.append({"source": slug, "target": nb, "type": "related_to"})

        return {"nodes": nodes_out, "edges": edges_out}
    except Exception as e:
        log.error(f"[graph/nodes] {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/graph/{entity}")
async def graph_neighbors(entity: str):
    """Return entities related to *entity* in the knowledge graph."""
    neighbors = await graph_mod.get_entity_neighbors_with_counts(state.redis, entity)
    return {"entity": entity, "neighbors": neighbors, "count": len(neighbors)}


@router.post("/graph/recall")
async def graph_recall(req: GraphRecallRequest):
    """Retrieve facts from the knowledge graph neighbourhood of an entity."""
    emb = encode(req.entity)
    facts = await graph_mod.entity_recall(state.redis, req.entity, emb, k=req.max_facts)
    return {
        "entity": req.entity,
        "facts": [{"content": f["content"], "score": f.get("score", 0.0)} for f in facts],
        "count": len(facts),
    }


@router.post("/graph/edge")
async def add_graph_edge(req: TypedEdgeRequest):
    """Add a typed relationship edge between two entities."""
    await graph_mod.add_typed_edge(
        state.redis, req.source_entity, req.target_entity,
        req.relationship_type, req.confidence, req.source, req.bidirectional,
    )
    return {
        "status": "ok", "source": req.source_entity, "target": req.target_entity,
        "type": req.relationship_type,
    }


@router.post("/graph/traverse")
async def graph_traverse(req: TraverseRequest):
    """Walk outward through typed edges for impact analysis."""
    results = await graph_mod.traverse(
        state.redis, req.start_entity,
        [req.relation] if req.relation else None,
        req.max_depth, req.max_nodes,
    )
    return {"entity": req.start_entity, "nodes": results, "count": len(results)}


@router.post("/facts/{element_id}/reinforce")
async def reinforce_fact_endpoint(element_id: str, req: ReinforceRequest):
    """Reinforce a fact — increment source_count, reset Ebbinghaus decay."""
    ok = await mem_store.reinforce_fact(state.redis, element_id, req.source or None)
    if not ok:
        return {"status": "not_found", "element_id": element_id}
    return {"status": "ok", "element_id": element_id}


@router.get("/facts/{element_id}/confidence")
async def get_fact_confidence(element_id: str):
    """Get effective confidence for a fact (Ebbinghaus decay applied)."""
    r = state.redis
    try:
        raw = await r.execute_command("VGETATTR", mem_store.FACT_KEY, element_id)
        if not raw:
            raise HTTPException(status_code=404, detail="fact not found")
        attrs = decode_attrs(raw)
        eff = mem_store.confidence_decay(attrs)
        return {
            "element_id": element_id,
            "base_confidence": attrs.get("confidence", 0.8),
            "effective_confidence": eff,
            "source_count": attrs.get("source_count", 1),
            "last_confirmed_ts": attrs.get("last_confirmed_ts", 0),
            "category": attrs.get("category", ""),
            "version": attrs.get("version", 1),
            "superseded_by": attrs.get("superseded_by", ""),
            "superseded_reason": attrs.get("superseded_reason", ""),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
