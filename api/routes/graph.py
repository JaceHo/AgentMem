"""Graph routes — knowledge graph, typed edges, traversal, confidence."""

import logging

from fastapi import APIRouter

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
    try:
        return await graph_mod.graph_stats(state.redis)
    except Exception as e:
        log.error(f"[graph/stats] {e}")
        return {"nodes": 0, "edges": 0, "total_nodes": 0, "total_edges": 0, "error": str(e)}


@router.get("/graph/nodes")
async def graph_nodes_endpoint(limit: int = 60):
    """Return top N most-connected entities with their edges for graph visualization."""
    r = state.redis
    try:
        prefix = graph_mod.GRAPH_PREFIX
        all_keys = []
        async for key in r.scan_iter(f"{prefix}*", count=500):
            all_keys.append(decode_bytes(key))

        if not all_keys:
            return {"nodes": [], "edges": []}

        pipe = r.pipeline(transaction=False)
        for k in all_keys:
            pipe.scard(k)
        scards = await pipe.execute(raise_on_error=False)

        ranked = sorted(
            [(k, int(sc) if isinstance(sc, int) else 0) for k, sc in zip(all_keys, scards)],
            key=lambda x: x[1], reverse=True
        )[:limit]

        top_keys = [k for k, _ in ranked]
        pipe = r.pipeline(transaction=False)
        for k in top_keys:
            pipe.smembers(k)
        members_list = await pipe.execute(raise_on_error=False)

        nodes_out = []
        edges_out = []
        seen_edges: set[tuple] = set()

        for (k, conn_count), members in zip(ranked, members_list):
            slug = k[len(prefix):]
            label = slug.replace("_", " ")
            nodes_out.append({"id": slug, "label": label, "connections": conn_count})

            if not isinstance(members, (set, list)):
                continue
            for m in members:
                nb = decode_bytes(m)
                pair = (min(slug, nb), max(slug, nb))
                if pair not in seen_edges:
                    seen_edges.add(pair)
                    edges_out.append({"source": slug, "target": nb, "type": "related_to"})

        return {"nodes": nodes_out, "edges": edges_out}
    except Exception as e:
        log.error(f"[graph/nodes] {e}")
        return {"nodes": [], "edges": [], "error": str(e)}


@router.get("/graph/{entity}")
async def graph_neighbors(entity: str):
    """Return entities related to *entity* in the knowledge graph."""
    neighbors = await graph_mod.get_entity_neighbors_with_counts(state.redis, entity)
    return {"entity": entity, "neighbors": neighbors, "count": len(neighbors)}


@router.post("/graph/recall")
async def graph_recall(req: GraphRecallRequest):
    """Retrieve facts from the knowledge graph neighbourhood of an entity."""
    emb = encode(req.entity)
    facts = await graph_mod.entity_recall(state.redis, req.entity, emb, k=req.k)
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
        state.redis, req.entity, req.relationship_types,
        req.max_depth, req.max_nodes,
    )
    return {"entity": req.entity, "nodes": results, "count": len(results)}


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
            return {"error": "not_found", "element_id": element_id}
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
    except Exception as e:
        return {"error": str(e), "element_id": element_id}
