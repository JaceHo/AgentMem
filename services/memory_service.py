"""
Memory Service - Business logic for memory CRUD operations.

Extracted from main.py route handlers to provide clean separation between
API layer and business logic. This enables:
- Unit testing without HTTP framework
- Reuse across different API endpoints
- Clear dependency injection boundaries
"""

import logging
from typing import Any

from config.settings import settings
from exceptions import EpisodeStorageError, FactStorageError
from utils.text_processing import (
    flatten_message_content,
    is_injected_system_content,
    is_trivial,
    contains_secret,
    strip_platform_noise,
)

import core.persona as persona_mod


class MemoryService:
    """Service for managing memory storage operations.
    
    Handles episode and fact storage with proper error handling,
    validation, and lifecycle management.
    """
    
    def __init__(self, redis_client, embedder, extractor, store_module, bm25_index=None, graph_module=None):
        """Initialize memory service with dependencies.
        
        Args:
            redis_client: Redis connection
            embedder: Embedding model wrapper
            extractor: Fact extraction module
            store_module: Core store module (mem_store)
            bm25_index: Optional BM25 index for hybrid retrieval updates
            graph_module: Optional knowledge graph module
        """
        self._redis = redis_client
        self._embedder = embedder
        self._extractor = extractor
        self._store = store_module
        self._bm25 = bm25_index
        self._graph = graph_module
        self._log = logging.getLogger("mem")
    
    async def store_episode(
        self,
        session_id: str,
        messages: list[dict],
        metadata: dict | None = None,
    ) -> str:
        """Store conversation episode in Tier 2 long-term memory.
        
        Args:
            session_id: Unique session identifier
            messages: List of message dicts with role/content
            metadata: Optional metadata (timestamp, tags, etc.)
            
        Returns:
            Episode ID (ULID string)
            
        Raises:
            EpisodeStorageError: If storage fails
        """
        try:
            if not session_id:
                raise EpisodeStorageError("session_id is required")
            if not messages:
                raise EpisodeStorageError("messages cannot be empty")

            cleaned = []
            for msg in messages:
                role = str(msg.get("role", "")).strip()
                if not role:
                    continue
                text = flatten_message_content(msg.get("content", ""))
                text = strip_platform_noise(text)
                if not text or is_injected_system_content(text):
                    continue
                cleaned.append({"role": role, "content": text})

            if not cleaned:
                raise EpisodeStorageError("No valid messages to store")

            turn_text = "\n".join(f"{item['role']}: {item['content']}" for item in cleaned)
            if not turn_text.strip():
                raise EpisodeStorageError("Flattened messages contain no storeable text")

            embedding = self._embedder.encode(turn_text[:500])
            prev_episode_id = await self._store.get_last_episode_id(self._redis, session_id)
            episode_id = await self._store.save_episode(
                self._redis,
                session_id,
                turn_text[:2000],
                embedding,
                prev_episode_id=prev_episode_id,
            )

            if prev_episode_id and prev_episode_id != episode_id:
                await self._store.update_episode_next_id(self._redis, prev_episode_id, episode_id)
            await self._store.set_last_episode_id(self._redis, session_id, episode_id)

            raw_messages = [
                {"role": item["role"], "content": item["content"]}
                for item in cleaned
            ]

            facts = await self._extractor.extract_hybrid(raw_messages, turn_text)
            stored_fact_ids: list[str] = []

            for fact in facts:
                if not getattr(fact, "content", "") or contains_secret(fact.content):
                    continue

                fact_embedding = self._embedder.encode(fact.content)
                similar = await self._store.knn_search(
                    self._redis,
                    self._store.FACT_KEY,
                    fact_embedding,
                    1,
                )
                if similar and similar[0].get("score", 0.0) > settings.dedup_similarity_threshold:
                    continue

                fact_id = await self._store.save_fact(
                    self._redis,
                    fact.content,
                    fact.category or "general",
                    fact.confidence,
                    fact_embedding,
                    language="en",
                    domain="general",
                    keywords=fact.keywords,
                    persons=fact.persons,
                    entities=fact.entities,
                    importance=fact.importance,
                    topic=fact.topic,
                    location=fact.location,
                    source_episode_id=episode_id,
                    triple_s=fact.triple_s,
                    triple_p=fact.triple_p,
                    triple_o=fact.triple_o,
                )
                stored_fact_ids.append(fact_id)

                if self._bm25:
                    attrs = {
                        "content": fact.content,
                        "category": fact.category or "general",
                        "language": "en",
                        "domain": "general",
                        "keywords": fact.keywords or [],
                        "importance": fact.importance,
                    }
                    if fact.triple_s and fact.triple_p and fact.triple_o:
                        attrs["triple_str"] = f"{fact.triple_s} | {fact.triple_p} | {fact.triple_o}"
                    await self._bm25.add(fact_id, fact.content, attrs)

                if fact.category:
                    await persona_mod.update(self._redis, fact.category, fact.content)

                if self._graph and (fact.persons or fact.entities):
                    await self._graph.record_entities(
                        self._redis,
                        fact.entities or [],
                        fact.persons or [],
                    )

                if fact.category == "procedure":
                    try:
                        proc_embedding = self._embedder.encode(fact.content)
                        await self._store.save_procedure(
                            self._redis,
                            task=fact.content,
                            procedure=fact.content,
                            embedding=proc_embedding,
                            tools_used=[],
                            domain="general",
                            language="en",
                        )
                    except Exception:
                        self._log.debug("Failed to store procedure fact, continuing")

            await self._store.set_session_context(self._redis, session_id, turn_text[:1500])

            return episode_id
        except EpisodeStorageError:
            raise
        except Exception as e:
            self._log.exception("[memory_service] failed to store episode")
            raise EpisodeStorageError(
                message=f"Failed to store episode: {e}",
                session_id=session_id,
                cause=e,
            )

    async def store_facts(
        self,
        session_id: str,
        facts: list[dict],
        source_episode_id: str | None = None,
    ) -> list[str]:
        """Store extracted facts in semantic memory.

        Args:
            session_id: Session that generated these facts
            facts: List of fact dicts with text/category/importance
            source_episode_id: Episode that produced these facts

        Returns:
            List of fact IDs

        Raises:
            FactStorageError: If any fact fails to store
        """
        stored_ids = []
        errors: list[tuple[int, str]] = []

        for i, fact in enumerate(facts):
            try:
                if "text" not in fact:
                    raise ValueError(f"Fact #{i} missing 'text' field")

                fact_id = await self._store.save_fact(
                    self._redis,
                    fact["text"],
                    fact.get("category", "general"),
                    fact.get("importance", 0.5),
                    self._embedder.encode(fact["text"]),
                    source_episode_id=source_episode_id,
                )
                stored_ids.append(fact_id)
            except Exception as e:
                errors.append((i, str(e)))

        if errors:
            for idx, err in errors:
                self._log.warning("[memory_service] fact #%d failed: %s", idx, err)

        if not stored_ids and errors:
            raise FactStorageError(
                message=f"All {len(errors)} facts failed to store",
                session_id=session_id,
            )

        return stored_ids
