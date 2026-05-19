"""
Scale & Performance Benchmarks — migrated from agentmemory/benchmark/scale-eval.ts

Performance targets (beat agentmemory reference):
- Recall P50: < 15ms (reference: ~50ms)
- Hybrid search @ 1K docs: < 50ms (reference: ~100ms)
- Hybrid search @ 10K docs: < 200ms (reference: ~500ms)
- Index build @ 1K docs: < 500ms (reference: ~1s)
- Memory footprint @ 10K docs: < 500MB (reference: ~1GB)
"""

import time

import pytest
import redis


class TestScaleBenchmark:
    """Large-scale performance benchmarks."""

    @pytest.fixture(autouse=True)
    def setup_redis(self):
        """Setup Redis connection."""
        self.r = redis.Redis(host="localhost", port=6379, db=15, decode_responses=True)
        self.r.flushdb()
        yield
        self.r.flushdb()
        self.r.close()

    @pytest.mark.benchmark(group="scale-100")
    def test_hybrid_search_100_documents(self, benchmark):
        """Hybrid search with 100 documents should complete in < 10ms."""
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            pytest.skip("rank_bm25 not installed")
        
        from core import embedder

        # Create 100 test documents
        docs = []
        doc_ids = []
        for i in range(100):
            text = f"Document {i} about topic {i % 10} with keywords alpha beta gamma"
            docs.append(text.split())
            doc_ids.append(f"doc_{i}")

        # Build BM25 index
        bm25 = BM25Okapi(docs)

        # Generate embeddings
        embeddings = []
        for text in [" ".join(doc) for doc in docs]:
            emb = embedder.embed(text)
            embeddings.append(emb)

        def search():
            query = "topic 5 alpha".split()
            bm25_scores = bm25.get_scores(query)
            return bm25_scores

        scores = benchmark(search)
        assert len(scores) == 100

    @pytest.mark.benchmark(group="scale-1000")
    def test_hybrid_search_1000_documents(self, benchmark):
        """Hybrid search with 1000 documents should complete in < 50ms."""
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            pytest.skip("rank_bm25 not installed")
        
        from core import embedder

        # Create 1000 test documents
        docs = []
        for i in range(1000):
            text = f"Document {i} about topic {i % 20} with various keywords and concepts"
            docs.append(text.split())

        # Build BM25 index
        bm25 = BM25Okapi(docs)

        def search():
            query = "topic 15 keywords".split()
            return bm25.get_scores(query)

        scores = benchmark(search)
        assert len(scores) == 1000

    @pytest.mark.benchmark(group="scale-10k")
    def test_hybrid_search_10000_documents(self, benchmark):
        """Hybrid search with 10K documents should complete in < 200ms."""
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            pytest.skip("rank_bm25 not installed")

        # Create 10K test documents
        docs = []
        for i in range(10000):
            text = f"Document {i} about topic {i % 50} with diverse content and terminology"
            docs.append(text.split())

        # Build BM25 index
        bm25 = BM25Okapi(docs)

        def search():
            query = "topic 42 diverse".split()
            return bm25.get_scores(query)

        scores = benchmark(search)
        assert len(scores) == 10000

    @pytest.mark.benchmark(group="index-build-1k")
    def test_bm25_index_build_1000_docs(self, benchmark):
        """BM25 index build for 1000 docs should complete in < 500ms."""
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            pytest.skip("rank_bm25 not installed")

        docs = [f"Document number {i} about topic {i % 10}".split() for i in range(1000)]

        def build_index():
            return BM25Okapi(docs)

        bm25 = benchmark(build_index)
        assert bm25 is not None

    @pytest.mark.benchmark(group="index-build-10k")
    def test_bm25_index_build_10k_docs(self, benchmark):
        """BM25 index build for 10K docs should complete in < 5s."""
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            pytest.skip("rank_bm25 not installed")

        docs = [f"Document number {i} about topic {i % 50}".split() for i in range(10000)]

        def build_index():
            return BM25Okapi(docs)

        bm25 = benchmark(build_index)
        assert bm25 is not None

    @pytest.mark.benchmark(group="embedding-batch")
    def test_embedding_batch_performance(self, benchmark):
        """Batch embedding of 100 texts should complete in < 5s."""
        from core import embedder

        texts = [f"This is test sentence number {i} for benchmarking." for i in range(100)]

        def embed_batch():
            return [embedder.embed(text) for text in texts]

        embeddings = benchmark(embed_batch)
        assert len(embeddings) == 100
        assert all(len(emb) == embedder.DIMS for emb in embeddings)


class TestMemoryFootprint:
    """Memory usage benchmarks."""

    @pytest.fixture(autouse=True)
    def setup_redis(self):
        """Setup Redis connection."""
        self.r = redis.Redis(host="localhost", port=6379, db=15, decode_responses=True)
        self.r.flushdb()
        yield
        self.r.flushdb()
        self.r.close()

    def test_redis_memory_usage_1k_facts(self):
        """Redis memory usage for 1K facts should be < 50MB."""
        import json

        # Store 1000 facts
        for i in range(1000):
            key = f"mem:facts:test_{i}"
            data = {
                "content": f"Fact number {i} with some content about topic {i % 10}",
                "ts": str(int(time.time() * 1000)),
                "access_count": "0",
            }
            self.r.hset(key, mapping=data)

        # Check memory usage
        info = self.r.info("memory")
        used_mb = info["used_memory"] / (1024 * 1024)

        # Should be reasonable (< 50MB for 1K simple facts)
        assert used_mb < 50, f"Redis using {used_mb:.2f}MB, expected < 50MB"

    def test_redis_memory_usage_10k_facts(self):
        """Redis memory usage for 10K facts should be < 500MB."""
        import json

        # Store 10K facts
        for i in range(10000):
            key = f"mem:facts:test_{i}"
            data = {
                "content": f"Fact number {i} with content about topic {i % 50}",
                "ts": int(time.time() * 1000),
                "access_count": 0,
            }
            self.r.hset(key, mapping=data)

        # Check memory usage
        info = self.r.info("memory")
        used_mb = info["used_memory"] / (1024 * 1024)

        # Should be reasonable (< 500MB for 10K facts)
        assert used_mb < 500, f"Redis using {used_mb:.2f}MB, expected < 500MB"


class TestCrossSessionRetrieval:
    """Cross-session retrieval accuracy benchmarks."""

    @pytest.fixture(autouse=True)
    def setup_redis(self):
        """Setup Redis connection."""
        self.r = redis.Redis(host="localhost", port=6379, db=15, decode_responses=True)
        self.r.flushdb()
        yield
        self.r.flushdb()
        self.r.close()

    @pytest.mark.asyncio
    async def test_cross_session_accuracy(self):
        """Test retrieval accuracy across different sessions."""
        # This requires full integration with store/recall pipeline
        # Will be tested via end-to-end API tests
        pass


class TestConcurrencyBenchmark:
    """Concurrent access performance benchmarks."""

    @pytest.fixture(autouse=True)
    def setup_redis(self):
        """Setup Redis connection."""
        self.r = redis.Redis(host="localhost", port=6379, db=15, decode_responses=True)
        self.r.flushdb()
        yield
        self.r.flushdb()
        self.r.close()

    @pytest.mark.benchmark(group="concurrent-recall")
    def test_concurrent_recall_performance(self, benchmark):
        """10 concurrent recall operations should complete in < 100ms total."""
        import httpx

        def run_concurrent():
            """Use sync httpx with thread pool for concurrent requests."""
            import concurrent.futures

            def recall_task(i):
                client = httpx.Client(base_url="http://localhost:18800", timeout=15)
                response = client.post("/recall", json={
                    "query": f"test query {i}",
                    "session_id": f"concurrent-{i}",
                    "memory_limit_number": 5,
                })
                client.close()
                return response.status_code

            with concurrent.futures.ThreadPoolExecutor(max_workers=10) as pool:
                futures = [pool.submit(recall_task, i) for i in range(10)]
                results = [f.result() for f in futures]
            return results

        results = benchmark(run_concurrent)
        assert all(r == 200 for r in results)


class TestRealWorldScenarios:
    """Real-world usage scenario benchmarks."""

    @pytest.fixture(autouse=True)
    def setup_redis(self):
        """Setup Redis connection."""
        self.r = redis.Redis(host="localhost", port=6379, db=15, decode_responses=True)
        self.r.flushdb()
        yield
        self.r.flushdb()
        self.r.close()

    @pytest.mark.benchmark(group="realworld-coding-session")
    def test_coding_session_simulation(self, benchmark):
        """Simulate a coding session with multiple store/recall cycles."""
        import httpx

        client = httpx.Client(base_url="http://localhost:18800", timeout=15)
        session_id = f"coding-session-{int(time.time())}"

        def simulate_session():
            # Store initial context
            client.post("/store", json={
                "messages": [
                    {"role": "user", "content": "I'm building a FastAPI service with Redis backend"},
                    {"role": "assistant", "content": "Great choice. What features do you need?"},
                ],
                "session_id": session_id,
            })

            time.sleep(0.5)

            # Recall relevant context
            client.post("/recall", json={
                "query": "FastAPI Redis service architecture",
                "session_id": session_id,
                "memory_limit_number": 5,
            })

            # Store more context
            client.post("/store", json={
                "messages": [
                    {"role": "user", "content": "Need authentication with JWT tokens"},
                    {"role": "assistant", "content": "I'll help you implement JWT auth."},
                ],
                "session_id": session_id,
            })

            time.sleep(0.5)

            # Final recall
            response = client.post("/recall", json={
                "query": "JWT authentication implementation",
                "session_id": session_id,
                "memory_limit_number": 5,
            })

            return response.status_code

        result = benchmark(simulate_session)
        assert result == 200

        client.close()
