# AgentMem Test Suite

Comprehensive test suite for AgentMem, migrated and enhanced from the [agentmemory](https://github.com/agentmemory/agentmemory) reference implementation.

## 🎯 Performance Targets (Beat Reference Implementation)

| Metric | AgentMem Target | Reference (agentmemory) | Improvement |
|--------|----------------|-------------------------|-------------|
| Recall P50 Latency | **< 15ms** | ~50ms | **3.3x faster** |
| Recall P95 Latency | **< 50ms** | ~200ms | **4x faster** |
| Hybrid Search @ 1K docs | **< 50ms** | ~100ms | **2x faster** |
| Hybrid Search @ 10K docs | **< 200ms** | ~500ms | **2.5x faster** |
| BM25 Index Build @ 1K | **< 500ms** | ~1s | **2x faster** |
| Memory Footprint @ 10K | **< 500MB** | ~1GB | **2x smaller** |

## 📁 Test Structure

```
tests/
├── unit/                    # Unit tests for core modules
│   ├── test_heat.py        # Heat scoring & re-ranking
│   ├── test_hybrid_search.py  # BM25 + vector hybrid search
│   └── test_retention.py   # Memory lifecycle & retention
├── integration/             # API endpoint integration tests
│   └── test_api.py         # Full API test suite
├── benchmark/               # Performance benchmarks
│   └── test_scale.py       # Scale & performance benchmarks
└── fixtures/                # Test fixtures and helpers
```

## 🚀 Running Tests

### Prerequisites

Ensure Redis is running:
```bash
redis-server --port 6379
```

Start AgentMem service:
```bash
bash agentmem.sh start
```

### Install Test Dependencies

```bash
uv pip install pytest pytest-asyncio pytest-benchmark
```

### Run All Tests

```bash
pytest
```

### Run Specific Test Categories

```bash
# Unit tests only
pytest tests/unit/

# Integration tests only
pytest tests/integration/

# Benchmarks only
pytest tests/benchmark/

# With verbose output
pytest -v

# With benchmark comparison
pytest --benchmark-compare
```

### Run Performance Benchmarks

```bash
# Full benchmark suite with detailed stats
pytest tests/benchmark/ --benchmark-verbose

# Save benchmark results
pytest tests/benchmark/ --benchmark-save=baseline

# Compare against previous run
pytest tests/benchmark/ --benchmark-compare=0001_baseline
```

### Filter by Performance Markers

```bash
# Only fast tests (< 1s)
pytest -m "not slow"

# Only integration tests
pytest -m integration

# Only benchmarks
pytest -m benchmark
```

## 📊 Benchmark Results

Benchmarks are automatically saved to `.benchmarks/` directory. View historical results:

```bash
pytest --benchmark-history
```

## 🔬 Test Coverage Goals

| Component | Target Coverage | Current |
|-----------|----------------|---------|
| core/heat.py | 100% | ✅ |
| core/embedder.py | 95% | 🔄 |
| core/store.py | 95% | 🔄 |
| main.py (API) | 90% | 🔄 |
| adapters/ | 85% | 🔄 |

## 🧪 Writing New Tests

### Unit Test Template

```python
import pytest
from core.module import function

def test_function_behavior():
    """Test description."""
    result = function(input)
    assert result == expected
```

### Async Test Template

```python
import pytest

@pytest.mark.asyncio
async def test_async_function():
    """Async test description."""
    result = await async_function(input)
    assert result == expected
```

### Benchmark Test Template

```python
import pytest

@pytest.mark.benchmark(group="my-group")
def test_performance(benchmark):
    """Performance test."""
    result = benchmark(function_to_benchmark, args)
    assert result is not None
```

## 🎯 Migration from agentmemory

Key adaptations from TypeScript → Python:

1. **Testing Framework**: Vitest → pytest
2. **Async Support**: Native async/await with pytest-asyncio
3. **Benchmarking**: Built-in performance.now() → pytest-benchmark
4. **Mock Strategy**: In-memory Maps → Redis test DB (db=15)
5. **Type System**: TypeScript interfaces → Python type hints + Pydantic

## 📈 Continuous Performance Monitoring

To ensure we continue beating the reference implementation:

1. Run benchmarks before each release
2. Compare against baseline in CI/CD
3. Track regression trends over time
4. Alert on >10% performance degradation

```bash
# CI/CD integration
pytest tests/benchmark/ --benchmark-json=report.json
```

## 🐛 Debugging Tests

```bash
# Show print statements
pytest -s

# Stop on first failure
pytest -x

# Show local variables on failure
pytest -l

# Run last failed tests
pytest --lf
```

## 📚 References

- [agentmemory Test Suite](https://github.com/agentmemory/agentmemory/tree/main/test)
- [pytest Documentation](https://docs.pytest.org/)
- [pytest-benchmark](https://pytest-benchmark.readthedocs.io/)
- [Redis Testing Best Practices](https://redis.io/docs/manual/testing/)
