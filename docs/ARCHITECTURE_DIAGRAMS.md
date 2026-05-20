# AgentMem Architecture Diagrams

## Current Architecture (Before Refactoring)

```mermaid
graph TB
    subgraph "Monolithic main.py (4,977 lines)"
        A[FastAPI Routes] --> B[Business Logic]
        B --> C[Redis Operations]
        B --> D[LLM Calls]
        B --> E[Background Tasks]
        
        F[Global State] --> B
        G[Utility Functions] --> B
        H[WebSocket Handlers] --> B
        
        B --> I[Error Handling<br/>Inconsistent]
        B --> J[Configuration<br/>Hardcoded]
    end
    
    K[Claude Code Hooks] --> A
    L[LangChain Adapter] --> A
    M[MCP Clients] --> A
    N[Web Dashboard] --> A
    
    style A fill:#ff6b6b
    style B fill:#ff6b6b
    style F fill:#ffd93d
    style J fill:#ffd93d
```

**Problems:**
- 🔴 Everything in one file (high coupling)
- 🟡 Global mutable state (race conditions)
- 🟡 Hardcoded configuration (inflexible)
- ⚪ No clear separation of concerns

---

## Target Architecture (After Refactoring)

```mermaid
graph TB
    subgraph "Client Layer"
        C1[Claude Code]
        C2[LangChain/LangGraph]
        C3[CrewAI/AutoGen]
        C4[MCP Clients]
        C5[Web Dashboard]
    end
    
    subgraph "API Layer (api/)"
        R1[Routes<br/>memory.py, session.py,<br/>capability.py, graph.py]
        M1[Middleware<br/>rate_limit, auth, cors]
        S1[Schemas<br/>request.py, response.py]
        D1[Dependencies<br/>DI container]
    end
    
    subgraph "Service Layer (services/)"
        SV1[Memory Service<br/>CRUD operations]
        SV2[Retrieval Service<br/>Search + Fusion]
        SV3[Consolidation Service<br/>Lifecycle management]
        SV4[Extraction Service<br/>Fact extraction]
        SV5[Context Injection<br/>Token budgeting]
        SV6[Persona Service<br/>User profile]
        SV7[Graph Service<br/>Knowledge graph]
        SV8[Tool Memory Service<br/>Reliability tracking]
    end
    
    subgraph "Lifecycle Layer (lifecycle/)"
        L1[Admission Gate<br/>A-MAC 5-factor]
        L2[Decay Engine<br/>Ebbinghaus curve]
        L3[Merge Strategy<br/>Near-duplicate detection]
        L4[Prune Scheduler<br/>Hard-delete scheduling]
        L5[Evolution Engine<br/>A-MEM enrichment]
    end
    
    subgraph "Core Layer (core/)"
        CO1[Embedder<br/>MiniLM-L12]
        CO2[Extractor<br/>Hybrid regex+LLM]
        CO3[Store<br/>Redis operations]
        CO4[Graph<br/>Entity relationships]
        CO5[Heat<br/>Access frequency]
        CO6[Scene<br/>Language/domain detection]
        CO7[Persona<br/>Profile updates]
        CO8[Summarizer<br/>Session compression]
        CO9[Capability<br/>Tool registry]
        CO10[Retrieval Planner<br/>Query planning]
    end
    
    subgraph "Infrastructure"
        Redis[(Redis 8<br/>HNSW Vectorset)]
        LLM[LLM Provider<br/>Local or Remote]
    end
    
    subgraph "Cross-Cutting Concerns"
        CFG[Config<br/>settings.py]
        CONC[Concurrency<br/>atomic counters,<br/>task manager]
        OBS[Observability<br/>metrics, logging,<br/>health checks]
        UTILS[Utils<br/>text processing,<br/>validation]
        EXC[Exceptions<br/>custom hierarchy]
    end
    
    C1 --> R1
    C2 --> R1
    C3 --> R1
    C4 --> R1
    C5 --> R1
    
    R1 --> D1
    R1 --> M1
    R1 --> S1
    
    D1 --> SV1
    D1 --> SV2
    D1 --> SV3
    D1 --> SV4
    D1 --> SV5
    D1 --> SV6
    D1 --> SV7
    D1 --> SV8
    
    SV1 --> L1
    SV1 --> CO3
    SV2 --> L1
    SV2 --> CO1
    SV2 --> CO4
    SV3 --> L2
    SV3 --> L3
    SV3 --> L4
    SV3 --> L5
    SV4 --> CO2
    SV5 --> CO8
    SV6 --> CO7
    SV7 --> CO4
    SV8 --> CO9
    
    L1 --> CO1
    L2 --> CO3
    L3 --> CO1
    L4 --> CO3
    L5 --> CO1
    
    CO1 --> Redis
    CO2 --> LLM
    CO3 --> Redis
    CO4 --> Redis
    CO9 --> Redis
    
    CFG -.-> R1
    CFG -.-> SV1
    CFG -.-> SV2
    CFG -.-> CO1
    
    CONC -.-> SV1
    CONC -.-> SV3
    
    OBS -.-> R1
    OBS -.-> SV1
    OBS -.-> CO3
    
    UTILS -.-> SV4
    UTILS -.-> SV5
    
    EXC -.-> R1
    EXC -.-> SV1
    
    style R1 fill:#4ecdc4
    style SV1 fill:#45b7d1
    style SV2 fill:#45b7d1
    style SV3 fill:#45b7d1
    style L1 fill:#96ceb4
    style L2 fill:#96ceb4
    style CO1 fill:#feca57
    style CO3 fill:#feca57
    style Redis fill:#ff6b6b
    style CFG fill:#a29bfe
    style CONC fill:#a29bfe
    style OBS fill:#a29bfe
```

**Benefits:**
- ✅ Clear separation of concerns (color-coded layers)
- ✅ Dependency flows downward (no circular dependencies)
- ✅ Each layer independently testable
- ✅ Cross-cutting concerns centralized
- ✅ Easy to extend (add new services without modifying existing code)

---

## Data Flow: Store Operation

```mermaid
sequenceDiagram
    participant Client as Claude Code
    participant API as api/routes/memory.py
    participant SVC as services/memory_service.py
    participant LC as lifecycle/admission_gate.py
    participant EXT as services/extraction_service.py
    participant CORE as core/store.py
    participant REDIS as Redis 8
    participant CONS as services/consolidation_service.py
    
    Client->>API: POST /store {messages, session_id}
    activate API
    
    API->>SVC: store_messages(messages, session_id)
    activate SVC
    
    SVC->>LC: should_store(messages)
    activate LC
    LC->>LC: A-MAC 5-factor gate
    LC-->>SVC: True/False
    deactivate LC
    
    alt Admission rejected
        SVC-->>API: StoreResult.skipped
        API-->>Client: 200 OK {stored: 0}
    else Admission accepted
        SVC->>EXT: extract_facts(messages)
        activate EXT
        EXT->>EXT: Hybrid extraction (regex + LLM)
        EXT-->>SVC: List[ExtractedFact]
        deactivate EXT
        
        SVC->>CORE: save_episode(session_id, messages)
        activate CORE
        CORE->>REDIS: VADD mem:episodes
        CORE-->>SVC: episode_id
        deactivate CORE
        
        loop For each fact
            SVC->>CORE: save_fact(fact)
            activate CORE
            CORE->>REDIS: VADD mem:facts
            CORE-->>SVC: fact_id
            deactivate CORE
            
            SVC->>SVC: Update persona
            SVC->>SVC: Update knowledge graph
        end
        
        SVC->>CONS: record_store()
        activate CONS
        CONS->>CONS: Check if consolidation needed
        alt Threshold reached
            CONS->>CONS: Schedule consolidation task
        end
        deactivate CONS
        
        SVC-->>API: StoreResult.success(count)
        API-->>Client: 200 OK {stored: count}
    end
    
    deactivate SVC
    deactivate API
```

---

## Data Flow: Recall Operation

```mermaid
sequenceDiagram
    participant Client as LangChain
    participant API as api/routes/memory.py
    participant SVC as services/retrieval_service.py
    participant PLAN as core/retrieval_planner.py
    participant SEARCH as services/search_service.py
    participant FUSE as services/fusion_service.py
    participant INJECT as services/context_injection.py
    participant REDIS as Redis 8
    
    Client->>API: POST /recall {query, session_id, token_budget}
    activate API
    
    API->>SVC: recall(query, session_id, token_budget)
    activate SVC
    
    alt Planning enabled
        SVC->>PLAN: plan_queries(query)
        activate PLAN
        PLAN->>PLAN: LLM-based query decomposition
        PLAN-->>SVC: List[planned_query]
        deactivate PLAN
    end
    
    SVC->>SEARCH: multi_source_search(queries)
    activate SEARCH
    
    par Parallel searches
        SEARCH->>REDIS: VSIM mem:facts (vector)
        SEARCH->>REDIS: BM25 search (keyword)
        SEARCH->>REDIS: Graph neighborhood
        SEARCH->>REDIS: Symbolic filter (time/person)
    end
    
    REDIS-->>SEARCH: Results from each source
    SEARCH-->>SVC: Raw results
    deactivate SEARCH
    
    SVC->>FUSE: wrrf_merge(results, weights)
    activate FUSE
    FUSE->>FUSE: Dynamic weighted RRF fusion
    FUSE->>FUSE: Keyword boost
    FUSE->>FUSE: Importance boost
    FUSE-->>SVC: Ranked results
    deactivate FUSE
    
    SVC->>INJECT: pack_context(results, token_budget)
    activate INJECT
    INJECT->>INJECT: Priority-based packing
    INJECT->>INJECT: Token budget enforcement
    INJECT-->>SVC: Formatted context string
    deactivate INJECT
    
    SVC-->>API: prependContext
    API-->>Client: {prependContext: "..."}
    
    deactivate SVC
    deactivate API
```

---

## Concurrency Model

```mermaid
graph LR
    subgraph "Async Task Management"
        T1[Route Handler 1] --> TM[Task Manager]
        T2[Route Handler 2] --> TM
        T3[Periodic Task] --> TM
        
        TM --> Q1[Background Queue<br/>max 50 tasks]
        Q1 --> W1[Worker 1]
        Q1 --> W2[Worker 2]
        Q1 --> W3[Worker 3]
        
        W1 --> E1[Exception Handler]
        W2 --> E1
        W3 --> E1
        
        E1 --> L[Structured Logger]
    end
    
    subgraph "Atomic Counters"
        C1[_store_attempts] --> Lock[asyncio.Lock]
        C2[_store_successes] --> Lock
        C3[_stores_since_consolidation] --> Lock
        
        Lock --> R[Redis Atomic Ops]
    end
    
    subgraph "Circuit Breakers"
        CB1[LLM Calls] --> CB[Circuit Breaker]
        CB2[External APIs] --> CB
        
        CB --> F[Fallback Strategy]
    end
    
    style TM fill:#4ecdc4
    style Lock fill:#96ceb4
    style CB fill:#ff6b6b
```

---

## Configuration Hierarchy

```mermaid
graph TB
    subgraph "Configuration Sources (Priority Order)"
        E1[Environment Variables<br/>Highest Priority]
        E2[.env File]
        E3[defaults.yaml<br/>Lowest Priority]
    end
    
    subgraph "config/settings.py"
        S[Pydantic Settings<br/>Validation & Type Safety]
    end
    
    subgraph "Usage"
        U1[API Routes]
        U2[Services]
        U3[Core Modules]
        U4[Lifecycle Managers]
    end
    
    E1 --> S
    E2 --> S
    E3 --> S
    
    S --> U1
    S --> U2
    S --> U3
    S --> U4
    
    style S fill:#4ecdc4
    style E1 fill:#ffd93d
```

**Example:**
```python
# config/settings.py
class Settings(BaseSettings):
    auto_consolidate_every: int = Field(default=50, env="AUTO_CONSOLIDATE_EVERY")

# Usage anywhere in codebase
from config.settings import settings

if stores >= settings.auto_consolidate_every:
    await consolidate()
```

**Override via environment:**
```bash
export AUTO_CONSOLIDATE_EVERY=100
python main.py
```

---

## Error Handling Strategy

```mermaid
graph TB
    subgraph "Exception Hierarchy"
        E0[AgentMemError<br/>Base Exception]
        E1[StorageError]
        E2[RetrievalError]
        E3[ConsolidationError]
        E4[ExtractionError]
        E5[ConfigurationError]
        
        E0 --> E1
        E0 --> E2
        E0 --> E3
        E0 --> E4
        E0 --> E5
    end
    
    subgraph "Error Handling Flow"
        H1[Route Handler] --> Try[Try/Except Block]
        Try --> |Success| R1[Return Response]
        Try --> |AgentMemError| H2[Log + HTTP 500]
        Try --> |ValidationError| H3[Log + HTTP 400]
        Try --> |Unexpected| H4[Log + HTTP 500 + Alert]
        
        H2 --> M1[Structured Log<br/>with correlation ID]
        H3 --> M1
        H4 --> M1
        H4 --> A1[Alert System]
    end
    
    subgraph "Retry Policy"
        RP1[Transient Errors] --> Retry[@retry decorator<br/>3 attempts, exponential backoff]
        RP2[Permanent Errors] --> NoRetry[No retry, immediate failure]
        
        Retry --> F1[Fallback Strategy]
    end
    
    style E0 fill:#ff6b6b
    style Try fill:#4ecdc4
    style Retry fill:#96ceb4
```

---

## Testing Pyramid

```mermaid
graph TB
    subgraph "Testing Strategy"
        T1[E2E Tests<br/>~10% of tests<br/>Full workflow scenarios]
        T2[Integration Tests<br/>~20% of tests<br/>API endpoints + Redis]
        T3[Unit Tests<br/>~70% of tests<br/>Individual services/functions]
        
        T4[Chaos Tests<br/>Failure injection<br/>Network partitions, timeouts]
        T5[Performance Benchmarks<br/>Latency, throughput<br/>Automated in CI]
        T6[Property-Based Tests<br/>Consolidation logic<br/>Edge cases]
    end
    
    T3 --> T2
    T2 --> T1
    
    T4 -.-> T2
    T5 -.-> T1
    T6 -.-> T3
    
    style T1 fill:#ff6b6b
    style T2 fill:#feca57
    style T3 fill:#4ecdc4
    style T4 fill:#a29bfe
    style T5 fill:#fd79a8
    style T6 fill:#00b894
```

**Target Coverage:**
- Unit tests: 85%+ line coverage
- Integration tests: All API endpoints
- E2E tests: Critical user journeys (store → consolidate → recall)
- Performance: Automated regression detection
- Chaos: Key failure scenarios tested

---

## Migration Timeline

```mermaid
gantt
    title AgentMem Refactoring Timeline (8 Weeks)
    dateFormat  YYYY-MM-DD
    section Phase 1: Foundation
    Directory structure           :done, p1_1, 2026-05-20, 2d
    Config centralization         :active, p1_2, 2026-05-22, 3d
    Atomic counters               :p1_3, after p1_2, 2d
    Utility extraction            :p1_4, after p1_3, 2d
    
    section Phase 2: Service Layer
    Memory service                :p2_1, after p1_4, 4d
    Retrieval service             :p2_2, after p2_1, 4d
    Consolidation service         :p2_3, after p2_2, 3d
    Route splitting               :p2_4, after p2_3, 3d
    
    section Phase 3: Lifecycle
    Admission gate extraction     :p3_1, after p2_4, 3d
    Decay engine                  :p3_2, after p3_1, 2d
    Merge strategy                :p3_3, after p3_2, 2d
    Prune scheduler               :p3_4, after p3_3, 2d
    
    section Phase 4: Observability
    Structured logging            :p4_1, after p3_4, 2d
    Metrics collection            :p4_2, after p4_1, 2d
    Circuit breakers              :p4_3, after p4_2, 2d
    Health checks                 :p4_4, after p4_3, 1d
    
    section Phase 5: Testing
    E2E scenarios                 :p5_1, after p4_4, 4d
    Property-based tests          :p5_2, after p5_1, 3d
    Chaos testing                 :p5_3, after p5_2, 2d
    Performance automation        :p5_4, after p5_3, 2d
    
    section Phase 6: Documentation
    ADRs                          :p6_1, after p5_4, 3d
    API docs                      :p6_2, after p6_1, 2d
    Developer guide               :p6_3, after p6_2, 2d
    Final cleanup                 :p6_4, after p6_3, 2d
```

---

## Success Metrics Dashboard

```mermaid
graph LR
    subgraph "Code Quality Metrics"
        M1[main.py Lines: 4977 → <500 ✅]
        M2[Avg File Size: <300 lines ✅]
        M3[Test Coverage: 60% → 85%+ ✅]
        M4[Cyclomatic Complexity: <10 ✅]
    end
    
    subgraph "Performance Metrics"
        M5[P50 Latency: ≤15ms ✅]
        M6[P95 Latency: <50ms ✅]
        M7[Throughput: ≥100 req/sec ✅]
        M8[Memory Footprint: <250MB ✅]
    end
    
    subgraph "Reliability Metrics"
        M9[Uptime: 99.9% ✅]
        M10[Silent Failures: 0 ✅]
        M11[Race Conditions: 0 ✅]
        M12[Auto-Recovery: Yes ✅]
    end
    
    subgraph "Developer Experience"
        M13[Onboarding Time: <1 day ✅]
        M14[Feature Dev Time: <1 hour ✅]
        M15[Documentation: Complete ✅]
        M16[Error Messages: Actionable ✅]
    end
    
    style M1 fill:#00b894
    style M2 fill:#00b894
    style M3 fill:#00b894
    style M5 fill:#00b894
    style M9 fill:#00b894
    style M11 fill:#00b894
```

All metrics tracked via Prometheus + Grafana dashboard (to be implemented in Phase 4).
