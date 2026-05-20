# ADR 001: Refactor Monolithic Architecture to Layered Design

## Status
Accepted

## Date
2026-05-20

## Context

AgentMem's current architecture concentrates ~5,000 lines of code in a single `main.py` file, mixing:
- FastAPI route definitions
- Business logic (memory storage, retrieval, consolidation)
- Background task management
- Utility functions
- Global state management
- WebSocket handlers
- Static file serving

This monolithic structure creates several problems:
1. **High cognitive load**: Developers must understand the entire file to make changes
2. **Difficult testing**: Business logic cannot be unit tested independently from HTTP layer
3. **Merge conflicts**: Multiple developers editing the same file causes frequent conflicts
4. **Violation of SOLID principles**: Single Responsibility Principle especially violated
5. **Poor scalability**: Adding new features requires modifying an already large file

Research of similar projects (agentmemory, Stash, MemPalace) shows that modular architectures with clear separation of concerns enable:
- Faster feature development
- Better testability
- Easier onboarding for contributors
- More maintainable codebase long-term

## Decision

Refactor AgentMem into a layered architecture with the following structure:

```
agentmem/
├── api/              # HTTP/WebSocket layer (routes, middleware, schemas)
├── services/         # Business logic layer (memory operations, retrieval, consolidation)
├── core/             # Domain layer (embeddings, extraction, storage primitives)
├── lifecycle/        # Memory lifecycle management (admission, decay, merge, prune)
├── concurrency/      # Thread-safety primitives (atomic counters, task management)
├── config/           # Centralized configuration
├── utils/            # Shared utilities
├── observability/    # Monitoring and diagnostics
└── main.py           # Entry point only (~200 lines)
```

### Key Principles

1. **Single Responsibility**: Each module has one reason to change
   - Routes handle HTTP concerns only
   - Services implement business logic
   - Core provides domain primitives

2. **Dependency Direction**: Dependencies flow inward
   - API depends on Services
   - Services depend on Core
   - Core depends on nothing (except external libraries)

3. **Interface Segregation**: Small, focused interfaces between layers
   - Services expose clean async APIs
   - No direct Redis access from routes
   - Configuration injected, not global

4. **Explicit over Implicit**: Clear data flow
   - Dependency injection via FastAPI Depends()
   - No hidden global state
   - All side effects documented

### Implementation Strategy

**Phase 1 (Week 1-2): Foundation**
- Create directory structure
- Extract configuration to `config/settings.py`
- Implement atomic counters in `concurrency/counters.py`
- Move utility functions to `utils/`

**Phase 2 (Week 2-4): Service Layer**
- Extract business logic into `services/` modules
- Split routes into `api/routes/` submodules
- Wire up dependency injection
- Maintain backward compatibility

**Phase 3 (Week 4-5): Lifecycle Management**
- Abstract memory lifecycle into `lifecycle/` module
- Separate admission, decay, merge, prune concerns
- Configurable policies per memory type

**Phase 4 (Week 5-6): Observability & Testing**
- Add structured logging with correlation IDs
- Implement Prometheus metrics
- Enhance test coverage to 85%+
- Add E2E scenario tests

## Consequences

### Positive
✅ **Improved maintainability**: Clear separation makes it easy to locate and modify code  
✅ **Better testability**: Services can be unit tested without HTTP layer  
✅ **Reduced coupling**: Layers communicate through well-defined interfaces  
✅ **Easier onboarding**: New contributors can understand one layer at a time  
✅ **Scalability**: New features added by extending services, not modifying monolith  
✅ **Performance isolation**: Can optimize individual layers independently  

### Negative
❌ **Initial complexity**: More files and directories to navigate initially  
❌ **Migration effort**: 8 weeks of refactoring work required  
❌ **Learning curve**: Team must understand new architecture patterns  
❌ **Temporary duplication**: During migration, some code exists in both old and new locations  

### Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| Breaking existing API | Contract tests ensure backward compatibility; shadow mode testing |
| Performance regression | Continuous benchmarking; rollback plan maintained |
| Migration takes too long | Phased approach; each phase delivers value independently |
| Team resistance | Involve team in design decisions; demonstrate benefits early |

## Alternatives Considered

### Alternative 1: Keep Monolithic Structure
**Pros:** Simpler initial understanding, fewer files  
**Cons:** Unsustainable long-term, violates best practices, hard to scale  
**Rejected because:** Technical debt would compound, making future changes increasingly difficult

### Alternative 2: Microservices Architecture
**Pros:** Maximum isolation, independent scaling  
**Cons:** Overkill for single-machine deployment, operational complexity, network latency  
**Rejected because:** AgentMem is designed as local-first service; microservices add unnecessary complexity

### Alternative 3: Plugin-Based Architecture
**Pros:** Extensible, community contributions easier  
**Cons:** Complex plugin interface, versioning challenges, debugging difficulty  
**Rejected because:** Current adapter pattern sufficient for framework integrations; plugins not needed yet

## References

1. Clean Architecture by Robert C. Martin
2. agentmemory architecture (GitHub: rohitg00/agentmemory)
3. Stash memory system (GitHub: stash-memory/stash)
4. Claude Code memory system analysis (eWeek, Neuron publications)
5. SOLID Principles of Object-Oriented Design

## Notes

This ADR establishes the foundation for all subsequent refactoring work. Future ADRs will document specific implementation decisions within this architectural framework.

---

**Related ADRs:**
- ADR 002: Atomic Counters for Concurrency Safety (pending)
- ADR 003: Configuration Management Strategy (pending)
- ADR 004: Error Handling and Retry Policies (pending)
