"""
Concurrency primitives for thread-safe operations in async contexts.

Provides atomic counters and task management to prevent race conditions
in concurrent background operations.
"""

import asyncio
from typing import Any


class AtomicCounter:
    """Thread-safe counter for async contexts using asyncio.Lock.
    
    Prevents race conditions when multiple coroutines increment/shared
    state concurrently (e.g., _store_attempts, _store_successes).
    
    Example:
        >>> counter = AtomicCounter()
        >>> await counter.increment()
        1
        >>> await counter.get()
        1
    """
    
    def __init__(self, initial_value: int = 0):
        self._value = initial_value
        self._lock = asyncio.Lock()
    
    async def increment(self, amount: int = 1) -> int:
        """Atomically increment counter and return new value."""
        async with self._lock:
            self._value += amount
            return self._value
    
    async def decrement(self, amount: int = 1) -> int:
        """Atomically decrement counter and return new value."""
        async with self._lock:
            self._value -= amount
            return self._value
    
    async def get(self) -> int:
        """Get current value (thread-safe read)."""
        async with self._lock:
            return self._value
    
    async def set(self, value: int) -> None:
        """Set value atomically."""
        async with self._lock:
            self._value = value
    
    async def reset(self) -> None:
        """Reset counter to zero."""
        async with self._lock:
            self._value = 0


class AtomicFloat:
    """Thread-safe float accumulator for async contexts.
    
    Used for accumulating latency measurements (_store_latency_sum_ms)
    where precision matters and we need floating point arithmetic.
    """
    
    def __init__(self, initial_value: float = 0.0):
        self._value = initial_value
        self._lock = asyncio.Lock()
    
    async def add(self, amount: float) -> float:
        """Atomically add amount and return new value."""
        async with self._lock:
            self._value += amount
            return self._value
    
    async def get(self) -> float:
        """Get current value (thread-safe read)."""
        async with self._lock:
            return self._value
    
    async def reset(self) -> None:
        """Reset to zero."""
        async with self._lock:
            self._value = 0.0


class TaskManager:
    """Manages tracked background tasks with concurrency limits.
    
    Replaces bare asyncio.create_task() calls with:
    - Concurrency limiting (max N concurrent tasks)
    - Exception logging (no silent failures)
    - Graceful shutdown (cancel all tasks)
    - Task tracking for monitoring
    
    Example:
        >>> tm = TaskManager(max_concurrent=50)
        >>> tm.spawn(some_coro(), name="background-work")
        >>> await tm.shutdown()  # Cancels all running tasks
    """
    
    def __init__(self, max_concurrent: int = 50):
        self._tasks: set[asyncio.Task] = set()
        self._max_concurrent = max_concurrent
        self._lock = asyncio.Lock()
    
    def spawn(self, coro, name: str = "bg") -> asyncio.Task | None:
        """Fire-and-forget a coroutine as a tracked background task.
        
        Args:
            coro: Coroutine to execute
            name: Human-readable task name for logging
            
        Returns:
            Task object or None if limit reached
        """
        import logging
        log = logging.getLogger("mem")
        
        # Check concurrency limit
        if len(self._tasks) >= self._max_concurrent:
            log.warning(
                "[task_manager] limit (%d) reached, dropping %s",
                self._max_concurrent, name
            )
            return None
        
        # Create and track task
        task = asyncio.create_task(coro, name=name)
        self._tasks.add(task)
        
        # Auto-cleanup on completion
        task.add_done_callback(lambda t: self._tasks.discard(t))
        
        # Log exceptions (prevents "Task exception was never retrieved" warning)
        def _log_exception(t: asyncio.Task):
            if t.cancelled():
                return
            exc = t.exception()
            if exc:
                log.error(
                    "[task_manager:%s] unhandled exception: %s",
                    name, exc, exc_info=True
                )
        
        task.add_done_callback(_log_exception)
        return task
    
    async def shutdown(self, timeout: float = 5.0) -> None:
        """Cancel all tracked tasks and wait for cleanup.
        
        Args:
            timeout: Max seconds to wait for tasks to cancel
        """
        import logging
        log = logging.getLogger("mem")
        
        if not self._tasks:
            return
        
        log.info("[task_manager] shutting down %d tasks...", len(self._tasks))
        
        # Cancel all tasks
        for task in self._tasks:
            task.cancel()
        
        # Wait for cancellation
        try:
            await asyncio.wait_for(
                asyncio.gather(*self._tasks, return_exceptions=True),
                timeout=timeout
            )
        except asyncio.TimeoutError:
            log.warning("[task_manager] timeout waiting for tasks to cancel")
        
        self._tasks.clear()
        log.info("[task_manager] shutdown complete")
    
    @property
    def active_count(self) -> int:
        """Number of currently running tasks."""
        return len(self._tasks)
    
    @property
    def is_empty(self) -> bool:
        """True if no tasks are running."""
        return len(self._tasks) == 0
