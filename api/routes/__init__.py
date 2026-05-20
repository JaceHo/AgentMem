"""API routes package - HTTP endpoint definitions."""

from api.routes.health import router as health_router

__all__ = ["health_router"]
