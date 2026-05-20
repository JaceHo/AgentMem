"""API routes package - HTTP endpoint definitions."""

from api.routes.health import router as health_router
from api.routes.capability import router as capability_router
from api.routes.graph import router as graph_router
from api.routes.compat import router as compat_router
from api.routes.admin import router as admin_router

__all__ = [
    "health_router",
    "capability_router",
    "graph_router",
    "compat_router",
    "admin_router",
]
