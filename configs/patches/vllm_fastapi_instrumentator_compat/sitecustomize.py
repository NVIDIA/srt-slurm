"""Compatibility patch for FastAPI lazy routers and Instrumentator.

FastAPI's lazy included routers do not necessarily expose a ``path`` field.
prometheus-fastapi-instrumentator 8.0 assumes that field exists, which can
make vLLM's ``/health`` endpoint return HTTP 500 before reaching the handler.
"""

from __future__ import annotations

try:
    from prometheus_fastapi_instrumentator import routing
    from starlette.routing import Match, Mount
except Exception:
    routing = None
else:
    if not getattr(routing, "_srt_included_router_compat", False):

        def _get_route_name(scope, routes, route_name=None):
            for route in routes:
                match, child_scope = route.matches(scope)
                path = getattr(route, "path", None) or scope.get("path", "")
                if match == Match.FULL:
                    route_name = path
                    child_scope = {**scope, **child_scope}
                    child_routes = getattr(route, "routes", None)
                    if isinstance(route, Mount) and child_routes:
                        child_route_name = _get_route_name(
                            child_scope,
                            child_routes,
                            route_name,
                        )
                        if child_route_name is None:
                            route_name = None
                        else:
                            route_name += child_route_name
                    return route_name
                if match == Match.PARTIAL and route_name is None:
                    route_name = path
            return None

        routing._get_route_name = _get_route_name
        routing._srt_included_router_compat = True
