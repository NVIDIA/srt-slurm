"""Compatibility patch for FastAPI lazy included routers and Instrumentator.

FastAPI 0.137 keeps included routers as `_IncludedRouter` entries. The
prometheus-fastapi-instrumentator 8.0 route helper assumes every matched route
has a `path` attribute, so vLLM's `/health` can fail with HTTP 500 before the
request reaches the actual health handler.
"""

from __future__ import annotations

try:
    from prometheus_fastapi_instrumentator import routing
    from starlette.routing import Match, Mount
except Exception:
    routing = None
else:
    if not getattr(routing, "_vigil_included_router_compat", False):

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
        routing._vigil_included_router_compat = True
