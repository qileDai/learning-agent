from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from fastapi import Header, HTTPException, Request, status

from app.config import settings


@dataclass(frozen=True)
class AccessContext:
    user_id: str
    role: str
    tenant_id: str
    authenticated: bool

    @property
    def is_admin(self) -> bool:
        return self.role == "admin"


@dataclass(frozen=True)
class _TokenRecord:
    token: str
    role: str
    tenant_id: str
    user_id: str


@lru_cache(maxsize=1)
def _token_registry() -> dict[str, _TokenRecord]:
    registry: dict[str, _TokenRecord] = {}
    raw = str(settings.auth_tokens or "").strip()
    if not raw:
        return registry
    for item in raw.split(";"):
        entry = item.strip()
        if not entry:
            continue
        parts = [part.strip() for part in entry.split("|")]
        if len(parts) < 2:
            continue
        token = parts[0]
        role = parts[1] or "user"
        tenant_id = parts[2] if len(parts) > 2 and parts[2] else settings.default_tenant_id
        user_id = parts[3] if len(parts) > 3 and parts[3] else f"{role}:{tenant_id}"
        registry[token] = _TokenRecord(token=token, role=role, tenant_id=tenant_id, user_id=user_id)
    return registry


def _default_context() -> AccessContext:
    return AccessContext(
        user_id="anonymous",
        role="admin",
        tenant_id=settings.default_tenant_id,
        authenticated=False,
    )


def _resolve_token(authorization: str | None, x_api_key: str | None) -> str:
    bearer = str(authorization or "").strip()
    if bearer.lower().startswith("bearer "):
        return bearer[7:].strip()
    return str(x_api_key or "").strip()


def get_access_context(
    request: Request,
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None),
    x_tenant_id: str | None = Header(default=None),
    x_user_id: str | None = Header(default=None),
) -> AccessContext:
    cached = getattr(request.state, "access_context", None)
    if isinstance(cached, AccessContext):
        return cached
    if not settings.auth_enabled:
        context = _default_context()
        request.state.access_context = context
        return context
    token = _resolve_token(authorization, x_api_key)
    if not token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing bearer token or X-API-Key")
    record = _token_registry().get(token)
    if record is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API token")
    tenant_id = str(x_tenant_id or record.tenant_id or settings.default_tenant_id).strip() or settings.default_tenant_id
    if settings.require_tenant_header and not str(x_tenant_id or "").strip():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Missing X-Tenant-Id header")
    if record.tenant_id not in {"*", "all"} and tenant_id != record.tenant_id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Tenant mismatch")
    context = AccessContext(
        user_id=str(x_user_id or record.user_id or f"{record.role}:{tenant_id}").strip() or f"{record.role}:{tenant_id}",
        role=str(record.role or "user").strip() or "user",
        tenant_id=tenant_id,
        authenticated=True,
    )
    request.state.access_context = context
    return context


def require_admin(context: AccessContext) -> AccessContext:
    if not context.is_admin:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin role required")
    return context
