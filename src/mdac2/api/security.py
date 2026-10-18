"""Loopback-only ASGI boundary; token is supplied by the installation launcher."""
import hmac
import ipaddress
import re

from fastapi import Request
from starlette.responses import JSONResponse

from mdac2.domain.artifact import strict_json


def error(status, code, message, *, retryable=False):
    return JSONResponse({"code": code, "message": message, "details": {},
                         "retryable": retryable, "run_id": None}, status_code=status)


class LocalBoundary:
    def __init__(self, app, *, token, hosts, public_static=False):
        if token is not None and (type(token) is not str or not re.fullmatch(r"[A-Za-z0-9_-]{32,256}", token)):
            raise ValueError("Strong installation token required")
        self.app, self.token = app, token
        self.public_static = public_static
        self.hosts = frozenset(hosts)
        if not self.hosts or any(not re.fullmatch(r"(?:127\.0\.0\.1|localhost|\[::1\]):[0-9]{1,5}", host) for host in self.hosts):
            raise ValueError("Exact loopback host:port allowlist required")
        self.origins = frozenset("http://" + host for host in self.hosts)

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        request = Request(scope)
        try:
            local = request.client is not None and ipaddress.ip_address(request.client.host).is_loopback
        except ValueError:
            local = False
        headers = request.headers
        host = headers.get("host", "")
        origin = headers.get("origin")
        authorization = headers.get("authorization", "")
        invalid = any(len(headers.getlist(key)) > 1 for key in ("host", "origin", "authorization"))
        if not local or invalid or host not in self.hosts or (origin is not None and origin not in self.origins) or headers.get('sec-fetch-site')=='cross-site':
            return await error(403, "LOCAL_ONLY", "Local trusted origin required")(scope, receive, send)
        public = self.public_static and scope['method'] in ('GET','HEAD') and (scope['path']=='/' or scope['path'].startswith('/assets/'))
        if self.token is not None and not public and not hmac.compare_digest(authorization.encode(), ("Bearer " + self.token).encode()):
            return await error(401, "AUTH_REQUIRED", "Installation token required")(scope, receive, send)
        async def secured_send(message):
            if message["type"] == "http.response.start":
                message.setdefault("headers", []).extend([(b"cache-control", b"no-store"), (b"x-content-type-options", b"nosniff")])
            await send(message)
        await self.app(scope, receive, secured_send)


async def json_body(request, *, limit=1024 * 1024):
    if request.headers.get("content-type", "").split(";")[0].strip().lower() != "application/json":
        raise ValueError("JSON content type required")
    body = bytearray()
    async for chunk in request.stream():
        if len(body) + len(chunk) > limit:
            raise ValueError("Request exceeds body limit")
        body.extend(chunk)
    return strict_json(bytes(body))


def fields(value, names):
    if type(value) is not dict or set(value) != set(names):
        raise ValueError("Unexpected fields")


def identifier(value):
    if type(value) is not str or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", value):
        raise ValueError("Invalid identifier")
    return value
