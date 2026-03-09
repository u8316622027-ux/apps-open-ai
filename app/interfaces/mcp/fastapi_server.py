"""Optional FastAPI transport for MCP JSON-RPC endpoints."""

from __future__ import annotations

import argparse
import asyncio
import gzip
import importlib
import json
from typing import Any

from app.interfaces.mcp.server import (
    MCP_PROTOCOL_VERSION,
    MAX_REQUEST_BODY_BYTES,
    MIN_GZIP_BYTES,
    _client_accepts_json,
    _client_accepts_sse,
    _is_json_content_type,
    _resolve_mcp_protocol_version,
    _resolve_http_request_id,
    _rpc_error,
    get_runtime_metrics,
    handle_jsonrpc_payload,
)


def _client_accepts_gzip(accept_encoding: str | None) -> bool:
    if accept_encoding is None:
        return False
    return "gzip" in accept_encoding.lower()


def _build_json_response_bytes(
    payload: dict[str, Any] | list[dict[str, Any]],
    *,
    accept_encoding: str | None,
) -> tuple[bytes, dict[str, str]]:
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    if len(encoded) < MIN_GZIP_BYTES or not _client_accepts_gzip(accept_encoding):
        return encoded, {}

    compressed = gzip.compress(encoded, compresslevel=5)
    return compressed, {"Content-Encoding": "gzip", "Vary": "Accept-Encoding"}


def _build_sse_data_bytes(payload: dict[str, Any] | list[dict[str, Any]]) -> bytes:
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return f"data: {encoded}\n\n".encode("utf-8")


async def _dispatch_jsonrpc_in_thread(
    request_payload: Any,
    *,
    registry: dict[str, Any] | None,
    request_id: str,
) -> dict[str, Any] | list[dict[str, Any]] | None:
    return await asyncio.to_thread(
        handle_jsonrpc_payload,
        request_payload,
        registry=registry,
        http_request_id=request_id,
    )


def create_fastapi_app(*, registry: dict[str, Any] | None = None) -> Any:
    try:
        fastapi = importlib.import_module("fastapi")
        responses = importlib.import_module("fastapi.responses")
    except ModuleNotFoundError as exc:  # pragma: no cover - environment-dependent
        raise RuntimeError(
            "FastAPI transport requires optional dependencies: fastapi and uvicorn."
        ) from exc

    app = fastapi.FastAPI(title="Apteka MCP", version="0.1.0")
    Response = responses.Response
    JSONResponse = responses.JSONResponse

    @app.get("/health")
    async def health(request: Any) -> Any:
        request_id = _resolve_http_request_id(request.headers.get("x-request-id"))
        protocol_version = _resolve_mcp_protocol_version(request.headers.get("mcp-protocol-version"))
        return JSONResponse(
            {"status": "ok"},
            headers={"X-Request-Id": request_id, "MCP-Protocol-Version": protocol_version},
        )

    @app.get("/metrics")
    async def metrics(request: Any) -> Any:
        request_id = _resolve_http_request_id(request.headers.get("x-request-id"))
        protocol_version = _resolve_mcp_protocol_version(request.headers.get("mcp-protocol-version"))
        return JSONResponse(
            get_runtime_metrics(),
            headers={"X-Request-Id": request_id, "MCP-Protocol-Version": protocol_version},
        )

    @app.get("/mcp")
    async def mcp_transport(request: Any) -> Any:
        request_id = _resolve_http_request_id(request.headers.get("x-request-id"))
        protocol_version = _resolve_mcp_protocol_version(request.headers.get("mcp-protocol-version"))
        if not _client_accepts_sse(request.headers.get("accept")):
            return Response(
                status_code=405,
                headers={
                    "Allow": "POST, GET",
                    "X-Request-Id": request_id,
                    "MCP-Protocol-Version": protocol_version,
                },
            )
        return Response(
            content=b": connected\n\n",
            status_code=200,
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Request-Id": request_id,
                "MCP-Protocol-Version": protocol_version,
            },
        )

    @app.post("/mcp")
    async def mcp_rpc(request: Any) -> Any:
        request_id = _resolve_http_request_id(request.headers.get("x-request-id"))
        protocol_version = _resolve_mcp_protocol_version(request.headers.get("mcp-protocol-version"))
        accept_header = request.headers.get("accept")
        sse_only = _client_accepts_sse(accept_header) and not _client_accepts_json(accept_header)
        content_type = request.headers.get("content-type")
        if not _is_json_content_type(content_type):
            error_payload = _rpc_error(None, -32600, "Invalid Request: Content-Type must be application/json")
            if sse_only:
                return Response(
                    content=_build_sse_data_bytes(error_payload),
                    status_code=415,
                    media_type="text/event-stream",
                    headers={
                        "Cache-Control": "no-cache",
                        "X-Request-Id": request_id,
                        "MCP-Protocol-Version": protocol_version,
                    },
                )
            return JSONResponse(
                error_payload,
                status_code=415,
                headers={"X-Request-Id": request_id, "MCP-Protocol-Version": protocol_version},
            )

        raw_body = await request.body()
        if len(raw_body) > MAX_REQUEST_BODY_BYTES:
            error_payload = _rpc_error(None, -32600, "Invalid Request: body is too large")
            if sse_only:
                return Response(
                    content=_build_sse_data_bytes(error_payload),
                    status_code=413,
                    media_type="text/event-stream",
                    headers={
                        "Cache-Control": "no-cache",
                        "X-Request-Id": request_id,
                        "MCP-Protocol-Version": protocol_version,
                    },
                )
            return JSONResponse(
                error_payload,
                status_code=413,
                headers={"X-Request-Id": request_id, "MCP-Protocol-Version": protocol_version},
            )

        try:
            request_payload = json.loads(raw_body.decode("utf-8"))
        except json.JSONDecodeError:
            error_payload = _rpc_error(None, -32700, "Parse error")
            if sse_only:
                return Response(
                    content=_build_sse_data_bytes(error_payload),
                    status_code=400,
                    media_type="text/event-stream",
                    headers={
                        "Cache-Control": "no-cache",
                        "X-Request-Id": request_id,
                        "MCP-Protocol-Version": protocol_version,
                    },
                )
            return JSONResponse(
                error_payload,
                status_code=400,
                headers={"X-Request-Id": request_id, "MCP-Protocol-Version": protocol_version},
            )

        response_payload = await _dispatch_jsonrpc_in_thread(
            request_payload,
            registry=registry,
            request_id=request_id,
        )
        if response_payload is None:
            return Response(
                status_code=204,
                headers={"X-Request-Id": request_id, "MCP-Protocol-Version": protocol_version},
            )

        if sse_only:
            return Response(
                content=_build_sse_data_bytes(response_payload),
                status_code=200,
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "X-Request-Id": request_id,
                    "MCP-Protocol-Version": protocol_version,
                },
            )

        response_bytes, encoding_headers = _build_json_response_bytes(
            response_payload,
            accept_encoding=request.headers.get("accept-encoding"),
        )
        headers = {"X-Request-Id": request_id, "MCP-Protocol-Version": protocol_version}
        headers.update(encoding_headers)
        return Response(
            content=response_bytes,
            status_code=200,
            media_type="application/json",
            headers=headers,
        )

    return app


def run_fastapi_server(host: str = "127.0.0.1", port: int = 8001) -> None:
    try:
        uvicorn = importlib.import_module("uvicorn")
    except ModuleNotFoundError as exc:  # pragma: no cover - environment-dependent
        raise RuntimeError("FastAPI transport requires optional dependency: uvicorn.") from exc

    app = create_fastapi_app()
    uvicorn.run(app, host=host, port=port)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run FastAPI MCP transport.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    args = parser.parse_args()
    run_fastapi_server(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
