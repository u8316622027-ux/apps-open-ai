"""Tests for optional FastAPI MCP transport adapter."""

from __future__ import annotations

import asyncio
import gzip
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from app.interfaces.mcp.fastapi_server import (
    _build_json_response_bytes,
    _client_accepts_gzip,
    _dispatch_jsonrpc_in_thread,
    create_fastapi_app,
)


class MCPFastAPIServerTests(unittest.TestCase):
    def test_dispatch_jsonrpc_in_thread_uses_asyncio_to_thread(self) -> None:
        async def run_case() -> None:
            with patch(
                "app.interfaces.mcp.fastapi_server.asyncio.to_thread",
                return_value={"jsonrpc": "2.0", "id": 1, "result": {"ok": True}},
            ) as mocked_to_thread:
                response = await _dispatch_jsonrpc_in_thread(
                    {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
                    registry=None,
                    request_id="req-1",
                )
            mocked_to_thread.assert_called_once()
            self.assertEqual(response["id"], 1)

        asyncio.run(run_case())

    def test_client_accepts_gzip_is_case_insensitive(self) -> None:
        self.assertTrue(_client_accepts_gzip("gzip"))
        self.assertTrue(_client_accepts_gzip("br, GZIP"))
        self.assertFalse(_client_accepts_gzip("br"))
        self.assertFalse(_client_accepts_gzip(None))

    def test_build_json_response_bytes_can_compress_payload(self) -> None:
        payload = {"data": "x" * 1000}
        body, headers = _build_json_response_bytes(payload, accept_encoding="gzip")

        self.assertEqual(headers.get("Content-Encoding"), "gzip")
        decoded = gzip.decompress(body).decode("utf-8")
        self.assertEqual(json.loads(decoded)["data"], "x" * 1000)

    def test_create_fastapi_app_raises_clear_error_when_dependency_missing(self) -> None:
        with patch(
            "app.interfaces.mcp.fastapi_server.importlib.import_module",
            side_effect=ModuleNotFoundError("fastapi"),
        ):
            with self.assertRaisesRegex(RuntimeError, "FastAPI transport requires"):
                create_fastapi_app()

    def test_create_fastapi_app_does_not_require_staticfiles_module(self) -> None:
        class FakeApp:
            def __init__(self) -> None:
                self.mounted: list[tuple[str, object, str]] = []
                self.get_routes: list[str] = []
                self.post_routes: list[str] = []

            def get(self, _path: str):
                self.get_routes.append(_path)

                def _decorator(func):
                    return func

                return _decorator

            def post(self, _path: str):
                self.post_routes.append(_path)

                def _decorator(func):
                    return func

                return _decorator

        fake_app = FakeApp()

        module_map = {
            "fastapi": SimpleNamespace(FastAPI=lambda **kwargs: fake_app),
            "fastapi.responses": SimpleNamespace(Response=object, JSONResponse=object),
        }

        with patch(
            "app.interfaces.mcp.fastapi_server.importlib.import_module",
            side_effect=lambda name: module_map[name],
        ):
            app = create_fastapi_app()

        self.assertIs(app, fake_app)
        self.assertFalse(fake_app.mounted)
        self.assertIn("/mcp", fake_app.get_routes)
        self.assertIn("/mcp", fake_app.post_routes)


if __name__ == "__main__":
    unittest.main()
