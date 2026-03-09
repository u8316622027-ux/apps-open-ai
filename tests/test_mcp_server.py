"""Tests for the minimal MCP HTTP server request handling."""

from __future__ import annotations

import http.client
import gzip
import hashlib
import json
import os
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from app.interfaces.mcp.tools.apteka_urls import get_apteka_base_url
from app.core.config import get_settings
from app.interfaces.mcp.server import (
    _build_access_log_message,
    _is_json_content_type,
    _reset_runtime_metrics_for_tests,
    _resolve_http_request_id,
    _reset_server_caches_for_tests,
    MCPHttpHandler,
    ThreadingHTTPServer,
    create_tool_registry,
    get_runtime_metrics,
    handle_jsonrpc_payload,
    handle_rpc_request,
    run_server,
)


class MCPServerTests(unittest.TestCase):
    def tearDown(self) -> None:
        get_settings.cache_clear()
        _reset_server_caches_for_tests()
        _reset_runtime_metrics_for_tests()

    def test_resolve_http_request_id_prefers_incoming_header(self) -> None:
        self.assertEqual(_resolve_http_request_id("req-123"), "req-123")

    def test_resolve_http_request_id_generates_when_missing(self) -> None:
        generated = _resolve_http_request_id(None)
        self.assertIsInstance(generated, str)
        self.assertTrue(len(generated) >= 8)

    def test_json_content_type_allows_application_json_with_charset(self) -> None:
        self.assertTrue(_is_json_content_type("application/json"))
        self.assertTrue(_is_json_content_type("application/json; charset=utf-8"))
        self.assertFalse(_is_json_content_type("text/plain"))

    def test_handle_jsonrpc_payload_supports_batch_requests(self) -> None:
        registry = create_tool_registry()
        response = handle_jsonrpc_payload(
            [
                {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
                {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
            ],
            registry=registry,
        )

        self.assertIsInstance(response, list)
        assert isinstance(response, list)
        self.assertEqual(len(response), 2)
        self.assertEqual(response[0]["id"], 1)
        self.assertEqual(response[1]["id"], 2)

    def test_handle_jsonrpc_payload_returns_invalid_request_for_empty_batch(self) -> None:
        response = handle_jsonrpc_payload([])

        self.assertIsInstance(response, dict)
        assert isinstance(response, dict)
        self.assertEqual(response["error"]["code"], -32600)

    def test_handle_jsonrpc_payload_notification_returns_none(self) -> None:
        registry = create_tool_registry()
        with patch(
            "app.interfaces.mcp.server.search_products",
            return_value={"query": "цитрамон", "count": 1, "products": [{"id": 1}]},
        ) as mocked_search:
            response = handle_jsonrpc_payload(
                {
                    "jsonrpc": "2.0",
                    "method": "tools/call",
                    "params": {
                        "name": "search_products",
                        "arguments": {"query": "цитрамон", "limit": 5},
                    },
                },
                registry=registry,
            )

        mocked_search.assert_called_once_with("цитрамон", limit=5)
        self.assertIsNone(response)

    def test_invalid_request_payload_type_returns_invalid_request_error(self) -> None:
        response = handle_rpc_request("not-a-dict")  # type: ignore[arg-type]

        self.assertIn("error", response)
        self.assertEqual(response["error"]["code"], -32600)

    def test_initialize_returns_server_info(self) -> None:
        registry = create_tool_registry()

        response = handle_rpc_request(
            {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
            registry=registry,
        )

        self.assertEqual(response["jsonrpc"], "2.0")
        self.assertEqual(response["id"], 1)
        self.assertIn("result", response)
        self.assertEqual(response["result"]["serverInfo"]["name"], "apteka-mcp")

    def test_tools_list_contains_required_tools(self) -> None:
        registry = create_tool_registry()

        response = handle_rpc_request(
            {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
            registry=registry,
        )

        names = {tool["name"] for tool in response["result"]["tools"]}
        self.assertSetEqual(
            names,
            {
                "add_to_my_cart",
                "checkout_order",
                "my_cart",
                "search_products",
                "set_widget_theme",
                "support_knowledge_search",
                "track_order_status_ui",
            },
        )

    def test_tools_list_includes_ui_metadata_for_search_products(self) -> None:
        registry = create_tool_registry()

        response = handle_rpc_request(
            {"jsonrpc": "2.0", "id": 20, "method": "tools/list", "params": {}},
            registry=registry,
        )

        tool_payload = next(
            tool for tool in response["result"]["tools"] if tool["name"] == "search_products"
        )

        self.assertEqual(tool_payload["outputTemplate"], "ui://widget/products.html")
        self.assertIn("ui", tool_payload)
        self.assertEqual(
            tool_payload["ui"]["domain"],
            "https://subgerminal-yevette-lactogenic.ngrok-free.dev",
        )
        self.assertIn(
            "https://subgerminal-yevette-lactogenic.ngrok-free.dev",
            tool_payload["ui"]["csp"]["resourceDomains"],
        )
        self.assertIn(get_apteka_base_url(), tool_payload["ui"]["csp"]["resourceDomains"])
        self.assertIn(get_apteka_base_url(), tool_payload["ui"]["csp"]["connectDomains"])
        self.assertIn("https://www.apteka.md", tool_payload["ui"]["csp"]["resourceDomains"])
        self.assertIn("https://cdn.jsdelivr.net", tool_payload["ui"]["csp"]["resourceDomains"])
        self.assertIn("_meta", tool_payload)
        self.assertEqual(
            tool_payload["_meta"]["openai/outputTemplate"],
            "ui://widget/products.html",
        )
        self.assertEqual(
            tool_payload["_meta"]["openai/widgetDomain"],
            "https://subgerminal-yevette-lactogenic.ngrok-free.dev",
        )

    def test_tools_list_uses_widget_domain_from_env(self) -> None:
        _reset_server_caches_for_tests()
        get_settings.cache_clear()
        with patch.dict(
            os.environ,
            {"MCP_WIDGET_DOMAIN": "https://widgets.example.com"},
            clear=False,
        ):
            registry = create_tool_registry()
            response = handle_rpc_request(
                {"jsonrpc": "2.0", "id": 20, "method": "tools/list", "params": {}},
                registry=registry,
            )

        tool_payload = next(
            tool for tool in response["result"]["tools"] if tool["name"] == "search_products"
        )
        self.assertEqual(tool_payload["ui"]["domain"], "https://widgets.example.com")
        self.assertIn("https://widgets.example.com", tool_payload["ui"]["csp"]["resourceDomains"])
        self.assertEqual(tool_payload["_meta"]["openai/widgetDomain"], "https://widgets.example.com")

    def test_default_registry_is_cached_between_requests(self) -> None:
        _reset_server_caches_for_tests()
        with patch(
            "app.interfaces.mcp.server.create_tool_registry",
            wraps=create_tool_registry,
        ) as mocked_create_registry:
            handle_rpc_request({"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}})
            handle_rpc_request({"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}})
            handle_rpc_request({"jsonrpc": "2.0", "id": 3, "method": "initialize", "params": {}})

        self.assertEqual(mocked_create_registry.call_count, 1)

    def test_tools_call_search_products_delegates_to_tool_function(self) -> None:
        registry = create_tool_registry()
        with patch(
            "app.interfaces.mcp.server.search_products",
            return_value={"query": "цитрамон", "count": 1, "products": [{"id": 1}]},
        ) as mocked_search:
            response = handle_rpc_request(
                {
                    "jsonrpc": "2.0",
                    "id": 3,
                    "method": "tools/call",
                    "params": {
                        "name": "search_products",
                        "arguments": {"query": "цитрамон", "limit": 5},
                    },
                },
                registry=registry,
            )

        mocked_search.assert_called_once_with("цитрамон", limit=5)
        self.assertFalse(response["result"]["isError"])
        self.assertEqual(response["result"]["structuredContent"]["count"], 1)
        self.assertFalse(response["result"]["structuredContent"]["no_results"])
        self.assertEqual(
            response["result"]["structuredContent"]["widget"]["open"]["template"],
            "ui://widget/products.html",
        )
        self.assertTrue(response["result"]["structuredContent"]["widget"]["open"]["replace_previous"])

    def test_tools_call_search_products_marks_no_results_and_widget_metadata(self) -> None:
        registry = create_tool_registry()
        with patch(
            "app.interfaces.mcp.server.search_products",
            return_value={"query": "none", "count": 0, "products": []},
        ):
            response = handle_rpc_request(
                {
                    "jsonrpc": "2.0",
                    "id": 301,
                    "method": "tools/call",
                    "params": {
                        "name": "search_products",
                        "arguments": {"query": "none", "limit": 5},
                    },
                },
                registry=registry,
            )

        structured = response["result"]["structuredContent"]
        self.assertTrue(structured["no_results"])
        self.assertEqual(structured["products"], [])
        self.assertEqual(structured["widget"]["open"]["template"], "ui://widget/products.html")

    def test_tools_call_search_products_uses_ttl_cache(self) -> None:
        registry = create_tool_registry()
        with patch(
            "app.interfaces.mcp.server.search_products",
            return_value={"query": "цитрамон", "count": 1, "products": [{"id": 1}]},
        ) as mocked_search:
            first = handle_rpc_request(
                {
                    "jsonrpc": "2.0",
                    "id": 31,
                    "method": "tools/call",
                    "params": {"name": "search_products", "arguments": {"query": "цитрамон", "limit": 5}},
                },
                registry=registry,
            )
            second = handle_rpc_request(
                {
                    "jsonrpc": "2.0",
                    "id": 32,
                    "method": "tools/call",
                    "params": {"name": "search_products", "arguments": {"query": "цитрамон", "limit": 5}},
                },
                registry=registry,
            )

        mocked_search.assert_called_once_with("цитрамон", limit=5)
        self.assertEqual(first["result"]["structuredContent"]["count"], 1)
        self.assertEqual(second["result"]["structuredContent"]["count"], 1)

    def test_tools_call_search_products_cache_expires_after_ttl(self) -> None:
        registry = create_tool_registry()
        with patch(
            "app.interfaces.mcp.server.search_products",
            return_value={"query": "цитрамон", "count": 1, "products": [{"id": 1}]},
        ) as mocked_search:
            with patch("app.interfaces.mcp.server._monotonic", side_effect=[0.0, 0.0, 60.0, 60.0]):
                handle_rpc_request(
                    {
                        "jsonrpc": "2.0",
                        "id": 41,
                        "method": "tools/call",
                        "params": {"name": "search_products", "arguments": {"query": "цитрамон", "limit": 5}},
                    },
                    registry=registry,
                )
                handle_rpc_request(
                    {
                        "jsonrpc": "2.0",
                        "id": 42,
                        "method": "tools/call",
                        "params": {"name": "search_products", "arguments": {"query": "цитрамон", "limit": 5}},
                    },
                    registry=registry,
                )

        self.assertEqual(mocked_search.call_count, 2)

    def test_tools_call_search_products_cache_respects_max_entries_lru(self) -> None:
        registry = create_tool_registry()
        with patch(
            "app.interfaces.mcp.server.get_settings",
            return_value=SimpleNamespace(
                mcp_search_cache_ttl_seconds=30.0,
                mcp_tracking_cache_ttl_seconds=10.0,
                mcp_tool_cache_max_entries=2,
            ),
        ):
            with patch(
                "app.interfaces.mcp.server.search_products",
                side_effect=[
                    {"query": "q1", "count": 1, "products": [{"id": 1}]},
                    {"query": "q2", "count": 1, "products": [{"id": 2}]},
                    {"query": "q3", "count": 1, "products": [{"id": 3}]},
                    {"query": "q1", "count": 1, "products": [{"id": 1}]},
                ],
            ) as mocked_search:
                with patch(
                    "app.interfaces.mcp.server._monotonic",
                    side_effect=[0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0],
                ):
                    for request_id, query in [(51, "q1"), (52, "q2"), (53, "q3"), (54, "q1")]:
                        handle_rpc_request(
                            {
                                "jsonrpc": "2.0",
                                "id": request_id,
                                "method": "tools/call",
                                "params": {"name": "search_products", "arguments": {"query": query}},
                            },
                            registry=registry,
                        )

        self.assertEqual(mocked_search.call_count, 4)

    def test_cache_policy_reads_ttl_from_settings(self) -> None:
        _reset_server_caches_for_tests()
        with patch(
            "app.interfaces.mcp.server.get_settings",
            return_value=SimpleNamespace(
                mcp_search_cache_ttl_seconds=42.0,
                mcp_tracking_cache_ttl_seconds=11.0,
                mcp_tool_cache_max_entries=2,
            ),
        ):
            registry = create_tool_registry()
            with patch(
                "app.interfaces.mcp.server.search_products",
                return_value={"query": "q", "count": 1, "products": [{"id": 1}]},
            ) as mocked_search:
                handle_rpc_request(
                    {
                        "jsonrpc": "2.0",
                        "id": 81,
                        "method": "tools/call",
                        "params": {"name": "search_products", "arguments": {"query": "q"}},
                    },
                    registry=registry,
                )
                handle_rpc_request(
                    {
                        "jsonrpc": "2.0",
                        "id": 82,
                        "method": "tools/call",
                        "params": {"name": "search_products", "arguments": {"query": "q"}},
                    },
                    registry=registry,
                )
        mocked_search.assert_called_once_with("q", limit=10)

    def test_tools_call_success_uses_compact_summary_text_content(self) -> None:
        registry = create_tool_registry()
        with patch(
            "app.interfaces.mcp.server.search_products",
            return_value={
                "query": "цитрамон",
                "count": 2,
                "products": [{"id": 1, "name": "A"}, {"id": 2, "name": "B"}],
                "extra": "x" * 1000,
            },
        ):
            response = handle_rpc_request(
                {
                    "jsonrpc": "2.0",
                    "id": 303,
                    "method": "tools/call",
                    "params": {
                        "name": "search_products",
                        "arguments": {"query": "цитрамон", "limit": 5},
                    },
                },
                registry=registry,
            )

        text_content = response["result"]["content"][0]["text"]
        self.assertIn('"ok":true', text_content)
        self.assertIn('"count":2', text_content)
        self.assertNotIn('"products":[', text_content)
        self.assertNotIn("xxxxxxxx", text_content)

    def test_unknown_method_returns_jsonrpc_error(self) -> None:
        registry = create_tool_registry()

        response = handle_rpc_request(
            {"jsonrpc": "2.0", "id": 4, "method": "unknown/method", "params": {}},
            registry=registry,
        )

        self.assertIn("error", response)
        self.assertEqual(response["error"]["code"], -32601)

    def test_resources_list_contains_widget_templates(self) -> None:
        response = handle_rpc_request(
            {"jsonrpc": "2.0", "id": 401, "method": "resources/list", "params": {}},
        )
        self.assertIn("result", response)
        resources = response["result"]["resources"]
        products_resource = next(
            resource for resource in resources if resource["uri"] == "ui://widget/products.html"
        )
        self.assertEqual(products_resource["mimeType"], "text/html;profile=mcp-app")
        self.assertIn("_meta", products_resource)
        self.assertEqual(
            products_resource["_meta"]["openai/widgetDomain"],
            "https://subgerminal-yevette-lactogenic.ngrok-free.dev",
        )
        self.assertIn("openai/widgetCSP", products_resource["_meta"])

    def test_resources_read_returns_products_template_content(self) -> None:
        response = handle_rpc_request(
            {
                "jsonrpc": "2.0",
                "id": 402,
                "method": "resources/read",
                "params": {"uri": "ui://widget/products.html"},
            },
        )
        self.assertIn("result", response)
        contents = response["result"]["contents"]
        self.assertEqual(contents[0]["uri"], "ui://widget/products.html")
        self.assertEqual(contents[0]["mimeType"], "text/html;profile=mcp-app")
        self.assertIn("search-toolbar", contents[0]["text"])
        self.assertIn("_meta", contents[0])
        self.assertEqual(
            contents[0]["_meta"]["openai/widgetDomain"],
            "https://subgerminal-yevette-lactogenic.ngrok-free.dev",
        )
        self.assertEqual(
            contents[0]["_meta"]["openai/widgetCSP"]["resource_domains"],
            [
                "https://subgerminal-yevette-lactogenic.ngrok-free.dev",
                get_apteka_base_url(),
                "https://www.apteka.md",
                "https://cdn.jsdelivr.net",
            ],
        )
        self.assertEqual(
            contents[0]["_meta"]["openai/widgetCSP"]["connect_domains"],
            [get_apteka_base_url()],
        )


    def test_tools_call_requires_string_name(self) -> None:
        registry = create_tool_registry()

        response = handle_rpc_request(
            {
                "jsonrpc": "2.0",
                "id": 11,
                "method": "tools/call",
                "params": {"arguments": {"query": "цитрамон"}},
            },
            registry=registry,
        )

        self.assertIn("error", response)
        self.assertEqual(response["error"]["code"], -32602)

    def test_tools_call_requires_object_arguments(self) -> None:
        registry = create_tool_registry()

        response = handle_rpc_request(
            {
                "jsonrpc": "2.0",
                "id": 12,
                "method": "tools/call",
                "params": {"name": "search_products", "arguments": ["bad"]},
            },
            registry=registry,
        )

        self.assertIn("error", response)
        self.assertEqual(response["error"]["code"], -32602)

    def test_tools_call_validates_required_fields_from_schema(self) -> None:
        registry = create_tool_registry()

        response = handle_rpc_request(
            {
                "jsonrpc": "2.0",
                "id": 13,
                "method": "tools/call",
                "params": {"name": "search_products", "arguments": {}},
            },
            registry=registry,
        )

        self.assertIn("error", response)
        self.assertEqual(response["error"]["code"], -32602)

    def test_tools_call_validates_argument_types_from_schema(self) -> None:
        registry = create_tool_registry()

        response = handle_rpc_request(
            {
                "jsonrpc": "2.0",
                "id": 14,
                "method": "tools/call",
                "params": {"name": "search_products", "arguments": {"query": "q", "limit": "5"}},
            },
            registry=registry,
        )

        self.assertIn("error", response)
        self.assertEqual(response["error"]["code"], -32602)

    def test_tools_call_validates_any_of_for_add_to_cart(self) -> None:
        registry = create_tool_registry()

        response = handle_rpc_request(
            {
                "jsonrpc": "2.0",
                "id": 15,
                "method": "tools/call",
                "params": {"name": "add_to_my_cart", "arguments": {"cart_session_id": "sess-1"}},
            },
            registry=registry,
        )

        self.assertIn("error", response)
        self.assertEqual(response["error"]["code"], -32602)

    def test_tools_call_error_returns_structured_error_payload(self) -> None:
        registry = create_tool_registry()
        with patch(
            "app.interfaces.mcp.server.search_products",
            side_effect=TimeoutError("external api timeout"),
        ):
            response = handle_rpc_request(
                {
                    "jsonrpc": "2.0",
                    "id": "req-42",
                    "method": "tools/call",
                    "params": {
                        "name": "search_products",
                        "arguments": {"query": "цитрамон", "limit": 5},
                    },
                },
                registry=registry,
            )

        self.assertIn("result", response)
        self.assertTrue(response["result"]["isError"])
        self.assertIn("structuredContent", response["result"])
        error_payload = response["result"]["structuredContent"]["error"]
        self.assertEqual(error_payload["request_id"], "req-42")
        self.assertEqual(error_payload["retriable"], True)
        self.assertEqual(error_payload["type"], "timeout_error")
        self.assertIn("timeout", error_payload["message"].lower())

    def test_tools_call_error_logs_with_request_id(self) -> None:
        registry = create_tool_registry()
        with patch(
            "app.interfaces.mcp.server.search_products",
            side_effect=RuntimeError("boom"),
        ):
            with patch("app.interfaces.mcp.server.logger") as mocked_logger:
                response = handle_rpc_request(
                    {
                        "jsonrpc": "2.0",
                        "id": "req-log-1",
                        "method": "tools/call",
                        "params": {
                            "name": "search_products",
                            "arguments": {"query": "цитрамон", "limit": 5},
                        },
                    },
                    registry=registry,
                    http_request_id="http-req-1",
                )

        self.assertTrue(response["result"]["isError"])
        mocked_logger.exception.assert_called_once()
        _, kwargs = mocked_logger.exception.call_args
        self.assertIn("extra", kwargs)
        self.assertEqual(kwargs["extra"]["rpc_request_id"], "req-log-1")
        self.assertEqual(kwargs["extra"]["http_request_id"], "http-req-1")
        self.assertEqual(kwargs["extra"]["tool_name"], "search_products")

    def test_runtime_metrics_collect_tool_stats_and_cache_stats(self) -> None:
        registry = create_tool_registry()
        with patch(
            "app.interfaces.mcp.server.search_products",
            return_value={"query": "q", "count": 1, "products": [{"id": 1}]},
        ):
            handle_rpc_request(
                {
                    "jsonrpc": "2.0",
                    "id": 91,
                    "method": "tools/call",
                    "params": {"name": "search_products", "arguments": {"query": "q"}},
                },
                registry=registry,
            )
            handle_rpc_request(
                {
                    "jsonrpc": "2.0",
                    "id": 92,
                    "method": "tools/call",
                    "params": {"name": "search_products", "arguments": {"query": "q"}},
                },
                registry=registry,
            )

        metrics = get_runtime_metrics()
        self.assertGreaterEqual(int(metrics["rpc_requests_total"]), 2)
        self.assertGreaterEqual(int(metrics["tool_calls_total"]), 2)
        self.assertGreaterEqual(int(metrics["cache_hits_total"]), 1)
        self.assertGreaterEqual(int(metrics["cache_misses_total"]), 1)
        by_tool = metrics["tools"]
        self.assertIn("search_products", by_tool)
        self.assertGreaterEqual(int(by_tool["search_products"]["calls"]), 2)
        self.assertGreaterEqual(float(by_tool["search_products"]["avg_latency_ms"]), 0.0)

    def test_build_access_log_message_is_compact_and_human_readable(self) -> None:
        message = _build_access_log_message(
            method="POST",
            path="/mcp",
            status_code=200,
            latency_ms=1.237,
            client_ip="127.0.0.1",
            user_agent="openai-mcp/1.0.0",
            request_id="req-123",
        )

        self.assertEqual(
            message,
            "[REQ] POST /mcp -> 200 | 1.24 ms | ip=127.0.0.1 | ua=openai-mcp/1.0.0 | id=req-123",
        )

    def test_run_server_handles_keyboard_interrupt_gracefully(self) -> None:
        class FakeServer:
            def __init__(self) -> None:
                self.closed = False

            def serve_forever(self) -> None:
                raise KeyboardInterrupt()

            def server_close(self) -> None:
                self.closed = True

        fake_server = FakeServer()
        with patch("app.interfaces.mcp.server._configure_runtime_logging"):
            with patch("app.interfaces.mcp.server.ThreadingHTTPServer", return_value=fake_server):
                with patch("builtins.print") as mocked_print:
                    run_server(host="127.0.0.1", port=8000)

        self.assertTrue(fake_server.closed)
        mocked_print.assert_any_call("MCP server started on http://127.0.0.1:8000/mcp")
        mocked_print.assert_any_call("MCP server stopped gracefully.")

    def test_track_order_status_tool_description_mentions_supported_inputs(self) -> None:
        registry = create_tool_registry()

        tool = registry["track_order_status_ui"]

        self.assertIn("order number", tool.description.lower())
        self.assertIn("phone", tool.description.lower())
        self.assertIn("country code", tool.description.lower())
        self.assertIn("operator", tool.description.lower())
        self.assertIn("wait", tool.description.lower())
        self.assertIn("status_hint", tool.description.lower())
        self.assertIn("packed", tool.description.lower())

    def test_tools_call_track_order_status_delegates_to_tracking_tool(self) -> None:
        registry = create_tool_registry()
        with patch(
            "app.interfaces.mcp.server.track_order_status_ui",
            return_value={"lookup": "ORD-123", "count": 1, "orders": [{"status": "processing"}]},
        ) as mocked_tracking:
            response = handle_rpc_request(
                {
                    "jsonrpc": "2.0",
                    "id": 5,
                    "method": "tools/call",
                    "params": {
                        "name": "track_order_status_ui",
                        "arguments": {"lookup": "ORD-123"},
                    },
                },
                registry=registry,
            )

        mocked_tracking.assert_called_once_with("ORD-123")
        self.assertFalse(response["result"]["isError"])
        self.assertEqual(response["result"]["structuredContent"]["count"], 1)

    def test_tools_call_track_order_status_uses_ttl_cache(self) -> None:
        registry = create_tool_registry()
        with patch(
            "app.interfaces.mcp.server.track_order_status_ui",
            return_value={"lookup": "ORD-123", "count": 1, "orders": [{"status": "processing"}]},
        ) as mocked_tracking:
            first = handle_rpc_request(
                {
                    "jsonrpc": "2.0",
                    "id": 61,
                    "method": "tools/call",
                    "params": {"name": "track_order_status_ui", "arguments": {"lookup": "ORD-123"}},
                },
                registry=registry,
            )
            second = handle_rpc_request(
                {
                    "jsonrpc": "2.0",
                    "id": 62,
                    "method": "tools/call",
                    "params": {"name": "track_order_status_ui", "arguments": {"lookup": "ORD-123"}},
                },
                registry=registry,
            )

        mocked_tracking.assert_called_once_with("ORD-123")
        self.assertEqual(first["result"]["structuredContent"]["count"], 1)
        self.assertEqual(second["result"]["structuredContent"]["count"], 1)

    def test_tools_call_track_order_status_cache_expires_after_ttl(self) -> None:
        registry = create_tool_registry()
        with patch(
            "app.interfaces.mcp.server.track_order_status_ui",
            return_value={"lookup": "ORD-123", "count": 1, "orders": [{"status": "processing"}]},
        ) as mocked_tracking:
            with patch("app.interfaces.mcp.server._monotonic", side_effect=[0.0, 0.0, 20.0, 20.0]):
                handle_rpc_request(
                    {
                        "jsonrpc": "2.0",
                        "id": 71,
                        "method": "tools/call",
                        "params": {"name": "track_order_status_ui", "arguments": {"lookup": "ORD-123"}},
                    },
                    registry=registry,
                )
                handle_rpc_request(
                    {
                        "jsonrpc": "2.0",
                        "id": 72,
                        "method": "tools/call",
                        "params": {"name": "track_order_status_ui", "arguments": {"lookup": "ORD-123"}},
                    },
                    registry=registry,
                )

        self.assertEqual(mocked_tracking.call_count, 2)

    def test_support_knowledge_search_tool_description_mentions_use_cases(self) -> None:
        registry = create_tool_registry()

        tool = registry["support_knowledge_search"]

        self.assertIn("faq", tool.description.lower())
        self.assertIn("order", tool.description.lower())
        self.assertIn("work schedule", tool.description.lower())
        self.assertIn("app", tool.description.lower())

    def test_tools_call_support_knowledge_search_delegates_to_faq_tool(self) -> None:
        registry = create_tool_registry()
        with patch(
            "app.interfaces.mcp.server.faq_search",
            return_value={
                "query": "как оформить заказ",
                "count": 1,
                "chunks": [{"id": 10, "text": "Через корзину"}],
            },
        ) as mocked_faq_search:
            response = handle_rpc_request(
                {
                    "jsonrpc": "2.0",
                    "id": 9,
                    "method": "tools/call",
                    "params": {
                        "name": "support_knowledge_search",
                        "arguments": {"query": "как оформить заказ", "limit": 3},
                    },
                },
                registry=registry,
            )

        mocked_faq_search.assert_called_once_with("как оформить заказ", limit=3)
        self.assertFalse(response["result"]["isError"])
        self.assertEqual(response["result"]["structuredContent"]["count"], 1)

    def test_tools_call_my_cart_delegates_to_cart_tool(self) -> None:
        registry = create_tool_registry()
        with patch(
            "app.interfaces.mcp.server.my_cart",
            return_value={"cart_session_id": "sess-1", "count": 0, "items": []},
        ) as mocked_my_cart:
            response = handle_rpc_request(
                {
                    "jsonrpc": "2.0",
                    "id": 6,
                    "method": "tools/call",
                    "params": {
                        "name": "my_cart",
                        "arguments": {"cart_session_id": "sess-1"},
                    },
                },
                registry=registry,
            )

        mocked_my_cart.assert_called_once_with(cart_session_id="sess-1")
        self.assertFalse(response["result"]["isError"])
        self.assertEqual(response["result"]["structuredContent"]["cart_session_id"], "sess-1")

    def test_tools_call_add_to_my_cart_delegates_to_cart_tool(self) -> None:
        registry = create_tool_registry()
        with patch(
            "app.interfaces.mcp.server.add_to_my_cart",
            return_value={
                "cart_session_id": "sess-1",
                "count": 1,
                "items": [{"product_id": "A12", "quantity": 2}],
            },
        ) as mocked_add_to_cart:
            response = handle_rpc_request(
                {
                    "jsonrpc": "2.0",
                    "id": 7,
                    "method": "tools/call",
                    "params": {
                        "name": "add_to_my_cart",
                        "arguments": {
                            "cart_session_id": "sess-1",
                            "product_id": "A12",
                            "quantity": 2,
                            "name": "Aspirin Cardio",
                            "price": 31.17,
                            "discount_price": 29.99,
                            "manufacturer": "Bayer",
                        },
                    },
                },
                registry=registry,
            )

        mocked_add_to_cart.assert_called_once_with(
            product_id="A12",
            quantity=2,
            items=None,
            cart_session_id="sess-1",
            use_add_endpoint=False,
            name="Aspirin Cardio",
            price=31.17,
            discount_price=29.99,
            manufacturer="Bayer",
            image_url=None,
        )
        self.assertFalse(response["result"]["isError"])
        self.assertEqual(response["result"]["structuredContent"]["count"], 1)

    def test_tools_call_add_to_my_cart_batch_items_delegates_to_cart_tool(self) -> None:
        registry = create_tool_registry()
        with patch(
            "app.interfaces.mcp.server.add_to_my_cart",
            return_value={
                "cart_session_id": "sess-1",
                "count": 2,
                "items": [
                    {"product_id": "16174", "quantity": 1},
                    {"product_id": "20859", "quantity": 1},
                ],
            },
        ) as mocked_add_to_cart:
            response = handle_rpc_request(
                {
                    "jsonrpc": "2.0",
                    "id": 8,
                    "method": "tools/call",
                    "params": {
                        "name": "add_to_my_cart",
                        "arguments": {
                            "cart_session_id": "sess-1",
                            "items": [{"product_id": "20859", "quantity": 1}],
                        },
                    },
                },
                registry=registry,
            )

        mocked_add_to_cart.assert_called_once_with(
            product_id=None,
            quantity=None,
            items=[{"product_id": "20859", "quantity": 1}],
            cart_session_id="sess-1",
            use_add_endpoint=False,
            name=None,
            price=None,
            discount_price=None,
            manufacturer=None,
            image_url=None,
        )
        self.assertFalse(response["result"]["isError"])
        self.assertEqual(response["result"]["structuredContent"]["count"], 2)

    def test_tools_call_add_to_my_cart_passes_use_add_endpoint_flag(self) -> None:
        registry = create_tool_registry()
        with patch(
            "app.interfaces.mcp.server.add_to_my_cart",
            return_value={
                "cart_session_id": "sess-1",
                "count": 1,
                "items": [{"product_id": "A12", "quantity": 1}],
            },
        ) as mocked_add_to_cart:
            response = handle_rpc_request(
                {
                    "jsonrpc": "2.0",
                    "id": 81,
                    "method": "tools/call",
                    "params": {
                        "name": "add_to_my_cart",
                        "arguments": {
                            "cart_session_id": "sess-1",
                            "product_id": "A12",
                            "use_add_endpoint": True,
                        },
                    },
                },
                registry=registry,
            )

        mocked_add_to_cart.assert_called_once_with(
            product_id="A12",
            quantity=None,
            items=None,
            cart_session_id="sess-1",
            use_add_endpoint=True,
            name=None,
            price=None,
            discount_price=None,
            manufacturer=None,
            image_url=None,
        )
        self.assertFalse(response["result"]["isError"])
        self.assertEqual(response["result"]["structuredContent"]["count"], 1)

    def test_tools_call_my_cart_uses_subject_fallback_when_arguments_empty(self) -> None:
        registry = create_tool_registry()
        subject = (
            "v1/34yeCSl5CCCNz0KLyvYVFk2it1LGMK0y67ZuwKsBpdoIn7ausSnvPLX6i1hgBmqGv4eQehFzqnghSPxOY0J0nlKVCycy1fbRV37qpp1"
        )
        expected_session_id = hashlib.sha256(f"cart:v1:{subject}".encode("utf-8")).hexdigest()

        with patch(
            "app.interfaces.mcp.server.my_cart",
            return_value={
                "cart_session_id": expected_session_id,
                "count": 1,
                "items": [{"product_id": "20859", "quantity": 1}],
            },
        ) as mocked_my_cart:
            my_cart_response = handle_rpc_request(
                {
                    "jsonrpc": "2.0",
                    "id": 802,
                    "method": "tools/call",
                    "params": {
                        "name": "my_cart",
                        "_meta": {"openai/subject": subject},
                        "arguments": {},
                    },
                },
                registry=registry,
            )

        mocked_my_cart.assert_called_once_with(cart_session_id=expected_session_id)
        self.assertFalse(my_cart_response["result"]["isError"])
        self.assertEqual(my_cart_response["result"]["structuredContent"]["count"], 1)

    def test_tools_call_add_to_my_cart_uses_subject_fallback_when_arguments_missing_session(self) -> None:
        registry = create_tool_registry()
        subject = (
            "v1/34yeCSl5CCCNz0KLyvYVFk2it1LGMK0y67ZuwKsBpdoIn7ausSnvPLX6i1hgBmqGv4eQehFzqnghSPxOY0J0nlKVCycy1fbRV37qpp1"
        )
        expected_session_id = hashlib.sha256(f"cart:v1:{subject}".encode("utf-8")).hexdigest()

        with patch(
            "app.interfaces.mcp.server.add_to_my_cart",
            return_value={
                "cart_session_id": expected_session_id,
                "count": 1,
                "items": [{"product_id": "20859", "quantity": 1}],
            },
        ) as mocked_add_to_my_cart:
            response = handle_rpc_request(
                {
                    "jsonrpc": "2.0",
                    "id": 813,
                    "method": "tools/call",
                    "params": {
                        "name": "add_to_my_cart",
                        "_meta": {"openai/subject": subject},
                        "arguments": {"product_id": "20859"},
                    },
                },
                registry=registry,
            )

        mocked_add_to_my_cart.assert_called_once_with(
            product_id="20859",
            quantity=None,
            items=None,
            cart_session_id=expected_session_id,
            use_add_endpoint=False,
            name=None,
            price=None,
            discount_price=None,
            manufacturer=None,
            image_url=None,
        )
        self.assertFalse(response["result"]["isError"])
        self.assertEqual(response["result"]["structuredContent"]["count"], 1)

    def test_tools_call_my_cart_prioritizes_arguments_over_subject_fallback(self) -> None:
        registry = create_tool_registry()
        subject = (
            "v1/34yeCSl5CCCNz0KLyvYVFk2it1LGMK0y67ZuwKsBpdoIn7ausSnvPLX6i1hgBmqGv4eQehFzqnghSPxOY0J0nlKVCycy1fbRV37qpp1"
        )

        with patch(
            "app.interfaces.mcp.server.my_cart",
            return_value={
                "cart_session_id": "6a6094887f254e18801aef64fa342348",
                "count": 0,
                "items": [],
            },
        ) as mocked_my_cart:
            my_cart_response = handle_rpc_request(
                {
                    "jsonrpc": "2.0",
                    "id": 812,
                    "method": "tools/call",
                    "params": {
                        "name": "my_cart",
                        "_meta": {"openai/subject": subject},
                        "arguments": {"cart_session_id": "6a6094887f254e18801aef64fa342348"},
                    },
                },
                registry=registry,
            )

        mocked_my_cart.assert_called_once_with(cart_session_id="6a6094887f254e18801aef64fa342348")
        self.assertFalse(my_cart_response["result"]["isError"])
        self.assertEqual(my_cart_response["result"]["structuredContent"]["count"], 0)

    def test_tools_call_my_cart_uses_shared_session_normalizer(self) -> None:
        registry = create_tool_registry()
        with patch(
            "app.interfaces.mcp.server.normalize_cart_session_id",
            wraps=lambda value: str(value).strip() or None,
        ) as mocked_normalize:
            with patch(
                "app.interfaces.mcp.server.my_cart",
                return_value={"cart_session_id": "session-123", "count": 0, "items": []},
            ):
                handle_rpc_request(
                    {
                        "jsonrpc": "2.0",
                        "id": 813,
                        "method": "tools/call",
                        "params": {
                            "name": "my_cart",
                            "arguments": {"cart_session_id": "  session-123  "},
                        },
                    },
                    registry=registry,
                )

        mocked_normalize.assert_called()

    def test_tools_call_checkout_order_delegates_to_checkout_tool(self) -> None:
        registry = create_tool_registry()
        with patch(
            "app.interfaces.mcp.server.checkout_order",
            return_value={"status": "delivery_method_selection", "cart_count": 1},
        ) as mocked_checkout:
            response = handle_rpc_request(
                {
                    "jsonrpc": "2.0",
                    "id": 10,
                    "method": "tools/call",
                    "params": {
                        "name": "checkout_order",
                        "arguments": {"cart_session_id": "sess-1"},
                    },
                },
                registry=registry,
            )

        mocked_checkout.assert_called_once_with(
            cart_session_id="sess-1",
            delivery_method=None,
            pickup_region_id=None,
            pickup_region_name=None,
            pickup_city_id=None,
            pickup_city_name=None,
            pickup_pharmacy_id=None,
            pickup_pharmacy_name=None,
            pickup_contact=None,
            courier_contact=None,
            courier_address=None,
            payment_method=None,
            dont_call_me=None,
            terms_accepted=None,
            comment=None,
        )
        self.assertFalse(response["result"]["isError"])
        self.assertEqual(response["result"]["structuredContent"]["status"], "delivery_method_selection")

    def test_checkout_order_schema_contains_courier_address_and_contact(self) -> None:
        registry = create_tool_registry()
        checkout_tool = registry["checkout_order"]
        properties = checkout_tool.input_schema["properties"]

        self.assertIn("courier_contact", properties)
        self.assertIn("courier_address", properties)


class MCPHttpTransportRequestIdTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._server = ThreadingHTTPServer(("127.0.0.1", 0), MCPHttpHandler)
        cls._host, cls._port = cls._server.server_address
        cls._thread = threading.Thread(target=cls._server.serve_forever, daemon=True)
        cls._thread.start()

    @classmethod
    def tearDownClass(cls) -> None:
        cls._server.shutdown()
        cls._server.server_close()
        cls._thread.join(timeout=2)

    def _post_mcp(self, payload: dict[str, object], headers: dict[str, str] | None = None):
        body = json.dumps(payload).encode("utf-8")
        request_headers = {
            "Content-Type": "application/json",
            "Content-Length": str(len(body)),
        }
        if headers:
            request_headers.update(headers)

        connection = http.client.HTTPConnection(self._host, self._port, timeout=5)
        connection.request("POST", "/mcp", body=body, headers=request_headers)
        response = connection.getresponse()
        raw_body = response.read()
        headers_map = {key.lower(): value for key, value in response.getheaders()}
        status_code = response.status
        connection.close()
        return status_code, headers_map, raw_body

    def _get(self, path: str, headers: dict[str, str] | None = None):
        connection = http.client.HTTPConnection(self._host, self._port, timeout=5)
        connection.request("GET", path, headers=headers or {})
        response = connection.getresponse()
        raw_body = response.read()
        headers_map = {key.lower(): value for key, value in response.getheaders()}
        status_code = response.status
        connection.close()
        return status_code, headers_map, raw_body

    def _post_raw(self, body: bytes, headers: dict[str, str] | None = None):
        request_headers = {"Content-Length": str(len(body))}
        if headers:
            request_headers.update(headers)
        connection = http.client.HTTPConnection(self._host, self._port, timeout=5)
        connection.request("POST", "/mcp", body=body, headers=request_headers)
        response = connection.getresponse()
        raw_body = response.read()
        headers_map = {key.lower(): value for key, value in response.getheaders()}
        status_code = response.status
        connection.close()
        return status_code, headers_map, raw_body

    def test_http_response_includes_incoming_x_request_id(self) -> None:
        status, headers, _ = self._post_mcp(
            {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
            headers={"X-Request-Id": "incoming-req-42"},
        )
        self.assertEqual(status, 200)
        self.assertEqual(headers.get("x-request-id"), "incoming-req-42")

    def test_http_response_includes_protocol_version_header(self) -> None:
        status, headers, _ = self._post_mcp(
            {"jsonrpc": "2.0", "id": 13, "method": "initialize", "params": {}},
            headers={"MCP-Protocol-Version": "2025-06-18"},
        )
        self.assertEqual(status, 200)
        self.assertEqual(headers.get("mcp-protocol-version"), "2025-06-18")

    def test_http_response_generates_x_request_id_when_missing(self) -> None:
        status, headers, _ = self._post_mcp(
            {"jsonrpc": "2.0", "id": 2, "method": "initialize", "params": {}},
        )
        self.assertEqual(status, 200)
        self.assertIn("x-request-id", headers)
        self.assertTrue(len(headers["x-request-id"]) >= 8)

    def test_http_response_uses_gzip_for_large_payload_when_accepted(self) -> None:
        with patch("app.interfaces.mcp.server.MIN_GZIP_BYTES", 1):
            status, headers, raw_body = self._post_mcp(
                {
                    "jsonrpc": "2.0",
                    "id": 3,
                    "method": "tools/call",
                    "params": {"name": "search_products", "arguments": {"query": "q"}},
                },
                headers={"Accept-Encoding": "gzip"},
            )
        self.assertEqual(status, 200)
        self.assertEqual(headers.get("content-encoding"), "gzip")
        decoded = gzip.decompress(raw_body).decode("utf-8")
        payload = json.loads(decoded)
        self.assertEqual(payload["id"], 3)
        self.assertIn("result", payload)

    def test_http_response_skips_gzip_when_not_accepted(self) -> None:
        status, headers, raw_body = self._post_mcp(
            {
                "jsonrpc": "2.0",
                "id": 4,
                "method": "tools/call",
                "params": {"name": "search_products", "arguments": {"query": "q"}},
            },
            headers={"Accept-Encoding": "br"},
        )
        self.assertEqual(status, 200)
        self.assertNotIn("content-encoding", headers)
        decoded = raw_body.decode("utf-8")
        payload = json.loads(decoded)
        self.assertEqual(payload["id"], 4)

    def test_http_metrics_endpoint_returns_runtime_snapshot(self) -> None:
        status_code, headers_map, raw_body = self._get("/metrics")

        self.assertEqual(status_code, 200)
        self.assertIn("x-request-id", headers_map)
        payload = json.loads(raw_body.decode("utf-8"))
        self.assertIn("rpc_requests_total", payload)
        self.assertIn("tools", payload)

    def test_http_get_mcp_returns_sse_when_requested(self) -> None:
        status_code, headers_map, raw_body = self._get(
            "/mcp",
            headers={"Accept": "text/event-stream"},
        )

        self.assertEqual(status_code, 200)
        self.assertIn("x-request-id", headers_map)
        self.assertIn("text/event-stream", headers_map.get("content-type", ""))
        self.assertIn(": connected", raw_body.decode("utf-8"))

    def test_http_get_mcp_returns_405_when_sse_not_requested(self) -> None:
        status_code, headers_map, _ = self._get("/mcp", headers={"Accept": "application/json"})

        self.assertEqual(status_code, 405)
        self.assertIn("x-request-id", headers_map)
        self.assertEqual(headers_map.get("allow"), "POST, GET")

    def test_http_post_mcp_can_stream_sse_response(self) -> None:
        status_code, headers_map, raw_body = self._post_mcp(
            {"jsonrpc": "2.0", "id": 11, "method": "initialize", "params": {}},
            headers={"Accept": "text/event-stream"},
        )

        self.assertEqual(status_code, 200)
        self.assertIn("x-request-id", headers_map)
        self.assertIn("text/event-stream", headers_map.get("content-type", ""))
        body_text = raw_body.decode("utf-8")
        self.assertIn("data: ", body_text)
        payload = json.loads(body_text.removeprefix("data: ").strip())
        self.assertEqual(payload["id"], 11)
        self.assertIn("result", payload)

    def test_http_post_mcp_prefers_json_when_accept_allows_json(self) -> None:
        status_code, headers_map, raw_body = self._post_mcp(
            {"jsonrpc": "2.0", "id": 12, "method": "initialize", "params": {}},
            headers={"Accept": "text/event-stream, application/json"},
        )

        self.assertEqual(status_code, 200)
        self.assertIn("x-request-id", headers_map)
        self.assertIn("application/json", headers_map.get("content-type", ""))
        payload = json.loads(raw_body.decode("utf-8"))
        self.assertEqual(payload["id"], 12)
        self.assertIn("result", payload)

    def test_http_post_mcp_returns_sse_error_when_sse_only_requested(self) -> None:
        status_code, headers_map, raw_body = self._post_raw(
            b'{"jsonrpc":"2.0","id":14,"method":"initialize","params":{}}',
            headers={"Accept": "text/event-stream"},
        )

        self.assertEqual(status_code, 415)
        self.assertIn("text/event-stream", headers_map.get("content-type", ""))
        body_text = raw_body.decode("utf-8")
        self.assertIn("data: ", body_text)
        payload = json.loads(body_text.removeprefix("data: ").strip())
        self.assertEqual(payload["error"]["code"], -32600)

    def test_http_request_writes_access_log_with_request_details(self) -> None:
        with patch("app.interfaces.mcp.server.logger") as mocked_logger:
            status, headers, _ = self._post_mcp(
                {"jsonrpc": "2.0", "id": 7, "method": "initialize", "params": {}},
                headers={"X-Request-Id": "req-access-1", "User-Agent": "test-agent"},
            )

        self.assertEqual(status, 200)
        self.assertEqual(headers.get("x-request-id"), "req-access-1")
        mocked_logger.info.assert_called()
        _, kwargs = mocked_logger.info.call_args
        self.assertIn("extra", kwargs)
        self.assertEqual(kwargs["extra"]["method"], "POST")
        self.assertEqual(kwargs["extra"]["path"], "/mcp")
        self.assertEqual(kwargs["extra"]["status_code"], 200)
        self.assertEqual(kwargs["extra"]["request_id"], "req-access-1")
        self.assertEqual(kwargs["extra"]["user_agent"], "test-agent")
        self.assertEqual(kwargs["extra"]["client_ip"], "127.0.0.1")


if __name__ == "__main__":
    unittest.main()
