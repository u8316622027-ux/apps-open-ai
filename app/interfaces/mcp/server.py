"""Minimal MCP HTTP server with tool registration."""

from __future__ import annotations

import argparse
from collections import OrderedDict
import gzip
import hashlib
import json
import logging
import os
from pathlib import Path
import threading
from functools import lru_cache
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from time import monotonic as _monotonic
from time import perf_counter as _perf_counter
from typing import Any, Callable
from uuid import uuid4

from app.core.config import get_settings
from app.interfaces.mcp.tool_registry import (
    ToolDefinition,
    create_tool_registry as _create_tool_registry_base,
    decorate_tool_result,
    serialize_tool_definition,
)
from app.interfaces.mcp.tools.cart_tools import add_to_my_cart, my_cart
from app.interfaces.mcp.tools.checkout_tools import checkout_order
from app.interfaces.mcp.tools.faq_tools import faq_search
from app.interfaces.mcp.tools.search_tools import search_products
from app.interfaces.mcp.tools.shared_context import normalize_cart_session_id
from app.interfaces.mcp.tools.tracking_tools import track_order_status_ui

MAX_REQUEST_BODY_BYTES = 1024 * 1024
MIN_GZIP_BYTES = 512
TOOL_RESPONSE_CACHE_TTL_SECONDS = 30.0
TRACKING_RESPONSE_CACHE_TTL_SECONDS = 10.0
TOOL_RESPONSE_CACHE_MAX_ENTRIES = 256
MCP_PROTOCOL_VERSION = "2025-06-18"
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())
_TOOL_RESPONSE_CACHE_LOCK = threading.Lock()
_TOOL_RESPONSE_CACHE: OrderedDict[str, tuple[float, dict[str, Any]]] = OrderedDict()
_RUNTIME_METRICS_LOCK = threading.Lock()
_RUNTIME_METRICS: dict[str, Any] = {
    "rpc_requests_total": 0,
    "tool_calls_total": 0,
    "tool_errors_total": 0,
    "cache_hits_total": 0,
    "cache_misses_total": 0,
    "tools": {},
}


def _search_products_handler(arguments: dict[str, Any]) -> dict[str, Any]:
    query = str(arguments.get("query", ""))
    limit = int(arguments.get("limit", 10))
    return search_products(query, limit=limit)


def _my_cart_handler(arguments: dict[str, Any]) -> dict[str, Any]:
    cart_session_id = arguments.get("cart_session_id")
    return my_cart(cart_session_id=str(cart_session_id) if cart_session_id is not None else None)


def _add_to_my_cart_handler(arguments: dict[str, Any]) -> dict[str, Any]:
    product_id_value = arguments.get("product_id")
    product_id = str(product_id_value) if product_id_value is not None else None
    quantity_value = arguments.get("quantity")
    quantity = int(quantity_value) if quantity_value is not None else None
    items_value = arguments.get("items")
    items = items_value if isinstance(items_value, list) else None
    cart_session_id = arguments.get("cart_session_id")
    name = arguments.get("name")
    price = arguments.get("price")
    discount_price = arguments.get("discount_price")
    manufacturer = arguments.get("manufacturer")
    image_url = arguments.get("image_url")
    use_add_endpoint = bool(arguments.get("use_add_endpoint")) if isinstance(
        arguments.get("use_add_endpoint"), bool
    ) else False
    return add_to_my_cart(
        product_id=product_id,
        quantity=quantity,
        items=items,
        cart_session_id=str(cart_session_id) if cart_session_id is not None else None,
        use_add_endpoint=use_add_endpoint,
        name=str(name) if isinstance(name, str) else None,
        price=float(price) if isinstance(price, (int, float)) and not isinstance(price, bool) else None,
        discount_price=(
            float(discount_price)
            if isinstance(discount_price, (int, float)) and not isinstance(discount_price, bool)
            else None
        ),
        manufacturer=str(manufacturer) if isinstance(manufacturer, str) else None,
        image_url=str(image_url) if isinstance(image_url, str) else None,
    )


def _track_order_status_ui_handler(arguments: dict[str, Any]) -> dict[str, Any]:
    lookup = str(arguments.get("lookup", ""))
    return track_order_status_ui(lookup)


def _support_knowledge_search_handler(arguments: dict[str, Any]) -> dict[str, Any]:
    query = str(arguments.get("query", ""))
    limit_value = arguments.get("limit")
    limit = int(limit_value) if limit_value is not None else None
    return faq_search(query, limit=limit)


def _checkout_order_handler(arguments: dict[str, Any]) -> dict[str, Any]:
    cart_session_id = arguments.get("cart_session_id")
    delivery_method = arguments.get("delivery_method")
    pickup_region_id = arguments.get("pickup_region_id")
    pickup_region_name = arguments.get("pickup_region_name")
    pickup_city_id = arguments.get("pickup_city_id")
    pickup_city_name = arguments.get("pickup_city_name")
    pickup_pharmacy_id = arguments.get("pickup_pharmacy_id")
    pickup_pharmacy_name = arguments.get("pickup_pharmacy_name")
    pickup_contact = arguments.get("pickup_contact")
    courier_contact = arguments.get("courier_contact")
    courier_address = arguments.get("courier_address")
    payment_method = arguments.get("payment_method")
    dont_call_me = arguments.get("dont_call_me")
    terms_accepted = arguments.get("terms_accepted")
    comment = arguments.get("comment")
    return checkout_order(
        cart_session_id=str(cart_session_id) if cart_session_id is not None else None,
        delivery_method=str(delivery_method) if delivery_method is not None else None,
        pickup_region_id=pickup_region_id,
        pickup_region_name=str(pickup_region_name) if pickup_region_name is not None else None,
        pickup_city_id=pickup_city_id,
        pickup_city_name=str(pickup_city_name) if pickup_city_name is not None else None,
        pickup_pharmacy_id=pickup_pharmacy_id,
        pickup_pharmacy_name=str(pickup_pharmacy_name) if pickup_pharmacy_name is not None else None,
        pickup_contact=pickup_contact if isinstance(pickup_contact, dict) else None,
        courier_contact=courier_contact if isinstance(courier_contact, dict) else None,
        courier_address=courier_address if isinstance(courier_address, dict) else None,
        payment_method=str(payment_method) if payment_method is not None else None,
        dont_call_me=bool(dont_call_me) if isinstance(dont_call_me, bool) else None,
        terms_accepted=bool(terms_accepted) if isinstance(terms_accepted, bool) else None,
        comment=str(comment) if comment is not None else None,
    )


def create_tool_registry() -> dict[str, ToolDefinition]:
    registry = _create_tool_registry_base()
    handler_by_name: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {
        "search_products": _search_products_handler,
        "add_to_my_cart": _add_to_my_cart_handler,
        "checkout_order": _checkout_order_handler,
        "support_knowledge_search": _support_knowledge_search_handler,
        "my_cart": _my_cart_handler,
        "track_order_status_ui": _track_order_status_ui_handler,
    }
    remapped: dict[str, ToolDefinition] = {}
    for name, tool in registry.items():
        handler = handler_by_name.get(name, tool.handler)
        remapped[name] = ToolDefinition(
            name=tool.name,
            description=tool.description,
            input_schema=tool.input_schema,
            handler=handler,
            output_template=tool.output_template,
            ui=tool.ui,
        )
    return remapped


@lru_cache(maxsize=1)
def _get_default_tool_registry() -> dict[str, ToolDefinition]:
    return create_tool_registry()


@lru_cache(maxsize=1)
def _get_default_tools_list_payload() -> dict[str, Any]:
    registry = _get_default_tool_registry()
    return {
        "tools": [serialize_tool_definition(tool) for tool in registry.values()]
    }


def _tool_uses_cart_session(tool_name: str) -> bool:
    return tool_name in {"my_cart", "add_to_my_cart", "checkout_order"}


def _derive_cart_session_id_from_subject(subject: str | None) -> str | None:
    if subject is None:
        return None
    normalized_subject = subject.strip()
    if not normalized_subject:
        return None
    digest = hashlib.sha256(f"cart:v1:{normalized_subject}".encode("utf-8")).hexdigest()
    return digest


def _extract_openai_subject_from_params(params: dict[str, Any]) -> str | None:
    meta = params.get("_meta")
    if not isinstance(meta, dict):
        return None
    value = meta.get("openai/subject")
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def _resolve_tool_arguments_with_cart_session_fallback(
    *,
    tool_name: str,
    arguments: dict[str, Any],
    params: dict[str, Any],
) -> dict[str, Any]:
    if not _tool_uses_cart_session(tool_name):
        return arguments

    incoming_session_id = normalize_cart_session_id(arguments.get("cart_session_id"))
    if incoming_session_id is not None:
        resolved_arguments = dict(arguments)
        resolved_arguments["cart_session_id"] = incoming_session_id
        return resolved_arguments

    subject = _extract_openai_subject_from_params(params)
    derived_session_id = _derive_cart_session_id_from_subject(subject)
    if derived_session_id is None:
        return arguments

    resolved_arguments = dict(arguments)
    resolved_arguments["cart_session_id"] = derived_session_id
    logger.debug(
        "mcp_cart_session_resolved_from_subject",
        extra={
            "tool_name": tool_name,
            "cart_session_source": "openai_subject_hash",
            "cart_session_id_prefix": derived_session_id[:12],
        },
    )
    return resolved_arguments


def _reset_server_caches_for_tests() -> None:
    _get_default_tool_registry.cache_clear()
    _get_default_tools_list_payload.cache_clear()
    _widget_resource_index.cache_clear()
    _get_tool_cache_config.cache_clear()
    with _TOOL_RESPONSE_CACHE_LOCK:
        _TOOL_RESPONSE_CACHE.clear()


def _reset_runtime_metrics_for_tests() -> None:
    with _RUNTIME_METRICS_LOCK:
        _RUNTIME_METRICS["rpc_requests_total"] = 0
        _RUNTIME_METRICS["tool_calls_total"] = 0
        _RUNTIME_METRICS["tool_errors_total"] = 0
        _RUNTIME_METRICS["cache_hits_total"] = 0
        _RUNTIME_METRICS["cache_misses_total"] = 0
        _RUNTIME_METRICS["tools"] = {}


def get_runtime_metrics() -> dict[str, Any]:
    with _RUNTIME_METRICS_LOCK:
        tools_snapshot: dict[str, dict[str, Any]] = {}
        tools = _RUNTIME_METRICS.get("tools", {})
        for tool_name, payload in tools.items():
            calls = int(payload.get("calls", 0))
            latency_total_ms = float(payload.get("latency_total_ms", 0.0))
            avg_latency_ms = (latency_total_ms / calls) if calls > 0 else 0.0
            tools_snapshot[tool_name] = {
                "calls": calls,
                "errors": int(payload.get("errors", 0)),
                "avg_latency_ms": round(avg_latency_ms, 3),
            }
        return {
            "rpc_requests_total": int(_RUNTIME_METRICS["rpc_requests_total"]),
            "tool_calls_total": int(_RUNTIME_METRICS["tool_calls_total"]),
            "tool_errors_total": int(_RUNTIME_METRICS["tool_errors_total"]),
            "cache_hits_total": int(_RUNTIME_METRICS["cache_hits_total"]),
            "cache_misses_total": int(_RUNTIME_METRICS["cache_misses_total"]),
            "tools": tools_snapshot,
        }


def _record_rpc_request() -> None:
    with _RUNTIME_METRICS_LOCK:
        _RUNTIME_METRICS["rpc_requests_total"] = int(_RUNTIME_METRICS["rpc_requests_total"]) + 1


def _record_tool_result(
    tool_name: str, *, latency_ms: float, errored: bool, cache_hit: bool | None = None
) -> None:
    with _RUNTIME_METRICS_LOCK:
        _RUNTIME_METRICS["tool_calls_total"] = int(_RUNTIME_METRICS["tool_calls_total"]) + 1
        if errored:
            _RUNTIME_METRICS["tool_errors_total"] = int(_RUNTIME_METRICS["tool_errors_total"]) + 1
        if cache_hit is True:
            _RUNTIME_METRICS["cache_hits_total"] = int(_RUNTIME_METRICS["cache_hits_total"]) + 1
        if cache_hit is False:
            _RUNTIME_METRICS["cache_misses_total"] = int(_RUNTIME_METRICS["cache_misses_total"]) + 1

        tools = _RUNTIME_METRICS.setdefault("tools", {})
        tool_payload = tools.setdefault(
            tool_name,
            {"calls": 0, "errors": 0, "latency_total_ms": 0.0},
        )
        tool_payload["calls"] = int(tool_payload.get("calls", 0)) + 1
        if errored:
            tool_payload["errors"] = int(tool_payload.get("errors", 0)) + 1
        tool_payload["latency_total_ms"] = float(tool_payload.get("latency_total_ms", 0.0)) + latency_ms


def handle_rpc_request(
    request_payload: dict[str, Any],
    *,
    registry: dict[str, ToolDefinition] | None = None,
    http_request_id: str | None = None,
) -> dict[str, Any]:
    """Handle a single JSON-RPC request."""

    if not isinstance(request_payload, dict):
        return _rpc_error(None, -32600, "Invalid Request")
    _record_rpc_request()

    active_registry = registry or _get_default_tool_registry()
    request_id = request_payload.get("id")
    method = request_payload.get("method")
    raw_params = request_payload.get("params")
    if raw_params is None:
        params: dict[str, Any] = {}
    elif isinstance(raw_params, dict):
        params = raw_params
    else:
        return _rpc_error(request_id, -32602, "Invalid params: params must be an object")

    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}, "resources": {}},
                "serverInfo": {"name": "apteka-mcp", "version": "0.1.0"},
            },
        }

    if method == "resources/list":
        resources = []
        for uri, resource_data in _widget_resource_index().items():
            relative_path = str(resource_data["path"])
            description = str(resource_data["description"])
            ui_domain = str(resource_data["domain"])
            resource_domains = list(resource_data["resource_domains"])
            connect_domains = list(resource_data["connect_domains"])
            resources.append(
                {
                    "uri": uri,
                    "name": relative_path,
                    "mimeType": "text/html;profile=mcp-app",
                    "description": description,
                    "_meta": {
                        "openai/widgetDescription": description,
                        "openai/widgetDomain": ui_domain,
                        "openai/widgetCSP": {
                            "connect_domains": connect_domains,
                            "resource_domains": resource_domains,
                        },
                    },
                }
            )
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {"resources": resources},
        }

    if method == "resources/read":
        uri = params.get("uri")
        if not isinstance(uri, str) or not uri.strip():
            return _rpc_error(request_id, -32602, "Invalid params: uri must be a non-empty string")

        resource_index = _widget_resource_index()
        normalized_uri = uri.strip()
        resource_data = resource_index.get(normalized_uri)
        if resource_data is None:
            return _rpc_error(request_id, -32602, f"Resource not found: {normalized_uri}")

        relative_path = str(resource_data["path"])
        file_path = Path(__file__).resolve().parents[2] / "widgets" / relative_path
        try:
            html_text = file_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return _rpc_error(request_id, -32602, f"Resource not found: {normalized_uri}")

        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "contents": [
                    {
                        "uri": normalized_uri,
                        "mimeType": "text/html;profile=mcp-app",
                        "text": html_text,
                        "_meta": {
                            "openai/widgetDescription": str(resource_data["description"]),
                            "openai/widgetDomain": str(resource_data["domain"]),
                            "openai/widgetCSP": {
                                "connect_domains": list(resource_data["connect_domains"]),
                                "resource_domains": list(resource_data["resource_domains"]),
                            },
                        },
                    }
                ]
            },
        }

    if method == "tools/list":
        if registry is None:
            tools_result = _get_default_tools_list_payload()
        else:
            tools_result = {"tools": [serialize_tool_definition(tool) for tool in active_registry.values()]}
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": tools_result,
        }

    if method == "tools/call":
        tool_name = params.get("name")
        if not isinstance(tool_name, str) or not tool_name.strip():
            return _rpc_error(request_id, -32602, "Invalid params: name must be a non-empty string")

        arguments = params.get("arguments") or {}
        if not isinstance(arguments, dict):
            return _rpc_error(request_id, -32602, "Invalid params: arguments must be an object")

        tool = active_registry.get(tool_name)
        if tool is None:
            return _rpc_error(request_id, -32601, f"Tool not found: {tool_name}")

        resolved_arguments = _resolve_tool_arguments_with_cart_session_fallback(
            tool_name=tool_name,
            arguments=arguments,
            params=params,
        )

        validation_error = _validate_input_schema(resolved_arguments, tool.input_schema)
        if validation_error is not None:
            return _rpc_error(request_id, -32602, f"Invalid params: {validation_error}")

        cache_ttl_seconds = _get_tool_cache_ttl_seconds(tool_name)
        cache_key = _build_tool_cache_key(tool_name, resolved_arguments)
        started_at = _perf_counter()
        if cache_ttl_seconds is not None and cache_key is not None:
            cached_payload = _get_cached_tool_payload(cache_key)
            if cached_payload is not None:
                structured_payload = decorate_tool_result(tool_name, tool, cached_payload)
                _record_tool_result(
                    tool_name,
                    latency_ms=(_perf_counter() - started_at) * 1000.0,
                    errored=False,
                    cache_hit=True,
                )
                return {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {
                        "content": [{"type": "text", "text": _build_tool_success_text(structured_payload)}],
                        "structuredContent": structured_payload,
                        "isError": False,
                    },
                }

        try:
            result_payload = tool.handler(resolved_arguments)
            structured_payload = decorate_tool_result(tool_name, tool, result_payload)
            if cache_ttl_seconds is not None and cache_key is not None:
                _set_cached_tool_payload(
                    cache_key,
                    structured_payload,
                    ttl_seconds=cache_ttl_seconds,
                )
            _record_tool_result(
                tool_name,
                latency_ms=(_perf_counter() - started_at) * 1000.0,
                errored=False,
                cache_hit=False if cache_key is not None else None,
            )
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "content": [{"type": "text", "text": _build_tool_success_text(structured_payload)}],
                    "structuredContent": structured_payload,
                    "isError": False,
                },
            }
        except Exception as exc:  # noqa: BLE001
            _record_tool_result(
                tool_name,
                latency_ms=(_perf_counter() - started_at) * 1000.0,
                errored=True,
                cache_hit=False if cache_key is not None else None,
            )
            error_payload = _classify_tool_error(exc)
            logger.exception(
                "mcp_tool_call_failed",
                extra={
                    "http_request_id": http_request_id,
                    "rpc_request_id": request_id,
                    "tool_name": tool_name,
                    "error_type": error_payload["type"],
                    "retriable": error_payload["retriable"],
                },
            )
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "content": [{"type": "text", "text": str(exc)}],
                    "structuredContent": {
                        "error": {
                            "type": error_payload["type"],
                            "message": error_payload["message"],
                            "retriable": error_payload["retriable"],
                            "request_id": request_id,
                            "http_request_id": http_request_id,
                        }
                    },
                    "isError": True,
                },
            }

    return _rpc_error(request_id, -32601, f"Method not found: {method}")


def handle_jsonrpc_payload(
    request_payload: Any,
    *,
    registry: dict[str, ToolDefinition] | None = None,
    http_request_id: str | None = None,
) -> dict[str, Any] | list[dict[str, Any]] | None:
    """Handle single or batch JSON-RPC payload and support notifications."""

    if isinstance(request_payload, list):
        if not request_payload:
            return _rpc_error(None, -32600, "Invalid Request")

        responses: list[dict[str, Any]] = []
        for item in request_payload:
            if isinstance(item, dict) and "id" not in item:
                handle_rpc_request(item, registry=registry, http_request_id=http_request_id)
                continue

            response_payload = handle_rpc_request(
                item, registry=registry, http_request_id=http_request_id
            )
            responses.append(response_payload)

        return responses or None

    if isinstance(request_payload, dict) and "id" not in request_payload:
        handle_rpc_request(request_payload, registry=registry, http_request_id=http_request_id)
        return None

    return handle_rpc_request(request_payload, registry=registry, http_request_id=http_request_id)


def _rpc_error(request_id: Any, code: int, message: str) -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "error": {"code": code, "message": message},
    }


@lru_cache(maxsize=1)
def _widget_resource_index() -> dict[str, dict[str, Any]]:
    registry = _get_default_tool_registry()
    mapping: dict[str, dict[str, Any]] = {}
    for tool in registry.values():
        uri = tool.output_template.strip()
        if not uri.startswith("ui://widget/"):
            continue
        relative_path = uri.removeprefix("ui://widget/").strip()
        if not relative_path:
            continue
        ui_domain = str(tool.ui.get("domain") or "").strip()
        csp = tool.ui.get("csp") if isinstance(tool.ui.get("csp"), dict) else {}
        resource_domains = csp.get("resourceDomains")
        if not isinstance(resource_domains, list):
            resource_domains = []
        connect_domains = csp.get("connectDomains")
        if not isinstance(connect_domains, list):
            connect_domains = []
        normalized_resource_domains = [str(domain).strip() for domain in resource_domains if str(domain).strip()]
        normalized_connect_domains = [str(domain).strip() for domain in connect_domains if str(domain).strip()]
        mapping[uri] = {
            "path": relative_path,
            "description": tool.description,
            "domain": ui_domain,
            "resource_domains": normalized_resource_domains,
            "connect_domains": normalized_connect_domains,
        }
    return mapping


def _validate_input_schema(arguments: dict[str, Any], schema: dict[str, Any]) -> str | None:
    return _validate_value(arguments, schema, path="arguments")


def _validate_value(value: Any, schema: dict[str, Any], *, path: str) -> str | None:
    expected_type = schema.get("type")
    if expected_type == "object":
        if not isinstance(value, dict):
            return f"{path} must be an object"

        required = schema.get("required")
        if isinstance(required, list):
            for field_name in required:
                if isinstance(field_name, str) and field_name not in value:
                    return f"{path}.{field_name} is required"

        properties = schema.get("properties")
        if isinstance(properties, dict):
            for field_name, field_schema in properties.items():
                if field_name not in value or not isinstance(field_schema, dict):
                    continue
                nested_error = _validate_value(
                    value[field_name],
                    field_schema,
                    path=f"{path}.{field_name}",
                )
                if nested_error is not None:
                    return nested_error

        any_of = schema.get("anyOf")
        if isinstance(any_of, list) and any_of:
            if not any(_matches_schema_variant(value, item) for item in any_of):
                return f"{path} must satisfy at least one anyOf schema"
        return None

    if expected_type == "array":
        if not isinstance(value, list):
            return f"{path} must be an array"
        item_schema = schema.get("items")
        if isinstance(item_schema, dict):
            for index, item in enumerate(value):
                nested_error = _validate_value(item, item_schema, path=f"{path}[{index}]")
                if nested_error is not None:
                    return nested_error
        return None

    if expected_type == "string":
        if not isinstance(value, str):
            return f"{path} must be a string"

    if expected_type == "integer":
        if not isinstance(value, int) or isinstance(value, bool):
            return f"{path} must be an integer"
        minimum = schema.get("minimum")
        if isinstance(minimum, int) and value < minimum:
            return f"{path} must be greater than or equal to {minimum}"

    if expected_type == "boolean":
        if not isinstance(value, bool):
            return f"{path} must be a boolean"

    enum = schema.get("enum")
    if isinstance(enum, list) and value not in enum:
        return f"{path} must be one of {enum}"

    return None


def _matches_schema_variant(value: dict[str, Any], schema_variant: Any) -> bool:
    if not isinstance(schema_variant, dict):
        return False
    required = schema_variant.get("required")
    if isinstance(required, list):
        for field_name in required:
            if isinstance(field_name, str) and field_name not in value:
                return False
    return True


def _is_json_content_type(content_type: str | None) -> bool:
    if content_type is None:
        return False
    mime_type = content_type.split(";", 1)[0].strip().lower()
    return mime_type == "application/json"


def _classify_tool_error(exc: Exception) -> dict[str, Any]:
    message = str(exc).strip() or exc.__class__.__name__
    if isinstance(exc, TimeoutError):
        return {"type": "timeout_error", "message": message, "retriable": True}
    if isinstance(exc, ConnectionError):
        return {"type": "connection_error", "message": message, "retriable": True}
    if isinstance(exc, ValueError):
        return {"type": "validation_error", "message": message, "retriable": False}
    return {"type": "tool_execution_error", "message": message, "retriable": False}


def _build_tool_cache_key(tool_name: str, arguments: dict[str, Any]) -> str | None:
    cache_policy = _get_tool_cache_config()["ttl_by_tool_name"]
    if tool_name not in cache_policy:
        return None
    encoded_args = json.dumps(arguments, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return f"{tool_name}:{encoded_args}"


def _get_tool_cache_ttl_seconds(tool_name: str) -> float | None:
    ttl_seconds = _get_tool_cache_config()["ttl_by_tool_name"].get(tool_name)
    if ttl_seconds is None or ttl_seconds <= 0:
        return None
    return ttl_seconds


@lru_cache(maxsize=1)
def _get_tool_cache_config() -> dict[str, Any]:
    settings = get_settings()
    search_ttl = float(getattr(settings, "mcp_search_cache_ttl_seconds", TOOL_RESPONSE_CACHE_TTL_SECONDS))
    tracking_ttl = float(
        getattr(settings, "mcp_tracking_cache_ttl_seconds", TRACKING_RESPONSE_CACHE_TTL_SECONDS)
    )
    max_entries = int(getattr(settings, "mcp_tool_cache_max_entries", TOOL_RESPONSE_CACHE_MAX_ENTRIES))
    if search_ttl <= 0:
        search_ttl = TOOL_RESPONSE_CACHE_TTL_SECONDS
    if tracking_ttl <= 0:
        tracking_ttl = TRACKING_RESPONSE_CACHE_TTL_SECONDS
    if max_entries <= 0:
        max_entries = TOOL_RESPONSE_CACHE_MAX_ENTRIES
    return {
        "ttl_by_tool_name": {
            "search_products": search_ttl,
            "track_order_status_ui": tracking_ttl,
        },
        "max_entries": max_entries,
    }


def _get_cached_tool_payload(cache_key: str) -> dict[str, Any] | None:
    now = _monotonic()
    with _TOOL_RESPONSE_CACHE_LOCK:
        record = _TOOL_RESPONSE_CACHE.get(cache_key)
        if record is None:
            return None
        expires_at, payload = record
        if now >= expires_at:
            _TOOL_RESPONSE_CACHE.pop(cache_key, None)
            return None
        _TOOL_RESPONSE_CACHE.move_to_end(cache_key)
        return payload


def _set_cached_tool_payload(
    cache_key: str,
    payload: dict[str, Any],
    *,
    ttl_seconds: float,
) -> None:
    expires_at = _monotonic() + ttl_seconds
    with _TOOL_RESPONSE_CACHE_LOCK:
        if cache_key in _TOOL_RESPONSE_CACHE:
            _TOOL_RESPONSE_CACHE.move_to_end(cache_key)
        _TOOL_RESPONSE_CACHE[cache_key] = (expires_at, payload)
        max_entries = int(_get_tool_cache_config()["max_entries"])
        while len(_TOOL_RESPONSE_CACHE) > max_entries:
            _TOOL_RESPONSE_CACHE.popitem(last=False)


def _build_tool_success_text(result_payload: dict[str, Any]) -> str:
    summary: dict[str, Any] = {"ok": True, "keys": sorted(result_payload.keys())}
    if "count" in result_payload:
        summary["count"] = result_payload.get("count")
    if "status" in result_payload:
        summary["status"] = result_payload.get("status")
    return json.dumps(summary, ensure_ascii=False, separators=(",", ":"))


def _resolve_http_request_id(incoming_request_id: str | None) -> str:
    if incoming_request_id is None:
        return uuid4().hex
    normalized = incoming_request_id.strip()
    if normalized:
        return normalized
    return uuid4().hex


def _build_access_log_message(
    *,
    method: str,
    path: str,
    status_code: int,
    latency_ms: float,
    client_ip: str,
    user_agent: str,
    request_id: str,
) -> str:
    return (
        f"[REQ] {method} {path} -> {status_code} | {latency_ms:.2f} ms | "
        f"ip={client_ip} | ua={user_agent} | id={request_id}"
    )


def _client_accepts_sse(accept_header: str | None) -> bool:
    if accept_header is None:
        return False
    return "text/event-stream" in accept_header.lower()


def _client_accepts_json(accept_header: str | None) -> bool:
    if accept_header is None:
        return True
    normalized = accept_header.lower()
    return "application/json" in normalized or "*/*" in normalized


def _resolve_mcp_protocol_version(protocol_version_header: str | None) -> str:
    if protocol_version_header is None:
        return MCP_PROTOCOL_VERSION
    normalized = protocol_version_header.strip()
    return normalized or MCP_PROTOCOL_VERSION


class MCPHttpHandler(BaseHTTPRequestHandler):
    """HTTP transport for minimal MCP JSON-RPC methods."""

    server_version = "AptekaMCP/0.1"
    protocol_version = "HTTP/1.1"

    def do_GET(self) -> None:  # noqa: N802
        started_at = _perf_counter()
        request_id = _resolve_http_request_id(self.headers.get("X-Request-Id"))
        protocol_version = _resolve_mcp_protocol_version(self.headers.get("MCP-Protocol-Version"))
        if self.path == "/mcp":
            if _client_accepts_sse(self.headers.get("Accept")):
                self._send_sse_comment(
                    "connected",
                    request_id=request_id,
                    protocol_version=protocol_version,
                )
                self._log_access(
                    request_id=request_id, status_code=HTTPStatus.OK, started_at=started_at
                )
                return
            self.send_response(HTTPStatus.METHOD_NOT_ALLOWED)
            self.send_header("Allow", "POST, GET")
            self.send_header("Content-Length", "0")
            self.send_header("X-Request-Id", request_id)
            self.send_header("MCP-Protocol-Version", protocol_version)
            self.end_headers()
            self._log_access(
                request_id=request_id,
                status_code=HTTPStatus.METHOD_NOT_ALLOWED,
                started_at=started_at,
            )
            return
        if self.path == "/health":
            self._send_json(
                {"status": "ok"},
                request_id=request_id,
                protocol_version=protocol_version,
            )
            self._log_access(
                request_id=request_id, status_code=HTTPStatus.OK, started_at=started_at
            )
            return
        if self.path == "/metrics":
            self._send_json(
                get_runtime_metrics(),
                request_id=request_id,
                protocol_version=protocol_version,
            )
            self._log_access(
                request_id=request_id, status_code=HTTPStatus.OK, started_at=started_at
            )
            return
        self.send_error(HTTPStatus.NOT_FOUND)
        self._log_access(request_id=request_id, status_code=HTTPStatus.NOT_FOUND, started_at=started_at)

    def do_POST(self) -> None:  # noqa: N802
        started_at = _perf_counter()
        request_id = _resolve_http_request_id(self.headers.get("X-Request-Id"))
        protocol_version = _resolve_mcp_protocol_version(self.headers.get("MCP-Protocol-Version"))
        if self.path != "/mcp":
            self.send_error(HTTPStatus.NOT_FOUND)
            self._log_access(
                request_id=request_id, status_code=HTTPStatus.NOT_FOUND, started_at=started_at
            )
            return
        accept_header = self.headers.get("Accept")
        sse_only = _client_accepts_sse(accept_header) and not _client_accepts_json(accept_header)

        if not _is_json_content_type(self.headers.get("Content-Type")):
            error_payload = _rpc_error(None, -32600, "Invalid Request: Content-Type must be application/json")
            if sse_only:
                self._send_sse_data(
                    error_payload,
                    status=HTTPStatus.UNSUPPORTED_MEDIA_TYPE,
                    request_id=request_id,
                    protocol_version=protocol_version,
                )
            else:
                self._send_json(
                    error_payload,
                    status=HTTPStatus.UNSUPPORTED_MEDIA_TYPE,
                    request_id=request_id,
                    protocol_version=protocol_version,
                )
            self._log_access(
                request_id=request_id,
                status_code=HTTPStatus.UNSUPPORTED_MEDIA_TYPE,
                started_at=started_at,
            )
            return

        try:
            content_length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            error_payload = _rpc_error(None, -32600, "Invalid Request: invalid Content-Length header")
            if sse_only:
                self._send_sse_data(
                    error_payload,
                    status=HTTPStatus.BAD_REQUEST,
                    request_id=request_id,
                    protocol_version=protocol_version,
                )
            else:
                self._send_json(
                    error_payload,
                    status=HTTPStatus.BAD_REQUEST,
                    request_id=request_id,
                    protocol_version=protocol_version,
                )
            self._log_access(request_id=request_id, status_code=HTTPStatus.BAD_REQUEST, started_at=started_at)
            return

        if content_length < 0:
            error_payload = _rpc_error(None, -32600, "Invalid Request: Content-Length must be non-negative")
            if sse_only:
                self._send_sse_data(
                    error_payload,
                    status=HTTPStatus.BAD_REQUEST,
                    request_id=request_id,
                    protocol_version=protocol_version,
                )
            else:
                self._send_json(
                    error_payload,
                    status=HTTPStatus.BAD_REQUEST,
                    request_id=request_id,
                    protocol_version=protocol_version,
                )
            self._log_access(request_id=request_id, status_code=HTTPStatus.BAD_REQUEST, started_at=started_at)
            return
        if content_length > MAX_REQUEST_BODY_BYTES:
            error_payload = _rpc_error(None, -32600, "Invalid Request: body is too large")
            if sse_only:
                self._send_sse_data(
                    error_payload,
                    status=HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                    request_id=request_id,
                    protocol_version=protocol_version,
                )
            else:
                self._send_json(
                    error_payload,
                    status=HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                    request_id=request_id,
                    protocol_version=protocol_version,
                )
            self._log_access(
                request_id=request_id,
                status_code=HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                started_at=started_at,
            )
            return

        raw_body = self.rfile.read(content_length)
        try:
            request_payload = json.loads(raw_body.decode("utf-8"))
        except json.JSONDecodeError:
            error_payload = _rpc_error(None, -32700, "Parse error")
            if sse_only:
                self._send_sse_data(
                    error_payload,
                    status=HTTPStatus.BAD_REQUEST,
                    request_id=request_id,
                    protocol_version=protocol_version,
                )
            else:
                self._send_json(
                    error_payload,
                    status=HTTPStatus.BAD_REQUEST,
                    request_id=request_id,
                    protocol_version=protocol_version,
                )
            self._log_access(request_id=request_id, status_code=HTTPStatus.BAD_REQUEST, started_at=started_at)
            return

        response_payload = handle_jsonrpc_payload(request_payload, http_request_id=request_id)
        if response_payload is None:
            self.send_response(HTTPStatus.NO_CONTENT)
            self.send_header("X-Request-Id", request_id)
            self.send_header("MCP-Protocol-Version", protocol_version)
            self.end_headers()
            self._log_access(
                request_id=request_id, status_code=HTTPStatus.NO_CONTENT, started_at=started_at
            )
            return
        if sse_only:
            self._send_sse_data(
                response_payload,
                request_id=request_id,
                protocol_version=protocol_version,
            )
            self._log_access(request_id=request_id, status_code=HTTPStatus.OK, started_at=started_at)
            return
        self._send_json(response_payload, request_id=request_id, protocol_version=protocol_version)
        self._log_access(request_id=request_id, status_code=HTTPStatus.OK, started_at=started_at)

    def log_message(self, fmt: str, *args: Any) -> None:
        # Keep server output concise in local development.
        return

    def _send_json(
        self,
        payload: dict[str, Any] | list[dict[str, Any]],
        *,
        status: HTTPStatus = HTTPStatus.OK,
        request_id: str | None = None,
        protocol_version: str = MCP_PROTOCOL_VERSION,
    ) -> None:
        encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        should_compress = self._should_use_gzip(len(encoded))
        if should_compress:
            encoded = gzip.compress(encoded, compresslevel=5)
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        if should_compress:
            self.send_header("Content-Encoding", "gzip")
            self.send_header("Vary", "Accept-Encoding")
        if request_id:
            self.send_header("X-Request-Id", request_id)
        self.send_header("MCP-Protocol-Version", protocol_version)
        self.end_headers()
        self.wfile.write(encoded)

    def _should_use_gzip(self, payload_size: int) -> bool:
        if payload_size < MIN_GZIP_BYTES:
            return False
        accept_encoding = str(self.headers.get("Accept-Encoding") or "").lower()
        return "gzip" in accept_encoding

    def _send_sse_comment(
        self,
        comment: str,
        *,
        request_id: str | None = None,
        protocol_version: str = MCP_PROTOCOL_VERSION,
    ) -> None:
        payload = f": {comment}\n\n".encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("Content-Length", str(len(payload)))
        if request_id:
            self.send_header("X-Request-Id", request_id)
        self.send_header("MCP-Protocol-Version", protocol_version)
        self.end_headers()
        self.wfile.write(payload)

    def _send_sse_data(
        self,
        payload: dict[str, Any] | list[dict[str, Any]],
        *,
        status: HTTPStatus = HTTPStatus.OK,
        request_id: str | None = None,
        protocol_version: str = MCP_PROTOCOL_VERSION,
    ) -> None:
        data_line = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        encoded = f"data: {data_line}\n\n".encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("Content-Length", str(len(encoded)))
        if request_id:
            self.send_header("X-Request-Id", request_id)
        self.send_header("MCP-Protocol-Version", protocol_version)
        self.end_headers()
        self.wfile.write(encoded)

    def _log_access(self, *, request_id: str, status_code: HTTPStatus, started_at: float) -> None:
        latency_ms = (_perf_counter() - started_at) * 1000.0
        client_host = str(self.client_address[0]) if self.client_address else ""
        client_port = int(self.client_address[1]) if self.client_address else 0
        user_agent = str(self.headers.get("User-Agent") or "")
        message = _build_access_log_message(
            method=self.command,
            path=self.path,
            status_code=int(status_code),
            latency_ms=latency_ms,
            client_ip=client_host,
            user_agent=user_agent,
            request_id=request_id,
        )
        logger.info(
            message,
            extra={
                "request_id": request_id,
                "method": self.command,
                "path": self.path,
                "status_code": int(status_code),
                "latency_ms": round(latency_ms, 3),
                "client_ip": client_host,
                "client_port": client_port,
                "user_agent": user_agent,
            },
        )


def _configure_runtime_logging() -> None:
    raw_level = str(os.getenv("MCP_LOG_LEVEL", "INFO")).strip().upper()
    log_level = getattr(logging, raw_level, logging.INFO)
    root_logger = logging.getLogger()
    if not root_logger.handlers:
        logging.basicConfig(
            level=log_level,
            format="%(asctime)s %(levelname)s %(name)s %(message)s",
        )
    else:
        root_logger.setLevel(log_level)
    logger.setLevel(log_level)


def run_server(host: str = "127.0.0.1", port: int = 8000) -> None:
    """Run MCP HTTP server."""

    _configure_runtime_logging()
    server = ThreadingHTTPServer((host, port), MCPHttpHandler)
    print(f"MCP server started on http://{host}:{port}/mcp")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("MCP server stopped gracefully.")
    finally:
        server.server_close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run minimal MCP HTTP server.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    run_server(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
