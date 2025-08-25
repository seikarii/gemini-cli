#!/usr/bin/env python3
"""
ExternalAPITool - Herramienta Genérica de Acceso a APIs Externas
=================================================================
Permite interactuar con APIs externas con autenticación, manejo de errores, y soporte avanzado para múltiples formatos.

Características:
- Métodos HTTP completos (GET, POST, PUT, DELETE, PATCH, OPTIONS, HEAD)
- Autenticación flexible (API key, Bearer, Basic, OAuth)
- Manejo de formatos (JSON, XML, form, texto)
- Lógica de reintentos y limitación de tasa por dominio
- Parsing robusto de respuestas y manejo de errores
- Soporte para batch requests y análisis de endpoints
"""

import asyncio
import json
import logging
import time
from typing import Any, Literal, cast
from urllib.parse import urlencode, urlparse

import httpx
from pydantic import BaseModel, Field

from crisalida_lib.ASTRAL_TOOLS.base import BaseTool, ToolCallResult

logger = logging.getLogger(__name__)


class APIAuthParameters(BaseModel):
    type: Literal["none", "api_key", "bearer", "basic", "oauth"] = Field(
        "none", description="Tipo de autenticación"
    )
    api_key: str | None = Field(
        None, description="API key para autenticación tipo api_key"
    )
    api_key_header: str = Field(
        "X-API-Key", description="Nombre del header para API key"
    )
    token: str | None = Field(None, description="Token Bearer u OAuth")
    username: str | None = Field(None, description="Usuario para autenticación básica")
    password: str | None = Field(
        None, description="Contraseña para autenticación básica"
    )
    oauth_params: dict[str, str] | None = Field(
        None, description="Parámetros adicionales para OAuth"
    )


class ExternalAPIParameters(BaseModel):
    action: Literal[
        "make_request", "test_connection", "get_api_info", "batch_requests"
    ] = Field(..., description="Acción a realizar")
    url: str = Field(..., description="URL del endpoint API")
    method: Literal["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"] = Field(
        "GET", description="Método HTTP"
    )
    auth: APIAuthParameters | None = Field(
        None, description="Parámetros de autenticación"
    )
    data: dict[str, Any] | str | None = Field(
        None, description="Datos del cuerpo de la petición"
    )
    params: dict[str, str] | None = Field(None, description="Parámetros de la URL")
    headers: dict[str, str] | None = Field(None, description="Headers adicionales")
    content_type: Literal["json", "form", "xml", "text"] = Field(
        "json", description="Tipo de contenido de la petición"
    )
    response_format: Literal["json", "xml", "text", "auto"] = Field(
        "auto", description="Formato esperado de la respuesta"
    )
    timeout: int = Field(30, description="Timeout de la petición en segundos")
    max_retries: int = Field(3, description="Máximo número de reintentos")
    retry_delay: float = Field(1.0, description="Delay entre reintentos en segundos")
    requests: list[dict[str, Any]] | None = Field(
        None, description="Lista de peticiones para batch"
    )


class ExternalAPITool(BaseTool):
    """
    Herramienta avanzada para realizar peticiones a APIs externas con autenticación y manejo robusto de errores.
    """

    def __init__(self):
        super().__init__()
        self._client = (
            httpx.AsyncClient()
        )  # This client is managed by __aenter__ and __aexit__
        self._rate_limiter = {}

    def _get_name(self) -> str:
        return "external_api"

    def _get_description(self) -> str:
        return "Realiza peticiones a APIs externas con autenticación, manejo de errores y análisis de endpoints."

    def _get_category(self) -> str:
        return "web_api"

    def _get_pydantic_schema(self) -> type[BaseModel]:
        return ExternalAPIParameters

    def _setup_auth(
        self, auth: APIAuthParameters | None, headers: dict[str, str]
    ) -> dict[str, str]:
        if not auth or auth.type == "none":
            return headers
        if auth.type == "api_key" and auth.api_key:
            headers[auth.api_key_header] = auth.api_key
        elif auth.type == "bearer" and auth.token:
            headers["Authorization"] = f"Bearer {auth.token}"
        elif auth.type == "basic" and auth.username and auth.password:
            import base64

            credentials = base64.b64encode(
                f"{auth.username}:{auth.password}".encode()
            ).decode()
            headers["Authorization"] = f"Basic {credentials}"
        elif auth.type == "oauth" and auth.token:
            headers["Authorization"] = f"Bearer {auth.token}"
            if auth.oauth_params:
                headers.update(auth.oauth_params)
        return headers

    def _prepare_request_data(self, data: Any, content_type: str) -> tuple[Any, str]:
        if data is None:
            return None, ""
        if content_type == "json":
            if isinstance(data, str):
                try:
                    json.loads(data)
                    return data, "application/json"
                except json.JSONDecodeError as e:
                    raise ValueError("Invalid JSON string provided") from e
            else:
                return json.dumps(data), "application/json"
        elif content_type == "form":
            if isinstance(data, dict):
                return urlencode(data), "application/x-www-form-urlencoded"
            else:
                return str(data), "application/x-www-form-urlencoded"
        elif content_type == "xml":
            return str(data), "application/xml"
        elif content_type == "text":
            return str(data), "text/plain"
        else:
            return data, ""

    def _parse_response(
        self, response: httpx.Response, expected_format: str
    ) -> tuple[Any, str]:
        content_type = response.headers.get("content-type", "").lower()
        if expected_format == "auto":
            if "application/json" in content_type:
                expected_format = "json"
            elif "xml" in content_type:
                expected_format = "xml"
            else:
                expected_format = "text"
        try:
            if expected_format == "json":
                return response.json(), "json"
            elif expected_format == "xml":
                return response.text, "xml"
            else:
                return response.text, "text"
        except Exception as e:
            logger.warning(f"Error parsing response: {e}")
            return response.text, "text_fallback"

    def _check_rate_limit(self, domain: str) -> bool:
        current_time = time.time()
        if domain not in self._rate_limiter:
            self._rate_limiter[domain] = []
        self._rate_limiter[domain] = [
            req_time
            for req_time in self._rate_limiter[domain]
            if current_time - req_time < 60
        ]
        if len(self._rate_limiter[domain]) >= 60:
            return False
        self._rate_limiter[domain].append(current_time)
        return True

    async def _make_single_request(
        self, params: ExternalAPIParameters
    ) -> dict[str, Any]:
        domain = urlparse(params.url).netloc
        if not self._check_rate_limit(domain):
            raise Exception(f"Rate limit exceeded for {domain}")
        headers = params.headers.copy() if params.headers else {}
        headers = self._setup_auth(params.auth, headers)
        request_data = None
        if params.data is not None and params.method in ["POST", "PUT", "PATCH"]:
            request_data, content_type_header = self._prepare_request_data(
                params.data, params.content_type
            )
            if content_type_header:
                headers["Content-Type"] = content_type_header
        last_exception = None
        for attempt in range(params.max_retries + 1):
            try:
                response = await self._client.request(
                    method=params.method,
                    url=params.url,
                    headers=headers,
                    params=params.params,
                    content=request_data,
                    timeout=params.timeout,
                )
                response_data, response_format = self._parse_response(
                    response, params.response_format
                )
                return {
                    "success": True,
                    "status_code": response.status_code,
                    "headers": dict(response.headers),
                    "data": response_data,
                    "format": response_format,
                    "url": str(response.url),
                    "method": params.method,
                    "attempt": attempt + 1,
                }
            except httpx.TimeoutException as e:
                last_exception = f"Request timeout after {params.timeout}s"
                logger.warning(f"Request timeout (attempt {attempt + 1}): {e}")
            except httpx.RequestError as e:
                last_exception = f"Request error: {str(e)}"
                logger.warning(f"Request error (attempt {attempt + 1}): {e}")
            except Exception as e:
                last_exception = f"Unexpected error: {str(e)}"
                logger.warning(f"Unexpected error (attempt {attempt + 1}): {e}")
            if attempt < params.max_retries:
                await asyncio.sleep(params.retry_delay * (2**attempt))
        return {
            "success": False,
            "error": last_exception or "Request failed after all retries",
            "url": params.url,
            "method": params.method,
            "attempts": params.max_retries + 1,
        }

    async def _test_connection(self, params: ExternalAPIParameters) -> str:
        try:
            for method in ["HEAD", "GET"]:
                test_params = ExternalAPIParameters(
                    action="make_request",
                    url=params.url,
                    method=cast(Literal["GET", "HEAD"], method),
                    auth=params.auth,
                    headers=params.headers,
                    timeout=10,
                    max_retries=1,
                    data=None,
                    params=None,
                    content_type="json",
                    response_format="auto",
                    retry_delay=1.0,
                    requests=None,
                )
                result = await self._make_single_request(test_params)
                if result["success"]:
                    return f"✅ Connection successful!\nMethod: {method}\nStatus: {result['status_code']}\nResponse headers: {list(result['headers'].keys())}"
                elif method == "GET":
                    return f"❌ Connection failed!\nError: {result.get('error', 'Unknown error')}"
            return "❌ Connection test failed"
        except Exception as e:
            return f"❌ Connection test error: {str(e)}"

    async def _get_api_info(self, params: ExternalAPIParameters) -> str:
        info = []
        info.append(f"API Endpoint Analysis: {params.url}")
        info.append("=" * 50)
        connection_result = await self._test_connection(params)
        info.append(f"Connection Test:\n{connection_result}")
        try:
            options_params = ExternalAPIParameters(
                action="make_request",
                url=params.url,
                method=cast(Literal["OPTIONS"], "OPTIONS"),
                auth=params.auth,
                headers=params.headers,
                timeout=10,
                max_retries=1,
                data=None,
                params=None,
                content_type="json",
                response_format="auto",
                retry_delay=1.0,
                requests=None,
            )
            result = await self._make_single_request(options_params)
            if result["success"]:
                info.append("\nSupported Methods:")
                allow_header = result["headers"].get(
                    "allow", result["headers"].get("Allow", "")
                )
                info.append(
                    f"  {allow_header}"
                    if allow_header
                    else "  Not specified in Allow header"
                )
                interesting_headers = [
                    "server",
                    "x-ratelimit-limit",
                    "x-api-version",
                    "access-control-allow-origin",
                ]
                found_headers = {
                    k: v
                    for k, v in result["headers"].items()
                    if k.lower() in interesting_headers
                }
                if found_headers:
                    info.append("\nAPI Info Headers:")
                    for k, v in found_headers.items():
                        info.append(f"  {k}: {v}")
        except Exception as e:
            info.append(f"\nOPTIONS request failed: {str(e)}")
        return "\n".join(info)

    async def _batch_requests(self, requests: list[dict[str, Any]]) -> dict[str, Any]:
        results = []
        for i, request_data in enumerate(requests):
            try:
                request_params = ExternalAPIParameters(**request_data)
                result = await self._make_single_request(request_params)
                result["request_index"] = i
                results.append(result)
                await asyncio.sleep(0.1)
            except Exception as e:
                results.append(
                    {
                        "success": False,
                        "error": str(e),
                        "request_index": i,
                        "request_data": request_data,
                    }
                )
        successful = sum(1 for r in results if r.get("success", False))
        failed = len(results) - successful
        return {
            "total_requests": len(results),
            "successful": successful,
            "failed": failed,
            "results": results,
        }

    async def execute(self, **kwargs) -> ToolCallResult:
        start_time = asyncio.get_event_loop().time()
        try:
            params = ExternalAPIParameters(**kwargs)
            if params.action == "make_request":
                result = await self._make_single_request(params)
                if result["success"]:
                    output = f"✅ Request successful!\nStatus: {result['status_code']}\nMethod: {result['method']}\nURL: {result['url']}\nFormat: {result['format']}\n"
                    data_str = str(result["data"])
                    output += (
                        f"Response: {data_str[:1000]}...\n"
                        if len(data_str) > 1000
                        else f"Response: {data_str}\n"
                    )
                    if result["attempt"] > 1:
                        output += f"(Succeeded on attempt {result['attempt']})"
                else:
                    output = f"❌ Request failed!\nError: {result['error']}\nURL: {result['url']}\nAttempts: {result['attempts']}"
            elif params.action == "test_connection":
                output = await self._test_connection(params)
            elif params.action == "get_api_info":
                output = await self._get_api_info(params)
            elif params.action == "batch_requests":
                if not params.requests:
                    return ToolCallResult(
                        command="external_api",
                        success=False,
                        output="",
                        error_message="requests parameter is required for batch_requests action",
                        execution_time=asyncio.get_event_loop().time() - start_time,
                    )
                batch_result = await self._batch_requests(params.requests)
                output = f"📊 Batch Request Results:\nTotal: {batch_result['total_requests']}\nSuccessful: {batch_result['successful']}\nFailed: {batch_result['failed']}\n\n"
                for result in batch_result["results"]:
                    idx = result.get("request_index", "?")
                    if result.get("success", False):
                        status = result.get("status_code", "?")
                        output += f"  [{idx}] ✅ Status {status}\n"
                    else:
                        error = result.get("error", "Unknown error")
                        output += f"  [{idx}] ❌ {error}\n"
            else:
                return ToolCallResult(
                    command="external_api",
                    success=False,
                    output="",
                    error_message=f"Unknown action: {params.action}",
                )
            execution_time = asyncio.get_event_loop().time() - start_time
            return ToolCallResult(
                command=f"external_api({params.action})",
                success=True,
                output=output,
                error_message=None,
                execution_time=execution_time,
                metadata={
                    "action": params.action,
                    "url": params.url,
                    "method": params.method,
                    "auth_type": params.auth.type if params.auth else "none",
                },
            )
        except Exception as e:
            execution_time = asyncio.get_event_loop().time() - start_time
            logger.error(f"External API error: {e}")
            return ToolCallResult(
                command="external_api",
                success=False,
                output="",
                error_message=str(e),
                execution_time=execution_time,
            )

    async def demo(self):
        """Demonstrate the external API tool's functionality."""
        print("🌐 EXTERNAL API TOOL DEMO")
        print("=" * 40)

        # Test a simple HTTP GET request
        result = await self.execute(
            action="make_request",
            url="https://httpbin.org/status/200",
            method="GET",
            data=None,
            params=None,
            content_type="json",
            response_format="auto",
            retry_delay=1.0,
            requests=None,
        )
        print(f"HTTP GET request: {result.success}")
        print(
            result.output[:200] + "..." if len(result.output) > 200 else result.output
        )

        # Test a POST request (example)
        result = await self.execute(
            action="make_request",
            url="https://httpbin.org/post",
            method="POST",
            data={"key": "value"},
            content_type="json",
            params=None,
            response_format="auto",
            retry_delay=1.0,
            requests=None,
        )
        print(f"\nHTTP POST request: {result.success}")
        print(result.output)

        print("\n✅ External API demo completed!")

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self._client.aclose()


if __name__ == "__main__":
    from crisalida_lib.ASTRAL_TOOLS.demos.external_api_demos import demo_external_api

    asyncio.run(demo_external_api())
