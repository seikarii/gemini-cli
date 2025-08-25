import asyncio
import logging
import os
from datetime import datetime
from typing import Any, Literal

from playwright.async_api import Browser, BrowserContext, Page, async_playwright
from pydantic import BaseModel, Field, field_validator

from crisalida_lib.ASTRAL_TOOLS.base import BaseTool, ToolCallResult

logger = logging.getLogger(__name__)


class VisualUISchema(BaseModel):
    """Parámetros para la herramienta de interacción visual/UI"""

    action: Literal[
        "start_browser",
        "navigate",
        "click",
        "type",
        "screenshot",
        "extract_text",
        "extract_element",
        "fill_form",
        "wait_for_element",
        "scroll",
        "get_page_info",
        "close_browser",
        "list_browsers",
        "get_element_attributes",
        "execute_script",
    ] = Field(
        ...,
        description="Acción: 'start_browser', 'navigate', 'click', 'type', 'screenshot', 'extract_text', 'extract_element', 'fill_form', 'wait_for_element', 'scroll', 'get_page_info', 'close_browser', 'list_browsers', 'get_element_attributes', 'execute_script'",
    )
    browser_id: str | None = Field(None, description="ID de la sesión de navegador")
    url: str | None = Field(None, description="URL a navegar")
    selector: str | None = Field(None, description="Selector CSS/XPath")
    text: str | None = Field(None, description="Texto a escribir o script JS")
    xpath: str | None = Field(None, description="Selector XPath")
    timeout: int = Field(default=10000, description="Timeout en ms")
    wait_for: Literal["load", "domcontentloaded", "networkidle"] | None = Field(
        None,
        description="Condición de espera: 'load', 'domcontentloaded', 'networkidle'",
    )
    screenshot_path: str | None = Field(
        None, description="Ruta para guardar screenshot"
    )
    form_data: dict[str, str] | None = Field(None, description="Datos de formulario")
    scroll_x: int = Field(default=0, description="Scroll horizontal")
    scroll_y: int = Field(default=0, description="Scroll vertical")
    browser_type: Literal["chromium", "firefox", "webkit"] = Field(
        default="chromium",
        description="Tipo de navegador: 'chromium', 'firefox', 'webkit'",
    )
    headless: bool = Field(default=True, description="Modo headless")

    @field_validator("action")
    @classmethod
    def validate_action(cls, v):
        valid_actions = [
            "start_browser",
            "navigate",
            "click",
            "type",
            "screenshot",
            "extract_text",
            "extract_element",
            "fill_form",
            "wait_for_element",
            "scroll",
            "get_page_info",
            "close_browser",
            "list_browsers",
            "get_element_attributes",
            "execute_script",
        ]
        if v not in valid_actions:
            raise ValueError(f"Action must be one of: {valid_actions}")
        return v

    @field_validator("browser_type")
    @classmethod
    def validate_browser_type(cls, v):
        valid_types = ["chromium", "firefox", "webkit"]
        if v not in valid_types:
            raise ValueError(f"Browser type must be one of: {valid_types}")
        return v


class BrowserSession:
    """Gestor de sesión de navegador"""

    def __init__(
        self,
        browser_id: str,
        browser: Browser,
        context: BrowserContext,
        page: Page,
    ):
        self.browser_id = browser_id
        self.browser = browser
        self.context = context
        self.page = page
        self.created_at = datetime.now()
        self.last_action: str | None = None  # Added type hint
        self.action_count = 0

    async def close(self):
        try:
            if self.context:
                await self.context.close()
            if self.browser:
                await self.browser.close()
        except Exception as e:
            logger.error(f"Error closing browser session {self.browser_id}: {e}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "browser_id": self.browser_id,
            "created_at": self.created_at.isoformat(),
            "last_action": self.last_action,
            "action_count": self.action_count,
            "current_url": getattr(self.page, "url", None),
        }


class VisualUITool(BaseTool):
    """Herramienta avanzada para interacción visual/UI y automatización web"""

    def __init__(self):
        super().__init__()
        self.browser_sessions: dict[str, BrowserSession] = {}
        self.playwright = None
        self.active_browser: str | None = None  # Added type hint

    def _get_name(self) -> str:
        return "visual_ui_interaction"

    def _get_description(self) -> str:
        return "Interactúa con páginas web y elementos UI: clic, tipeo, screenshot, extracción de texto y atributos"

    def _get_category(self) -> str:
        return "ui_automation"

    def _get_pydantic_schema(self) -> type[BaseModel]:
        return VisualUISchema

    async def execute(self, **kwargs) -> ToolCallResult:
        start_time = datetime.now()
        try:
            action = kwargs.get("action")
            if action == "start_browser":
                return await self._start_browser(**kwargs)
            elif action == "navigate":
                return await self._navigate(**kwargs)
            elif action == "click":
                return await self._click(**kwargs)
            elif action == "type":
                return await self._type(**kwargs)
            elif action == "screenshot":
                return await self._screenshot(**kwargs)
            elif action == "extract_text":
                return await self._extract_text(**kwargs)
            elif action == "extract_element":
                return await self._extract_element(**kwargs)
            elif action == "fill_form":
                return await self._fill_form(**kwargs)
            elif action == "wait_for_element":
                return await self._wait_for_element(**kwargs)
            elif action == "scroll":
                return await self._scroll(**kwargs)
            elif action == "get_page_info":
                return await self._get_page_info(**kwargs)
            elif action == "get_element_attributes":
                return await self._get_element_attributes(**kwargs)
            elif action == "execute_script":
                return await self._execute_script(**kwargs)
            elif action == "close_browser":
                return await self._close_browser(**kwargs)
            elif action == "list_browsers":
                return await self._list_browsers(**kwargs)
            else:
                raise ValueError(f"Unknown action: {action}")
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"Visual UI tool error: {e}")
            return ToolCallResult(
                command=f"visual_ui_interaction({action})",
                success=False,
                output="",
                error_message=str(e),
                execution_time=execution_time,
            )

    async def _ensure_playwright(self):
        if self.playwright is None:
            self.playwright = await async_playwright().start()

    async def _start_browser(self, **kwargs) -> ToolCallResult:
        start_time = datetime.now()  # Added this line
        browser_id = kwargs.get("browser_id", f"browser_{len(self.browser_sessions)}")
        browser_type = kwargs.get("browser_type", "chromium")
        headless = kwargs.get("headless", True)
        if browser_id in self.browser_sessions:
            raise ValueError(f"Browser with ID '{browser_id}' already exists")
        await self._ensure_playwright()
        browser = getattr(self.playwright, browser_type).launch
        browser = await browser(headless=headless)
        context = await browser.new_context()
        page = await context.new_page()
        session = BrowserSession(browser_id, browser, context, page)
        self.browser_sessions[browser_id] = session
        self.active_browser = browser_id
        output = f"Started {browser_type} browser session: {browser_id}"
        return ToolCallResult(
            command=f"start_browser({browser_id})",
            success=True,
            output=output,
            execution_time=(
                datetime.now() - start_time
            ).total_seconds(),  # Changed from 0.5
            metadata={
                "browser_id": browser_id,
                "browser_type": browser_type,
                "headless": headless,
            },
            error_message=None,  # Added this line
        )

    async def _navigate(self, **kwargs) -> ToolCallResult:
        start_time = datetime.now()  # Added this line
        browser_id = kwargs.get("browser_id", self.active_browser)
        url = kwargs.get("url")
        wait_for = kwargs.get("wait_for", "load")
        timeout = kwargs.get("timeout", 10000)
        if not browser_id or browser_id not in self.browser_sessions:
            raise ValueError("Valid browser ID required")
        if not url:
            raise ValueError("URL is required")
        session = self.browser_sessions[browser_id]
        page = session.page
        response = await page.goto(url, wait_until=wait_for, timeout=timeout)
        session.last_action = f"navigate:{url}"
        session.action_count += 1
        title = await page.title()
        current_url = page.url
        output = f"Navigated to: {current_url}"
        if title:
            output += f" (Title: {title})"
        return ToolCallResult(
            command=f"navigate({browser_id}, {url})",
            success=True,
            output=output,
            execution_time=(
                datetime.now() - start_time
            ).total_seconds(),  # Changed from 0.5
            metadata={
                "browser_id": browser_id,
                "url": current_url,
                "title": title,
                "status": response.status if response else None,
            },
            error_message=None,  # Added this line
        )

    async def _click(self, **kwargs) -> ToolCallResult:
        start_time = datetime.now()  # Added this line
        browser_id = kwargs.get("browser_id", self.active_browser)
        selector = kwargs.get("selector")
        xpath = kwargs.get("xpath")
        timeout = kwargs.get("timeout", 10000)
        if not browser_id or browser_id not in self.browser_sessions:
            raise ValueError("Valid browser ID required")
        if not selector and not xpath:
            raise ValueError("Either CSS selector or XPath is required")
        session = self.browser_sessions[browser_id]
        page = session.page
        element = page.locator(f"xpath={xpath}") if xpath else page.locator(selector)
        await element.click(timeout=timeout)
        session.last_action = f"click:{selector or xpath}"
        session.action_count += 1
        output = f"Clicked element: {selector or xpath}"
        return ToolCallResult(
            command=f"click({browser_id})",
            success=True,
            output=output,
            execution_time=(
                datetime.now() - start_time
            ).total_seconds(),  # Changed from 0.2
            metadata={"browser_id": browser_id, "selector": selector, "xpath": xpath},
            error_message=None,  # Added this line
        )

    async def _type(self, **kwargs) -> ToolCallResult:
        start_time = datetime.now()  # Added this line
        browser_id = kwargs.get("browser_id", self.active_browser)
        selector = kwargs.get("selector")
        xpath = kwargs.get("xpath")
        text = kwargs.get("text")
        timeout = kwargs.get("timeout", 10000)
        if not browser_id or browser_id not in self.browser_sessions:
            raise ValueError("Valid browser ID required")
        if not selector and not xpath:
            raise ValueError("Either CSS selector or XPath is required")
        if not text:
            raise ValueError("Text is required")
        session = self.browser_sessions[browser_id]
        page = session.page
        element = page.locator(f"xpath={xpath}") if xpath else page.locator(selector)
        await element.fill(text, timeout=timeout)
        session.last_action = f"type:{selector or xpath}"
        session.action_count += 1
        output = f"Typed text into element: {selector or xpath}"
        return ToolCallResult(
            command=f"type({browser_id})",
            success=True,
            output=output,
            execution_time=(
                datetime.now() - start_time
            ).total_seconds(),  # Changed from 0.2
            metadata={
                "browser_id": browser_id,
                "selector": selector,
                "xpath": xpath,
                "text_length": len(text),
            },
            error_message=None,  # Added this line
        )

    async def _screenshot(self, **kwargs) -> ToolCallResult:
        start_time = datetime.now()  # Added this line
        browser_id = kwargs.get("browser_id", self.active_browser)
        screenshot_path = kwargs.get("screenshot_path")
        if not browser_id or browser_id not in self.browser_sessions:
            raise ValueError("Valid browser ID required")
        session = self.browser_sessions[browser_id]
        page = session.page
        if not screenshot_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            screenshot_path = f"/tmp/screenshot_{browser_id}_{timestamp}.png"
        await page.screenshot(path=screenshot_path, full_page=True)
        session.last_action = "screenshot"
        session.action_count += 1
        file_size = (
            os.path.getsize(screenshot_path) if os.path.exists(screenshot_path) else 0
        )
        output = f"Screenshot saved: {screenshot_path} ({file_size} bytes)"
        return ToolCallResult(
            command=f"screenshot({browser_id})",
            success=True,
            output=output,
            execution_time=(
                datetime.now() - start_time
            ).total_seconds(),  # Changed from 0.5
            metadata={
                "browser_id": browser_id,
                "screenshot_path": screenshot_path,
                "file_size": file_size,
            },
            error_message=None,  # Added this line
        )

    async def _extract_text(self, **kwargs) -> ToolCallResult:
        start_time = datetime.now()  # Added this line
        browser_id = kwargs.get("browser_id", self.active_browser)
        selector = kwargs.get("selector")
        xpath = kwargs.get("xpath")
        timeout = kwargs.get("timeout", 10000)
        if not browser_id or browser_id not in self.browser_sessions:
            raise ValueError("Valid browser ID required")
        session = self.browser_sessions[browser_id]
        page = session.page
        if selector or xpath:
            element = (
                page.locator(f"xpath={xpath}") if xpath else page.locator(selector)
            )
            text = await element.text_content(timeout=timeout)
            extracted_from = selector or xpath
        else:
            text = await page.text_content("body")
            extracted_from = "entire page"
        session.last_action = f"extract_text:{extracted_from}"
        session.action_count += 1
        output = f"Extracted text from {extracted_from}: {len(text)} characters"
        output += f"\nText: {text[:500]}{'...' if len(text) > 500 else ''}"
        return ToolCallResult(
            command=f"extract_text({browser_id})",
            success=True,
            output=output,
            execution_time=(
                datetime.now() - start_time
            ).total_seconds(),  # Changed from 0.2
            metadata={
                "browser_id": browser_id,
                "selector": selector,
                "xpath": xpath,
                "text": text,
                "text_length": len(text),
            },
            error_message=None,  # Added this line
        )

    async def _extract_element(self, **kwargs) -> ToolCallResult:
        start_time = datetime.now()  # Added this line
        browser_id = kwargs.get("browser_id", self.active_browser)
        selector = kwargs.get("selector")
        xpath = kwargs.get("xpath")
        timeout = kwargs.get("timeout", 10000)
        if not browser_id or browser_id not in self.browser_sessions:
            raise ValueError("Valid browser ID required")
        if not selector and not xpath:
            raise ValueError("Either CSS selector or XPath is required")
        session = self.browser_sessions[browser_id]
        page = session.page
        element = (
            page.locator(f"xpath={xpath}") if xpath else page.locator(selector)
        )  # Added this line
        element_info = {}
        try:
            element_info["tag_name"] = await element.evaluate("el => el.tagName")
            element_info["text_content"] = await element.text_content(timeout=timeout)
            element_info["inner_html"] = await element.inner_html(timeout=timeout)
            element_info["outer_html"] = await element.evaluate("el => el.outerHTML")
            for attr in ["id", "class", "href", "src", "value", "type", "name"]:
                value = await element.get_attribute(attr)
                if value:
                    element_info[f"attr_{attr}"] = value
            bbox = await element.bounding_box()
            if bbox:
                element_info["bounding_box"] = bbox
        except Exception as e:
            logger.warning(f"Could not extract some element information: {e}")
        session.last_action = f"extract_element:{selector or xpath}"
        session.action_count += 1
        output = f"Extracted element info: {selector or xpath}"
        output += f"\nTag: {element_info.get('tag_name', 'unknown')}"
        output += f"\nText: {element_info.get('text_content', '')[:100]}{'...' if len(element_info.get('text_content', '')) > 100 else ''}"
        return ToolCallResult(
            command=f"extract_element({browser_id})",
            success=True,
            output=output,
            execution_time=(
                datetime.now() - start_time
            ).total_seconds(),  # Changed from 0.3
            metadata={
                "browser_id": browser_id,
                "selector": selector,
                "xpath": xpath,
                "element_info": element_info,
            },
            error_message=None,  # Added this line
        )

    async def _fill_form(self, **kwargs) -> ToolCallResult:
        start_time = datetime.now()  # Added this line
        browser_id = kwargs.get("browser_id", self.active_browser)
        form_data = kwargs.get("form_data")
        timeout = kwargs.get("timeout", 10000)
        if not browser_id or browser_id not in self.browser_sessions:
            raise ValueError("Valid browser ID required")
        if not form_data:
            raise ValueError("Form data is required")
        session = self.browser_sessions[browser_id]
        page = session.page
        filled_fields = []
        errors = []
        for field_selector, value in form_data.items():
            try:
                element = page.locator(field_selector)
                await element.fill(str(value), timeout=timeout)
                filled_fields.append(field_selector)
            except Exception as e:
                errors.append(f"{field_selector}: {str(e)}")
        session.last_action = f"fill_form:{len(filled_fields)}_fields"
        session.action_count += 1
        output = f"Filled {len(filled_fields)} form fields"
        if filled_fields:
            output += f"\nFilled: {', '.join(filled_fields)}"
        if errors:
            output += f"\nErrors: {'; '.join(errors)}"
        return ToolCallResult(
            command=f"fill_form({browser_id})",
            success=len(errors) == 0,
            output=output,
            execution_time=(
                datetime.now() - start_time
            ).total_seconds(),  # Changed from 0.5
            metadata={
                "browser_id": browser_id,
                "filled_fields": filled_fields,
                "errors": errors,
                "total_fields": len(form_data),
            },
            error_message=None,  # Added this line
        )

    async def _wait_for_element(self, **kwargs) -> ToolCallResult:
        start_time = datetime.now()  # Added this line
        browser_id = kwargs.get("browser_id", self.active_browser)
        selector = kwargs.get("selector")
        xpath = kwargs.get("xpath")
        timeout = kwargs.get("timeout", 10000)
        if not browser_id or browser_id not in self.browser_sessions:
            raise ValueError("Valid browser ID required")
        if not selector and not xpath:
            raise ValueError("Either CSS selector or XPath is required")
        session = self.browser_sessions[browser_id]
        page = session.page
        element = page.locator(f"xpath={xpath}") if xpath else page.locator(selector)
        await element.wait_for(timeout=timeout)
        wait_time = (datetime.now() - start_time).total_seconds()
        session.last_action = f"wait_for_element:{selector or xpath}"
        session.action_count += 1
        output = f"Element appeared: {selector or xpath} (waited {wait_time:.2f}s)"
        return ToolCallResult(
            command=f"wait_for_element({browser_id})",
            success=True,
            output=output,
            execution_time=wait_time,
            metadata={
                "browser_id": browser_id,
                "selector": selector,
                "xpath": xpath,
                "wait_time": wait_time,
            },
            error_message=None,  # Added this line
        )

    async def _scroll(self, **kwargs) -> ToolCallResult:
        start_time = datetime.now()  # Added this line
        browser_id = kwargs.get("browser_id", self.active_browser)
        scroll_x = kwargs.get("scroll_x", 0)
        scroll_y = kwargs.get("scroll_y", 0)
        if not browser_id or browser_id not in self.browser_sessions:
            raise ValueError("Valid browser ID required")
        session = self.browser_sessions[browser_id]
        page = session.page
        await page.evaluate(f"window.scrollTo({scroll_x}, {scroll_y})")
        session.last_action = f"scroll:{scroll_x},{scroll_y}"
        session.action_count += 1
        output = f"Scrolled to position: ({scroll_x}, {scroll_y})"
        return ToolCallResult(
            command=f"scroll({browser_id})",
            success=True,
            output=output,
            execution_time=(
                datetime.now() - start_time
            ).total_seconds(),  # Changed from 0.1
            metadata={
                "browser_id": browser_id,
                "scroll_x": scroll_x,
                "scroll_y": scroll_y,
            },
            error_message=None,  # Added this line
        )

    async def _get_page_info(self, **kwargs) -> ToolCallResult:
        start_time = datetime.now()  # Added this line
        browser_id = kwargs.get("browser_id", self.active_browser)
        if not browser_id or browser_id not in self.browser_sessions:
            raise ValueError("Valid browser ID required")
        session = self.browser_sessions[browser_id]
        page = session.page
        page_info = {
            "url": page.url,
            "title": await page.title(),
            "viewport": page.viewport_size,
        }
        try:
            page_info["scroll_position"] = await page.evaluate(
                "() => ({x: window.scrollX, y: window.scrollY})"
            )
            page_info["page_size"] = await page.evaluate(
                "() => ({width: document.body.scrollWidth, height: document.body.scrollHeight})"
            )
            page_info["element_counts"] = await page.evaluate(
                """() => ({
                links: document.querySelectorAll('a').length,
                images: document.querySelectorAll('img').length,
                forms: document.querySelectorAll('form').length,
                inputs: document.querySelectorAll('input').length,
                buttons: document.querySelectorAll('button').length
            })"""
            )
        except Exception as e:
            logger.warning(f"Could not get all page metrics: {e}")
        session.last_action = "get_page_info"
        session.action_count += 1
        output = f"Page info for: {page_info['url']}"
        output += f"\nTitle: {page_info['title']}"
        output += f"\nViewport: {page_info.get('viewport', 'unknown')}"
        if "element_counts" in page_info:
            counts = page_info["element_counts"]
            output += f"\nElements: {counts['links']} links, {counts['images']} images, {counts['forms']} forms"
        return ToolCallResult(
            command=f"get_page_info({browser_id})",
            success=True,
            output=output,
            execution_time=(
                datetime.now() - start_time
            ).total_seconds(),  # Changed from 0.2
            metadata={"browser_id": browser_id, "page_info": page_info},
            error_message=None,  # Added this line
        )

    async def _get_element_attributes(self, **kwargs) -> ToolCallResult:
        start_time = datetime.now()  # Added this line
        browser_id = kwargs.get("browser_id", self.active_browser)
        selector = kwargs.get("selector")
        xpath = kwargs.get("xpath")
        timeout = kwargs.get("timeout", 10000)
        if not browser_id or browser_id not in self.browser_sessions:
            raise ValueError("Valid browser ID required")
        if not selector and not xpath:
            raise ValueError("Either CSS selector or XPath is required")
        session = self.browser_sessions[browser_id]
        page = session.page
        element = (
            page.locator(f"xpath={xpath}") if xpath else page.locator(selector)
        )  # Added this line
        attributes = await element.evaluate(
            """(el) => {
            const attrs = {};
            for (const attr of el.attributes) {
                attrs[attr.name] = attr.value;
            }
            return attrs;
        }""",
            timeout=timeout,
        )
        session.last_action = f"get_attributes:{selector or xpath}"
        session.action_count += 1
        output = f"Element attributes for {selector or xpath}:"
        for name, value in attributes.items():
            output += f"\n  {name}: {value}"
        return ToolCallResult(
            command=f"get_element_attributes({browser_id})",
            success=True,
            output=output,
            execution_time=(
                datetime.now() - start_time
            ).total_seconds(),  # Changed from 0.2
            metadata={
                "browser_id": browser_id,
                "selector": selector,
                "xpath": xpath,
                "attributes": attributes,
            },
            error_message=None,  # Added this line
        )

    async def _execute_script(self, **kwargs) -> ToolCallResult:
        start_time = datetime.now()  # Added this line
        browser_id = kwargs.get("browser_id", self.active_browser)
        script = kwargs.get("text")
        if not browser_id or browser_id not in self.browser_sessions:
            raise ValueError("Valid browser ID required")
        if not script:
            raise ValueError("JavaScript code is required")
        session = self.browser_sessions[browser_id]
        page = session.page
        result = await page.evaluate(script)
        session.last_action = "execute_script"
        session.action_count += 1
        output = (
            f"Executed JavaScript: {script[:100]}{'...' if len(script) > 100 else ''}"
        )
        if result is not None:
            output += f"\nResult: {repr(result)}"
        return ToolCallResult(
            command=f"execute_script({browser_id})",
            success=True,
            output=output,
            execution_time=(
                datetime.now() - start_time
            ).total_seconds(),  # Changed from 0.2
            metadata={"browser_id": browser_id, "script": script, "result": result},
            error_message=None,  # Added this line
        )

    async def _close_browser(self, **kwargs) -> ToolCallResult:
        start_time = datetime.now()  # Added this line
        browser_id = kwargs.get("browser_id", self.active_browser)
        if not browser_id or browser_id not in self.browser_sessions:
            raise ValueError("Valid browser ID required")
        session = self.browser_sessions[browser_id]
        await session.close()
        if self.active_browser == browser_id:
            self.active_browser = None
        del self.browser_sessions[browser_id]
        output = f"Closed browser session: {browser_id}"
        return ToolCallResult(
            command=f"close_browser({browser_id})",
            success=True,
            output=output,
            execution_time=(
                datetime.now() - start_time
            ).total_seconds(),  # Changed from 0.5
            error_message=None,  # Added this line
        )

    async def _list_browsers(self, **kwargs) -> ToolCallResult:
        start_time = datetime.now()  # Added this line
        sessions_info = []
        for _browser_id, session in self.browser_sessions.items():
            info = session.to_dict()
            try:
                info["current_url"] = session.page.url
                info["title"] = await session.page.title()
            except Exception:  # Catch all exceptions as browser might be closed
                info["current_url"] = "unknown"
                info["title"] = "unknown"
            sessions_info.append(info)
        output = f"Active browser sessions: {len(sessions_info)}"
        if sessions_info:
            for info in sessions_info:
                status = "🟢" if info["browser_id"] == self.active_browser else "⚪"
                output += f"\n  {status} {info['session_id']}: {info['title']} ({info['current_url']})"
        return ToolCallResult(
            command="list_browsers",
            success=True,
            output=output,
            execution_time=(
                datetime.now() - start_time
            ).total_seconds(),  # Changed from 0.1
            metadata={"sessions": sessions_info, "active_browser": self.active_browser},
            error_message=None,  # Added this line
        )

    async def cleanup(self):
        for session in self.browser_sessions.values():
            await session.close()
        self.browser_sessions.clear()
        if self.playwright:
            await self.playwright.stop()
            self.playwright = None

    async def demo(self):
        """Demonstrates the visual UI tools by running the demo script."""
        from crisalida_lib.ASTRAL_TOOLS.demos.visual_ui_interaction_demos import (
            demo_visual_ui,
        )

        return await demo_visual_ui()


if __name__ == "__main__":
    from crisalida_lib.ASTRAL_TOOLS.demos.visual_ui_interaction_demos import (
        demo_visual_ui,
    )

    asyncio.run(demo_visual_ui())
