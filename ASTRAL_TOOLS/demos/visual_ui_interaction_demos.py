#!/usr/bin/env python3
import asyncio

from crisalida_lib.ASTRAL_TOOLS.visual_ui_interaction import VisualUITool


async def demo_visual_ui():
    """Demuestra la herramienta de interacción visual/UI"""
    print("🖥️ VISUAL UI INTERACTION TOOL DEMO")
    print("=" * 50)
    tool = VisualUITool()
    try:
        print("\n1. Starting browser...")
        result = await tool.execute(
            action="start_browser", browser_id="demo_browser", headless=True
        )
        print(f"Start browser: {result.output}")
        print("\n2. Navigating to test page...")
        result = await tool.execute(
            action="navigate",
            browser_id="demo_browser",
            url="https://httpbin.org/forms/post",
        )
        print(f"Navigate: {result.output}")
        print("\n3. Getting page info...")
        result = await tool.execute(action="get_page_info", browser_id="demo_browser")
        print(f"Page info: {result.output}")
        print("\n4. Extracting page text...")
        result = await tool.execute(action="extract_text", browser_id="demo_browser")
        print(f"Extract text: {result.output}")
        print("\n5. Taking screenshot...")
        result = await tool.execute(action="screenshot", browser_id="demo_browser")
        print(f"Screenshot: {result.output}")
        print("\n6. Filling form...")
        form_data = {
            "input[name='custname']": "Test User",
            "input[name='custtel']": "123-456-7890",
            "input[name='custemail']": "test@example.com",
        }
        result = await tool.execute(
            action="fill_form", browser_id="demo_browser", form_data=form_data
        )
        print(f"Fill form: {result.output}")
        print("\n7. Executing JavaScript...")
        result = await tool.execute(
            action="execute_script", browser_id="demo_browser", text="document.title"
        )
        print(f"Execute script: {result.output}")
        print("\n8. Closing browser...")
        result = await tool.execute(action="close_browser", browser_id="demo_browser")
        print(f"Close browser: {result.output}")
    except Exception as e:
        print(f"Demo error: {e}")
    finally:
        await tool.cleanup()
    print("\n✅ Visual UI interaction tool demo completed!")


if __name__ == "__main__":
    asyncio.run(demo_visual_ui())
