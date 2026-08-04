"""Browser action tools for navigation and interaction."""

import asyncio
import json
import time
import uuid
from typing import Any, Dict, List
from .registry import register_tool
from .vision_inspector import analyze_screenshot_b64

PAGE_MUTATION_TOOLS = {
    "navigate",
    "navigate_back",
    "navigate_forward",
    "reload_page",
    "click_element",
    "preview_click_at",
    "click_at",
    "dblclick_at",
    "dblclick_element",
    "right_click_element",
    "hover_element",
    "type_text",
    "clear_text",
    "select_option",
    "check_element",
    "uncheck_element",
    "focus_element",
    "press_key",
    "press_key_on",
    "action_sequence",
    "scroll_by",
    "scroll_to_element",
    "drag_and_drop",
}

SELECTOR_ARG_BY_TOOL = {
    "click_element": "selector",
    "dblclick_element": "selector",
    "right_click_element": "selector",
    "hover_element": "selector",
    "type_text": "selector",
    "clear_text": "selector",
    "select_option": "selector",
    "check_element": "selector",
    "uncheck_element": "selector",
    "focus_element": "selector",
    "press_key_on": "selector",
    "scroll_to_element": "selector",
    "drag_and_drop": "source_selector",
}

CLICK_PREVIEW_TTL_SECONDS = 120
CLICK_PREVIEW_MARKER_ID = "__frontend_evaluator_click_preview_marker"
_PENDING_CLICK_PREVIEWS: Dict[str, Dict[str, Any]] = {}
DEFAULT_PREVIEW_CLICK_QUESTION = "Is the red marker positioned on a reasonable clickable target? If clicked, what is the most likely result? If uncertain, say so explicitly."


def _build_preview_tool_result(summary_text: str, screenshot_b64: str | None) -> Dict[str, Any]:
    """Create a tool result that preserves preview text and optionally attaches an image."""
    tool_message_content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": summary_text,
        }
    ]
    if screenshot_b64:
        tool_message_content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{screenshot_b64}"},
            }
        )
    return {
        "display_text": summary_text,
        "tool_message_content": tool_message_content,
        "preview_screenshot_b64": screenshot_b64,
    }


async def _run_js_action(executor: Any, script: str, action_name: str) -> str:
    """Execute a JS action and normalize success/error messaging."""
    try:
        raw = await executor.evaluate_js(script)
        result = raw.get("result") if isinstance(raw, dict) else raw
        if isinstance(result, dict):
            if result.get("ok"):
                return f"Successfully {action_name}"
            error = result.get("error", "unknown error")
            return f"{action_name.capitalize()} failed: {error}"
        if result is False:
            return f"{action_name.capitalize()} failed"
        return f"Successfully {action_name}"
    except Exception as e:
        return f"{action_name.capitalize()} failed: {str(e)}"


async def _get_point_snapshot(executor: Any, x: int, y: int) -> Dict[str, Any]:
    """Get page state and hit-target info for a viewport coordinate."""
    script = f"""
(() => {{
  const x = {int(x)};
  const y = {int(y)};
  const el = document.elementFromPoint(x, y);
  const inViewport = x >= 0 && y >= 0 && x <= window.innerWidth && y <= window.innerHeight;
  return {{
    ok: true,
    url: window.location.href,
    viewport: {{ width: window.innerWidth, height: window.innerHeight }},
    scroll: {{ x: window.scrollX, y: window.scrollY }},
    dpr: window.devicePixelRatio || 1,
    in_viewport: inViewport,
    target: el ? {{
      tag: (el.tagName || "").toLowerCase(),
      id: el.id || "",
      classes: typeof el.className === "string" ? el.className : "",
      role: el.getAttribute("role") || "",
      text: (el.innerText || el.textContent || "").trim().slice(0, 120)
    }} : null
  }};
}})()
"""
    raw = await executor.evaluate_js(script)
    return raw.get("result") if isinstance(raw, dict) else raw


async def _set_click_preview_marker(executor: Any, x: int, y: int, token: str) -> None:
    marker_id_js = json.dumps(CLICK_PREVIEW_MARKER_ID)
    token_js = json.dumps(token)
    script = f"""
(() => {{
  const markerId = {marker_id_js};
  const token = {token_js};
  let marker = document.getElementById(markerId);
  if (!marker) {{
    marker = document.createElement('div');
    marker.id = markerId;
    marker.style.position = 'fixed';
    marker.style.width = '18px';
    marker.style.height = '18px';
    marker.style.marginLeft = '-9px';
    marker.style.marginTop = '-9px';
    marker.style.border = '3px solid #ff2d20';
    marker.style.borderRadius = '9999px';
    marker.style.background = 'rgba(255, 45, 32, 0.2)';
    marker.style.boxShadow = '0 0 0 2px rgba(255,255,255,0.95)';
    marker.style.pointerEvents = 'none';
    marker.style.zIndex = '2147483647';
    document.documentElement.appendChild(marker);
  }}
  marker.style.left = '{int(x)}px';
  marker.style.top = '{int(y)}px';
  marker.setAttribute('data-preview-token', token);
  return {{ ok: true }};
}})()
"""
    await executor.evaluate_js(script)


async def _clear_click_preview_marker(executor: Any) -> None:
    marker_id_js = json.dumps(CLICK_PREVIEW_MARKER_ID)
    script = f"""
(() => {{
  const marker = document.getElementById({marker_id_js});
  if (marker && marker.parentNode) marker.parentNode.removeChild(marker);
  return {{ ok: true }};
}})()
"""
    await executor.evaluate_js(script)


@register_tool(
    name="navigate",
    description="Navigate to a URL in the browser. Use this to load the application page.",
)
async def navigate(executor: Any, url: str) -> str:
    """Navigate to a URL.

    Args:
        executor: PlaywrightExecutor instance
        url: URL to navigate to

    Returns:
        Success message with final URL
    """
    try:
        result = await executor.navigate(url)
        if isinstance(result, dict) and not result.get("success"):
            return f"Navigation to {url} failed: {result.get('error', 'unknown error')}"
        return f"Successfully navigated to {result['url']}"
    except TimeoutError:
        return f"Navigation to {url} timed out after 30 seconds"
    except Exception as e:
        return f"Navigation failed: {str(e)}"


@register_tool(
    name="navigate_back",
    description="Navigate back in browser history.",
)
async def navigate_back(executor: Any) -> str:
    script = "(() => { history.back(); return { ok: true }; })()"
    return await _run_js_action(executor, script, "navigated back")


@register_tool(
    name="navigate_forward",
    description="Navigate forward in browser history.",
)
async def navigate_forward(executor: Any) -> str:
    script = "(() => { history.forward(); return { ok: true }; })()"
    return await _run_js_action(executor, script, "navigated forward")


@register_tool(
    name="reload_page",
    description="Reload the current page.",
)
async def reload_page(executor: Any) -> str:
    script = "(() => { location.reload(); return { ok: true }; })()"
    return await _run_js_action(executor, script, "reloaded page")


@register_tool(
    name="click_element",
    description="Click an element on the page. Supports text selectors (text=Button), "
    "role selectors (role=button[name='Submit']), and CSS selectors.",
)
async def click_element(executor: Any, selector: str) -> str:
    """Click an element on the page.

    Args:
        executor: PlaywrightExecutor instance
        selector: Element selector (text=, role=, or CSS)

    Returns:
        Success or error message
    """
    try:
        await executor.click(selector)
        return f"Successfully clicked element: {selector}"
    except TimeoutError:
        return f"Element not found or not clickable: {selector}"
    except Exception as e:
        return f"Click failed: {str(e)}"


@register_tool(
    name="preview_click_at",
    description=(
        "Preview a coordinate click by placing a red marker at viewport (x, y). "
        "Returns a confirmation token, hit-target info, and an annotated preview screenshot. "
        "If no question is provided, a default visual question will be asked about the likely reaction to clicking the marked point."
    ),
)
async def preview_click_at(
    executor: Any,
    x: int,
    y: int,
    question: str | None = None,
    full_page: bool = False,
) -> str:
    """Preview a coordinate click, optionally answer a visual question, and issue a confirmation token."""
    screenshot_b64 = None
    resolved_question = question or DEFAULT_PREVIEW_CLICK_QUESTION
    try:
        snapshot = await _get_point_snapshot(executor, x, y)
        token = uuid.uuid4().hex
        await _set_click_preview_marker(executor, x, y, token)
        screenshot_b64 = await executor.screenshot(full_page=full_page)
    except Exception as e:
        return f"Preview failed: {str(e)}"

    _PENDING_CLICK_PREVIEWS[token] = {
        "x": int(x),
        "y": int(y),
        "url": snapshot.get("url"),
        "viewport": snapshot.get("viewport"),
        "scroll": snapshot.get("scroll"),
        "created_at": time.time(),
    }

    target = snapshot.get("target") or {}
    summary = (
        f"Click preview ready. token={token}; point=({int(x)}, {int(y)}); "
        f"in_viewport={snapshot.get('in_viewport', False)}; "
        f"target={target.get('tag', 'none')}#{target.get('id', '')}; "
        f"text='{target.get('text', '')[:60]}'. "
        "The preview screenshot is attached to this tool result. Review the red marker position yourself before deciding the next step."
    )

    visual_result = await analyze_screenshot_b64(resolved_question, screenshot_b64)
    summary = f"{summary} Question: {resolved_question} Visual answer: {visual_result}"

    return _build_preview_tool_result(summary, screenshot_b64)


@register_tool(
    name="click_at",
    description=(
        "Execute a coordinate click at viewport (x, y) using a token from preview_click_at. "
        "Validates URL and scroll state before clicking."
    ),
)
async def click_at(executor: Any, x: int, y: int, token: str) -> str:
    """Execute coordinate click with two-step token confirmation."""
    preview = _PENDING_CLICK_PREVIEWS.get(token)
    if not preview:
        return "Click failed: invalid or expired preview token. Run preview_click_at first."

    now = time.time()
    if now - float(preview.get("created_at", 0.0)) > CLICK_PREVIEW_TTL_SECONDS:
        _PENDING_CLICK_PREVIEWS.pop(token, None)
        return "Click failed: preview token expired. Run preview_click_at again."

    if int(x) != int(preview["x"]) or int(y) != int(preview["y"]):
        return (
            f"Click failed: coordinates do not match preview token "
            f"({preview['x']}, {preview['y']})."
        )

    try:
        snapshot = await _get_point_snapshot(executor, x, y)
    except Exception as e:
        return f"Click failed: unable to validate current page state: {str(e)}"

    expected_url = preview.get("url")
    expected_scroll = preview.get("scroll") or {}
    current_scroll = snapshot.get("scroll") or {}
    if snapshot.get("url") != expected_url:
        return "Click failed: URL changed since preview. Run preview_click_at again."
    if (
        int(current_scroll.get("x", 0)) != int(expected_scroll.get("x", 0))
        or int(current_scroll.get("y", 0)) != int(expected_scroll.get("y", 0))
    ):
        return "Click failed: scroll position changed since preview. Run preview_click_at again."

    try:
        await executor.click_at(int(x), int(y))
        await _clear_click_preview_marker(executor)
    except Exception as e:
        return f"Click failed: {str(e)}"
    finally:
        _PENDING_CLICK_PREVIEWS.pop(token, None)

    target = snapshot.get("target") or {}
    return (
        f"Successfully clicked at ({int(x)}, {int(y)}), "
        f"target={target.get('tag', 'none')}#{target.get('id', '')}."
    )


@register_tool(
    name="dblclick_at",
    description=(
        "Execute a coordinate double-click at viewport (x, y) using a token from preview_click_at. "
        "Use this when the intended gesture is explicitly a double-click rather than two repeated single clicks. "
        "Validates URL and scroll state before executing."
    ),
)
async def dblclick_at(executor: Any, x: int, y: int, token: str) -> str:
    """Execute coordinate double-click with two-step token confirmation."""
    preview = _PENDING_CLICK_PREVIEWS.get(token)
    if not preview:
        return "Double-click failed: invalid or expired preview token. Run preview_click_at first."

    now = time.time()
    if now - float(preview.get("created_at", 0.0)) > CLICK_PREVIEW_TTL_SECONDS:
        _PENDING_CLICK_PREVIEWS.pop(token, None)
        return "Double-click failed: preview token expired. Run preview_click_at again."

    if int(x) != int(preview["x"]) or int(y) != int(preview["y"]):
        return (
            f"Double-click failed: coordinates do not match preview token "
            f"({preview['x']}, {preview['y']})."
        )

    try:
        snapshot = await _get_point_snapshot(executor, x, y)
    except Exception as e:
        return f"Double-click failed: unable to validate current page state: {str(e)}"

    expected_url = preview.get("url")
    expected_scroll = preview.get("scroll") or {}
    current_scroll = snapshot.get("scroll") or {}
    if snapshot.get("url") != expected_url:
        return "Double-click failed: URL changed since preview. Run preview_click_at again."
    if (
        int(current_scroll.get("x", 0)) != int(expected_scroll.get("x", 0))
        or int(current_scroll.get("y", 0)) != int(expected_scroll.get("y", 0))
    ):
        return "Double-click failed: scroll position changed since preview. Run preview_click_at again."

    try:
        await executor.dblclick_at(int(x), int(y))
        await _clear_click_preview_marker(executor)
    except Exception as e:
        return f"Double-click failed: {str(e)}"
    finally:
        _PENDING_CLICK_PREVIEWS.pop(token, None)

    target = snapshot.get("target") or {}
    return (
        f"Successfully double-clicked at ({int(x)}, {int(y)}), "
        f"target={target.get('tag', 'none')}#{target.get('id', '')}."
    )


@register_tool(
    name="dblclick_element",
    description="Double-click an element on the page.",
)
async def dblclick_element(executor: Any, selector: str) -> str:
    escaped = json.dumps(selector)
    script = f"""
(() => {{
  const el = document.querySelector({escaped});
  if (!el) return {{ ok: false, error: "element not found" }};
  el.dispatchEvent(new MouseEvent("dblclick", {{ bubbles: true, cancelable: true, view: window }}));
  return {{ ok: true }};
}})()
"""
    return await _run_js_action(executor, script, f"double-clicked element: {selector}")


@register_tool(
    name="right_click_element",
    description="Right-click (context menu click) an element on the page.",
)
async def right_click_element(executor: Any, selector: str) -> str:
    escaped = json.dumps(selector)
    script = f"""
(() => {{
  const el = document.querySelector({escaped});
  if (!el) return {{ ok: false, error: "element not found" }};
  el.dispatchEvent(new MouseEvent("contextmenu", {{ bubbles: true, cancelable: true, button: 2, view: window }}));
  return {{ ok: true }};
}})()
"""
    return await _run_js_action(executor, script, f"right-clicked element: {selector}")


@register_tool(
    name="type_text",
    description="Type text into an input field. Use this to fill forms or text inputs.",
)
async def type_text(executor: Any, selector: str, text: str) -> str:
    """Type text into an input field.

    Args:
        executor: PlaywrightExecutor instance
        selector: Input element selector
        text: Text to type

    Returns:
        Success or error message
    """
    try:
        await executor.type_text(selector, text)
        return f"Successfully typed '{text}' into {selector}"
    except TimeoutError:
        return f"Input field not found: {selector}"
    except Exception as e:
        return f"Type failed: {str(e)}"


@register_tool(
    name="clear_text",
    description="Clear text from an input or textarea element.",
)
async def clear_text(executor: Any, selector: str) -> str:
    escaped = json.dumps(selector)
    script = f"""
(() => {{
  const el = document.querySelector({escaped});
  if (!el) return {{ ok: false, error: "element not found" }};
  if (!("value" in el)) return {{ ok: false, error: "element has no value property" }};
  el.value = "";
  el.dispatchEvent(new Event("input", {{ bubbles: true }}));
  el.dispatchEvent(new Event("change", {{ bubbles: true }}));
  return {{ ok: true }};
}})()
"""
    return await _run_js_action(executor, script, f"cleared text in {selector}")


@register_tool(
    name="hover_element",
    description="Hover over an element to trigger :hover styles and interactions. "
    "Supports text selectors (text=Button), role selectors (role=button[name='Submit']), and CSS selectors.",
)
async def hover_element(executor: Any, selector: str) -> str:
    """Hover over an element on the page.

    Args:
        executor: PlaywrightExecutor instance
        selector: Element selector (text=, role=, or CSS)

    Returns:
        Success or error message
    """
    try:
        await executor.hover(selector)
        return f"Successfully hovered element: {selector}"
    except TimeoutError:
        return f"Element not found or not hoverable: {selector}"
    except Exception as e:
        return f"Hover failed: {str(e)}"


@register_tool(
    name="select_option",
    description="Select an option in a <select> by value, label text, or index.",
)
async def select_option(
    executor: Any,
    selector: str,
    value: str = "",
    label: str = "",
    index: int = -1,
) -> str:
    selector_js = json.dumps(selector)
    value_js = json.dumps(value)
    label_js = json.dumps(label)
    script = f"""
(() => {{
  const el = document.querySelector({selector_js});
  if (!el) return {{ ok: false, error: "element not found" }};
  if (!(el instanceof HTMLSelectElement)) return {{ ok: false, error: "element is not a select" }};

  const value = {value_js};
  const label = {label_js};
  const index = {index};

  let found = false;
  if (value) {{
    found = Array.from(el.options).some((opt) => opt.value === value);
    if (found) el.value = value;
  }} else if (label) {{
    const match = Array.from(el.options).find((opt) => opt.text.trim() === label.trim());
    if (match) {{
      el.value = match.value;
      found = true;
    }}
  }} else if (Number.isInteger(index) && index >= 0 && index < el.options.length) {{
    el.selectedIndex = index;
    found = true;
  }} else {{
    return {{ ok: false, error: "provide one of value, label, or valid index" }};
  }}

  if (!found) return {{ ok: false, error: "option not found" }};
  el.dispatchEvent(new Event("input", {{ bubbles: true }}));
  el.dispatchEvent(new Event("change", {{ bubbles: true }}));
  return {{ ok: true }};
}})()
"""
    return await _run_js_action(executor, script, f"selected option in {selector}")


@register_tool(
    name="check_element",
    description="Check a checkbox or radio input element.",
)
async def check_element(executor: Any, selector: str) -> str:
    escaped = json.dumps(selector)
    script = f"""
(() => {{
  const el = document.querySelector({escaped});
  if (!el) return {{ ok: false, error: "element not found" }};
  if (!(el instanceof HTMLInputElement)) return {{ ok: false, error: "element is not an input" }};
  if (!["checkbox", "radio"].includes(el.type)) return {{ ok: false, error: "input is not checkbox/radio" }};
  el.checked = true;
  el.dispatchEvent(new Event("input", {{ bubbles: true }}));
  el.dispatchEvent(new Event("change", {{ bubbles: true }}));
  return {{ ok: true }};
}})()
"""
    return await _run_js_action(executor, script, f"checked element: {selector}")


@register_tool(
    name="uncheck_element",
    description="Uncheck a checkbox input element.",
)
async def uncheck_element(executor: Any, selector: str) -> str:
    escaped = json.dumps(selector)
    script = f"""
(() => {{
  const el = document.querySelector({escaped});
  if (!el) return {{ ok: false, error: "element not found" }};
  if (!(el instanceof HTMLInputElement)) return {{ ok: false, error: "element is not an input" }};
  if (el.type !== "checkbox") return {{ ok: false, error: "only checkbox can be unchecked" }};
  el.checked = false;
  el.dispatchEvent(new Event("input", {{ bubbles: true }}));
  el.dispatchEvent(new Event("change", {{ bubbles: true }}));
  return {{ ok: true }};
}})()
"""
    return await _run_js_action(executor, script, f"unchecked element: {selector}")


@register_tool(
    name="focus_element",
    description="Focus an element by selector.",
)
async def focus_element(executor: Any, selector: str) -> str:
    escaped = json.dumps(selector)
    script = f"""
(() => {{
  const el = document.querySelector({escaped});
  if (!el) return {{ ok: false, error: "element not found" }};
  if (typeof el.focus !== "function") return {{ ok: false, error: "element is not focusable" }};
  el.focus();
  return {{ ok: true }};
}})()
"""
    return await _run_js_action(executor, script, f"focused element: {selector}")


@register_tool(
    name="press_key",
    description="Press a keyboard key on the currently focused element. Use duration (ms) for long press (e.g. duration=500 to hold for 500ms).",
)
async def press_key(executor: Any, key: str, duration: int = 0) -> str:
    if hasattr(executor, "keyboard_press"):
        await executor.keyboard_press(key, duration)
        return f"Pressed key: {key}" + (f" (hold {duration}ms)" if duration > 0 else "")

    key_js = json.dumps(key)
    if duration > 0:
        script = f"""
(async () => {{
  const target = document.activeElement || document.body;
  if (!target) return {{ ok: false, error: "no active element" }};
  const key = {key_js};
  target.dispatchEvent(new KeyboardEvent("keydown", {{ key, bubbles: true, cancelable: true }}));
  target.dispatchEvent(new KeyboardEvent("keypress", {{ key, bubbles: true, cancelable: true }}));
  await new Promise(r => setTimeout(r, {duration}));
  target.dispatchEvent(new KeyboardEvent("keyup", {{ key, bubbles: true, cancelable: true }}));
  if (key.length === 1 && "value" in target && !target.readOnly && !target.disabled) {{
    target.value = `${{target.value || ""}}${{key}}`;
    target.dispatchEvent(new Event("input", {{ bubbles: true }}));
  }}
  return {{ ok: true }};
}})()
"""
    else:
        script = f"""
(() => {{
  const target = document.activeElement || document.body;
  if (!target) return {{ ok: false, error: "no active element" }};
  const key = {key_js};
  const down = new KeyboardEvent("keydown", {{ key, bubbles: true, cancelable: true }});
  const press = new KeyboardEvent("keypress", {{ key, bubbles: true, cancelable: true }});
  const up = new KeyboardEvent("keyup", {{ key, bubbles: true, cancelable: true }});
  target.dispatchEvent(down);
  target.dispatchEvent(press);
  target.dispatchEvent(up);
  if (key.length === 1 && "value" in target && !target.readOnly && !target.disabled) {{
    target.value = `${{target.value || ""}}${{key}}`;
    target.dispatchEvent(new Event("input", {{ bubbles: true }}));
  }}
  return {{ ok: true }};
}})()
"""
    return await _run_js_action(executor, script, f"pressed key: {key}" + (f" (hold {duration}ms)" if duration > 0 else ""))


@register_tool(
    name="press_key_on",
    description="Focus an element then press a keyboard key on it. Use duration (ms) for long press.",
)
async def press_key_on(executor: Any, selector: str, key: str, duration: int = 0) -> str:
    if hasattr(executor, "keyboard_press"):
        await executor.evaluate_js(f"(() => {{ const el = document.querySelector({json.dumps(selector)}); if (el && typeof el.focus === 'function') el.focus(); }})()")
        await executor.keyboard_press(key, duration)
        return f"Pressed key {key} on {selector}" + (f" (hold {duration}ms)" if duration > 0 else "")

    selector_js = json.dumps(selector)
    key_js = json.dumps(key)
    if duration > 0:
        script = f"""
(async () => {{
  const el = document.querySelector({selector_js});
  if (!el) return {{ ok: false, error: "element not found" }};
  if (typeof el.focus === "function") el.focus();
  const key = {key_js};
  el.dispatchEvent(new KeyboardEvent("keydown", {{ key, bubbles: true, cancelable: true }}));
  el.dispatchEvent(new KeyboardEvent("keypress", {{ key, bubbles: true, cancelable: true }}));
  await new Promise(r => setTimeout(r, {duration}));
  el.dispatchEvent(new KeyboardEvent("keyup", {{ key, bubbles: true, cancelable: true }}));
  if (key.length === 1 && "value" in el && !el.readOnly && !el.disabled) {{
    el.value = `${{el.value || ""}}${{key}}`;
    el.dispatchEvent(new Event("input", {{ bubbles: true }}));
  }}
  return {{ ok: true }};
}})()
"""
    else:
        script = f"""
(() => {{
  const el = document.querySelector({selector_js});
  if (!el) return {{ ok: false, error: "element not found" }};
  if (typeof el.focus === "function") el.focus();
  const key = {key_js};
  const down = new KeyboardEvent("keydown", {{ key, bubbles: true, cancelable: true }});
  const press = new KeyboardEvent("keypress", {{ key, bubbles: true, cancelable: true }});
  const up = new KeyboardEvent("keyup", {{ key, bubbles: true, cancelable: true }});
  el.dispatchEvent(down);
  el.dispatchEvent(press);
  el.dispatchEvent(up);
  if (key.length === 1 && "value" in el && !el.readOnly && !el.disabled) {{
    el.value = `${{el.value || ""}}${{key}}`;
    el.dispatchEvent(new Event("input", {{ bubbles: true }}));
  }}
  return {{ ok: true }};
}})()
"""
    return await _run_js_action(executor, script, f"pressed key {key} on {selector}" + (f" (hold {duration}ms)" if duration > 0 else ""))


@register_tool(
    name="scroll_by",
    description="Scroll the page by x/y pixel offsets.",
)
async def scroll_by(executor: Any, x: int = 0, y: int = 300) -> str:
    script = f"(() => {{ window.scrollBy({x}, {y}); return {{ ok: true }}; }})()"
    return await _run_js_action(executor, script, f"scrolled page by ({x}, {y})")


@register_tool(
    name="scroll_to_element",
    description="Scroll an element into view.",
)
async def scroll_to_element(executor: Any, selector: str) -> str:
    escaped = json.dumps(selector)
    script = f"""
(() => {{
  const el = document.querySelector({escaped});
  if (!el) return {{ ok: false, error: "element not found" }};
  el.scrollIntoView({{ behavior: "instant", block: "center", inline: "center" }});
  return {{ ok: true }};
}})()
"""
    return await _run_js_action(executor, script, f"scrolled to element: {selector}")


@register_tool(
    name="drag_and_drop",
    description="Drag an element onto another element.",
)
async def drag_and_drop(executor: Any, source_selector: str, target_selector: str) -> str:
    source_js = json.dumps(source_selector)
    target_js = json.dumps(target_selector)
    script = f"""
(() => {{
  const source = document.querySelector({source_js});
  const target = document.querySelector({target_js});
  if (!source) return {{ ok: false, error: "source element not found" }};
  if (!target) return {{ ok: false, error: "target element not found" }};
  const dataTransfer = new DataTransfer();
  source.dispatchEvent(new DragEvent("dragstart", {{ bubbles: true, cancelable: true, dataTransfer }}));
  target.dispatchEvent(new DragEvent("dragover", {{ bubbles: true, cancelable: true, dataTransfer }}));
  target.dispatchEvent(new DragEvent("drop", {{ bubbles: true, cancelable: true, dataTransfer }}));
  source.dispatchEvent(new DragEvent("dragend", {{ bubbles: true, cancelable: true, dataTransfer }}));
  return {{ ok: true }};
}})()
"""
    return await _run_js_action(
        executor, script, f"dragged {source_selector} to {target_selector}"
    )


@register_tool(
    name="action_sequence",
    description=(
        "Execute a rapid sequence of browser actions without LLM round-trips between them. "
        "Use when you need to observe before/after state (e.g. hover comparison), operate games/animations, "
        "or perform any multi-step interaction that must happen faster than individual tool calls allow. "
        "Supported action types: "
        '{"key": "<key>", "hold_ms": <ms>} — press a key (hold_ms=0 for instant press-release, >0 to hold); '
        '{"click": "<selector>"} — click an element by CSS/text/role selector; '
        '{"hover": "<selector>"} — hover an element; '
        '{"scroll": {"x": 0, "y": 300}} — scroll by pixel offsets; '
        '{"screenshot": true, "label": "<name>"} — capture a screenshot at this point; '
        '{"wait_ms": <ms>} — pause between actions. '
        "Returns a summary of all actions plus any captured screenshots for visual comparison."
    ),
)
async def action_sequence(executor: Any, actions: List[Dict[str, Any]]) -> str:
    """Execute a rapid multi-action sequence with optional mid-sequence screenshots."""
    if not actions:
        return "No actions provided."

    use_playwright = hasattr(executor, "keyboard_press")
    results: List[str] = []
    screenshots: List[Dict[str, Any]] = []

    for idx, action in enumerate(actions):
        # Auto-repair: if the LLM passed a string, try to parse it as JSON
        if isinstance(action, str):
            try:
                import json as _json
                action = _json.loads(action)
            except Exception:
                results.append(f"[{idx}] skipped: not a dict (got string that is not valid JSON: {action[:100]})")
                continue

        if not isinstance(action, dict):
            results.append(f"[{idx}] skipped: not a dict (got {type(action).__name__})")
            continue

        if action.get("screenshot"):
            label = str(action.get("label", f"step_{idx}"))
            try:
                b64 = await executor.screenshot(full_page=False)
                screenshots.append({"label": label, "b64": b64})
                results.append(f"[{idx}] screenshot '{label}' captured")
            except Exception as e:
                results.append(f"[{idx}] screenshot failed: {e}")

        elif "wait_ms" in action:
            ms = max(0, int(action["wait_ms"]))
            if use_playwright:
                await executor.page.wait_for_timeout(ms)
            else:
                await asyncio.sleep(ms / 1000.0)
            results.append(f"[{idx}] waited {ms}ms")

        elif "hover" in action:
            selector = str(action["hover"])
            try:
                await executor.hover(selector)
                results.append(f"[{idx}] hovered '{selector}'")
            except Exception as e:
                results.append(f"[{idx}] hover '{selector}' failed: {e}")

        elif "click" in action:
            selector = str(action["click"])
            try:
                await executor.click(selector)
                results.append(f"[{idx}] clicked '{selector}'")
            except Exception as e:
                results.append(f"[{idx}] click '{selector}' failed: {e}")

        elif "scroll" in action:
            scroll = action["scroll"]
            sx = int(scroll.get("x", 0)) if isinstance(scroll, dict) else 0
            sy = int(scroll.get("y", 300)) if isinstance(scroll, dict) else int(scroll)
            script = f"(() => {{ window.scrollBy({sx}, {sy}); return {{ ok: true }}; }})()"
            await _run_js_action(executor, script, f"scroll ({sx}, {sy})")
            results.append(f"[{idx}] scrolled ({sx}, {sy})")

        elif "key" in action:
            key = str(action["key"])
            hold_ms = max(0, int(action.get("hold_ms", 0)))
            if use_playwright:
                if hold_ms > 0:
                    await executor.page.keyboard.down(key)
                    await executor.page.wait_for_timeout(hold_ms)
                    await executor.page.keyboard.up(key)
                else:
                    await executor.page.keyboard.press(key)
            else:
                key_js = json.dumps(key)
                if hold_ms > 0:
                    script = f"""
(async () => {{
  const target = document.activeElement || document.body;
  const key = {key_js};
  target.dispatchEvent(new KeyboardEvent("keydown", {{ key, code: "Key" + key.toUpperCase(), bubbles: true, cancelable: true }}));
  await new Promise(r => setTimeout(r, {hold_ms}));
  target.dispatchEvent(new KeyboardEvent("keyup", {{ key, code: "Key" + key.toUpperCase(), bubbles: true, cancelable: true }}));
  return {{ ok: true }};
}})()
"""
                else:
                    script = f"""
(() => {{
  const target = document.activeElement || document.body;
  const key = {key_js};
  target.dispatchEvent(new KeyboardEvent("keydown", {{ key, code: "Key" + key.toUpperCase(), bubbles: true, cancelable: true }}));
  target.dispatchEvent(new KeyboardEvent("keypress", {{ key, code: "Key" + key.toUpperCase(), bubbles: true, cancelable: true }}));
  target.dispatchEvent(new KeyboardEvent("keyup", {{ key, code: "Key" + key.toUpperCase(), bubbles: true, cancelable: true }}));
  return {{ ok: true }};
}})()
"""
                await _run_js_action(executor, script, f"key {key}")
            results.append(f"[{idx}] key '{key}'" + (f" hold {hold_ms}ms" if hold_ms else ""))

        else:
            results.append(f"[{idx}] skipped: unrecognized action")

    summary = f"Executed {len(actions)} actions: " + "; ".join(results)

    if not screenshots:
        return summary

    content_parts: List[Dict[str, Any]] = [{"type": "text", "text": summary}]
    for ss in screenshots:
        content_parts.append({
            "type": "text",
            "text": f"\n--- Screenshot: {ss['label']} ---",
        })
        content_parts.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{ss['b64']}"},
        })

    return {
        "display_text": summary,
        "tool_message_content": content_parts,
    }


@register_tool(
    name="wait_for_milliseconds",
    description="Wait for a period to allow animations or async UI updates.",
)
async def wait_for_milliseconds(milliseconds: int = 500) -> str:
    if milliseconds < 0:
        return "Wait failed: milliseconds must be >= 0"
    await asyncio.sleep(milliseconds / 1000.0)
    return f"Successfully waited for {milliseconds}ms"
