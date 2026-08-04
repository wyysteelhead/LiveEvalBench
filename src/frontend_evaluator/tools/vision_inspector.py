"""Vision inspection tool for on-demand screenshot analysis."""

from typing import Any, Optional

from ..llm.message_utils import get_provider_from_llm, normalize_multimodal_content_for_provider
from .registry import register_tool


_VISION_LLM: Optional[Any] = None


def set_vision_llm(llm: Any) -> None:
    """Set the multimodal LLM used by screenshot inspection tools."""
    global _VISION_LLM
    _VISION_LLM = llm


async def analyze_screenshot_b64(question: str, screenshot_b64: str) -> str:
    """Ask the vision model to answer a question about a provided screenshot."""
    if _VISION_LLM is None:
        return "Vision inspection unavailable: multimodal LLM is not configured."

    prompt = (
        "You are verifying a frontend UI from a screenshot. "
        "Answer the user's question with direct visual evidence. "
        "Be concise and explicit about what is visible vs uncertain.\n\n"
        "IMPORTANT rules:\n"
        "1. If you cannot confidently determine the answer from the screenshot alone, "
        "respond 'UNCERTAIN: <reason>' rather than guessing.\n"
        "2. Never assume a state (checked/unchecked, changed/unchanged, present/absent) "
        "based on expectation alone — only report what you can directly observe in the image.\n"
        "3. If the question asks whether something changed (e.g. after a click or hover), "
        "but you only have this single screenshot with no prior state for comparison, "
        "explicitly say you cannot compare and respond UNCERTAIN.\n\n"
        f"Question: {question}"
    )

    try:
        from langchain_core.messages import HumanMessage
    except Exception as e:
        return f"Vision inspection unavailable: {e}"

    message = HumanMessage(content=[
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{screenshot_b64}"},
        },
        {
            "type": "text",
            "text": prompt,
        },
    ])
    message = HumanMessage(
        content=normalize_multimodal_content_for_provider(
            message.content,
            get_provider_from_llm(_VISION_LLM),
        )
    )

    try:
        response = await _VISION_LLM.ainvoke([message])
        return str(response.content).strip()
    except Exception as e:
        return f"Vision analysis failed: {e}"


@register_tool(
    name="inspect_last_screenshot",
    description=(
        "Analyze the current page screenshot to answer a visual question. "
        "Use this when you need visual evidence (hover styles, contrast, layout issues). "
        "LIMITATION: This tool only captures the CURRENT state. For before/after comparisons "
        "(e.g. state changes after click/hover), use action_sequence with inline screenshots instead. "
        "The vision model will honestly report UNCERTAIN when it cannot determine the answer from the screenshot alone."
    ),
)
async def inspect_last_screenshot(executor: Any, question: str, full_page: bool = False) -> str:
    """Capture a screenshot and ask the vision model to answer the question."""
    try:
        screenshot_b64 = await executor.screenshot(full_page=full_page)
    except Exception as e:
        return f"Failed to capture screenshot: {e}"
    return await analyze_screenshot_b64(question, screenshot_b64)
