"""Visual checker — screenshot + Claude multimodal analysis."""

import base64
from typing import Any, Dict, List, Optional

from .source_scanner import CheckResult


_VISUAL_QUESTIONS = [
    {
        "check_id": "vis_color_contrast",
        "category": "visual",
        "question": (
            "Does the text appear to have sufficient contrast against its background? "
            "Answer YES if text is clearly readable, NO if any text looks hard to read."
        ),
    },
    {
        "check_id": "vis_layout_overflow",
        "category": "layout",
        "question": (
            "Does the page have any unintended horizontal overflow or broken layout? "
            "Look for elements extending beyond the right edge, unexpected horizontal scrollbars, "
            "or content visually clipped in an unintentional way. "
            "Answer YES if the layout looks correct, NO if you see clear horizontal overflow or broken layout."
        ),
    },
    {
        "check_id": "vis_loading_indicator",
        "category": "feedback",
        "question": (
            "If the application has any slow operations (> ~2 seconds), is there a clear loading indicator "
            "(spinner, skeleton, progress bar)? Fast responses that complete quickly do NOT require a loading indicator. "
            "Answer YES if loading indicators are present where needed, or if all operations are fast enough to not need one. "
            "NO only if there is a visibly slow operation with no feedback. "
            "N/A if no async operations exist."
        ),
    },
    {
        "check_id": "vis_shadow_layering",
        "category": "visual",
        "question": (
            "Do overlapping elements (modals, dropdowns, tooltips) use appropriate "
            "shadow or z-index layering to appear correctly stacked? "
            "Answer YES if layering looks correct, NO if elements appear incorrectly stacked, "
            "N/A if no overlapping elements are present."
        ),
    },
    {
        "check_id": "vis_responsive_mobile",
        "category": "layout",
        "question": (
            "At this mobile viewport width, does the layout look correct? "
            "Answer YES if there are no overlapping elements, broken alignment, or clipped content, "
            "NO if the layout is broken at this width."
        ),
    },
]


async def check(
    executor: Any,
    llm: Any,
    behavior_results: Optional[List[Dict]] = None,
) -> List[CheckResult]:
    """Run visual checks using screenshots and Claude multimodal analysis.

    Args:
        executor: Executor instance implementing ExecutorInterface
        llm: LangChain LLM instance (should support vision/multimodal)
        behavior_results: Optional list of behavior test results for cross-validation

    Returns:
        List of CheckResult for each visual check
    """
    results: List[CheckResult] = []

    # Measure page height to decide whether the screenshot will be partial
    _FULL_PAGE_THRESHOLD = 5000  # px — beyond this, warn LLM the image is partial
    page_height = 720
    try:
        res = await executor.evaluate_js("document.documentElement.scrollHeight")
        page_height = int(res.get("result") or 720)
    except Exception:
        pass

    is_partial = page_height > _FULL_PAGE_THRESHOLD

    # Take full-page screenshot; fall back to viewport if executor doesn't support it
    try:
        screenshot_b64 = await executor.screenshot(full_page=not is_partial)
    except Exception:
        try:
            screenshot_b64 = await executor.screenshot(full_page=False)
            is_partial = True
        except Exception as e:
            results.append(CheckResult(
                check_id="vis_screenshot",
                category="visual",
                passed=False,
                detail=f"Screenshot failed: {e}",
                score=0.0,
            ))
            return results

    # Take mobile screenshot (375px viewport) for vis_responsive_mobile
    mobile_screenshot_b64 = None
    try:
        await executor.evaluate_js(
            "() => { window.resizeTo(375, window.innerHeight); }"
        )
        await executor.evaluate_js(
            "() => new Promise(r => setTimeout(r, 300))"
        )
        mobile_screenshot_b64 = await executor.screenshot(full_page=False)
    except Exception:
        mobile_screenshot_b64 = screenshot_b64  # fallback to desktop

    # Run each visual question
    for idx, check_def in enumerate(_VISUAL_QUESTIONS):
        try:
            shot = mobile_screenshot_b64 if check_def["check_id"] == "vis_responsive_mobile" else screenshot_b64
            # Mobile screenshot is always viewport-only
            shot_is_partial = False if check_def["check_id"] == "vis_responsive_mobile" else is_partial
            result = await _ask_visual_question(
                llm=llm,
                screenshot_b64=shot,
                check_id=check_def["check_id"],
                category=check_def["category"],
                question=check_def["question"],
                behavior_results=behavior_results,
                is_partial=shot_is_partial,
            )
            result.screenshot = shot
            results.append(result)
        except Exception as e:
            results.append(CheckResult(
                check_id=check_def["check_id"],
                category=check_def["category"],
                passed=False,
                detail=f"Visual check failed: {e}",
                score=0.0,
            ))

    return results


async def _ask_visual_question(
    llm: Any,
    screenshot_b64: str,
    check_id: str,
    category: str,
    question: str,
    behavior_results: Optional[List[Dict]] = None,
    is_partial: bool = False,
) -> CheckResult:
    """Ask Claude a visual question about a screenshot."""
    from langchain_core.messages import HumanMessage
    from ..llm.message_utils import get_provider_from_llm, normalize_multimodal_content_for_provider

    context = ""
    if behavior_results:
        relevant = [
            r for r in behavior_results
            if r.get("category") == category
        ]
        if relevant:
            context = "\n\nRelated behavior test results:\n" + "\n".join(
                f"- {r.get('check_id', '?')}: {'PASS' if r.get('passed') else 'FAIL'} — {r.get('detail', '')}"
                for r in relevant
            )

    partial_note = (
        "\n\nNote: This page is very long. The screenshot shows only the top portion of the page. "
        "Base your answer only on what is visible in the image."
    ) if is_partial else ""

    prompt = (
        f"{question}\n\n"
        f"Answer with YES, NO, or N/A followed by a brief explanation (1-2 sentences)."
        f"{partial_note}{context}"
    )

    message = HumanMessage(content=[
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{screenshot_b64}",
            },
        },
        {
            "type": "text",
            "text": prompt,
        },
    ])
    message = HumanMessage(
        content=normalize_multimodal_content_for_provider(
            message.content,
            get_provider_from_llm(llm),
        )
    )

    response = await llm.ainvoke([message])
    answer = response.content.strip().upper()

    if answer.startswith("YES"):
        passed = True
        score = 1.0
    elif answer.startswith("N/A"):
        passed = True  # N/A is not a failure
        score = 1.0
    else:
        passed = False
        score = 0.0

    return CheckResult(
        check_id=check_id,
        category=category,
        passed=passed,
        detail=response.content.strip(),
        score=score,
    )
