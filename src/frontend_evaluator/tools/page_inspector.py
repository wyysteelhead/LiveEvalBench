"""Page inspection tools for understanding page structure."""

import json
from typing import Any, Dict, List, Optional

from .registry import register_tool


@register_tool(
    name="get_page_context",
    description="Get the current page context including accessibility tree snapshot. "
    "Use this to understand what elements are on the page and their structure.",
)
async def get_page_context(executor: Any) -> str:
    """Get page context with accessibility tree.

    Returns a filtered accessibility tree that excludes meaningless container divs
    and focuses on interactive and semantic elements.

    Args:
        executor: PlaywrightExecutor instance

    Returns:
        JSON string with page title, URL, and accessibility tree
    """
    try:
        # Get context from executor
        result = await executor.get_context()

        # Extract data
        title = result.get("title", "")
        url = result.get("url", "")
        snapshot = result.get("accessibility_tree")

        # Filter and simplify the tree
        filtered_tree = _filter_accessibility_tree(snapshot)

        context = {
            "title": title,
            "url": url,
            "accessibility_tree": filtered_tree,
        }

        diagnostics = result.get("diagnostics")
        if diagnostics:
            context["diagnostics"] = diagnostics

        return json.dumps(context, indent=2)

    except Exception as e:
        return f"Failed to get page context: {str(e)}"


@register_tool(
    name="get_global_dom_summary",
    description="Get a structured summary of the current page's global DOM and accessibility structure. "
    "Use this to identify major rendered regions, headings, and interactive element counts.",
)
async def get_global_dom_summary(executor: Any) -> str:
    """Get a concise structural summary of the rendered page.

    Returns a DOM/accessibility-derived summary intended for task planning and
    fast page understanding. Unlike get_page_context, this tool emphasizes
    top-level modules, headings, and interaction density instead of returning
    the full filtered tree.
    """
    try:
        result = await executor.get_context()
        title = result.get("title", "")
        url = result.get("url", "")
        snapshot = result.get("accessibility_tree")
        filtered_tree = _filter_accessibility_tree(snapshot)
        summary = _summarize_accessibility_tree(filtered_tree)
        payload = {
            "title": title,
            "url": url,
            "summary": summary,
        }
        diagnostics = result.get("diagnostics")
        if diagnostics:
            payload["diagnostics"] = diagnostics
        return json.dumps(payload, indent=2)
    except Exception as e:
        return f"Failed to get global DOM summary: {str(e)}"


def _filter_accessibility_tree(node: dict, depth: int = 0, max_depth: int = 10) -> dict:
    """Filter accessibility tree to remove noise and focus on meaningful elements.

    Args:
        node: Accessibility tree node
        depth: Current depth in tree
        max_depth: Maximum depth to traverse

    Returns:
        Filtered node dictionary
    """
    if not node or depth > max_depth:
        return None

    # Skip generic containers without meaningful role or name
    role = node.get("role", "")
    name = node.get("name", "")
    value = node.get("value", "")

    # Skip if generic role and no name/value
    if role == "generic" and not name and not value:
        # But still process children
        children = node.get("children", [])
        filtered_children = []
        for child in children:
            filtered_child = _filter_accessibility_tree(child, depth + 1, max_depth)
            if filtered_child:
                filtered_children.append(filtered_child)
        # Return children directly without wrapper
        return {"children": filtered_children} if filtered_children else None

    # Build filtered node
    filtered = {"role": role}

    if name:
        filtered["name"] = name
    if value:
        filtered["value"] = value

    # Add description if present
    if "description" in node and node["description"]:
        filtered["description"] = node["description"]

    # Recursively filter children
    children = node.get("children", [])
    if children:
        filtered_children = []
        for child in children:
            filtered_child = _filter_accessibility_tree(child, depth + 1, max_depth)
            if filtered_child:
                filtered_children.append(filtered_child)

        if filtered_children:
            filtered["children"] = filtered_children

    return filtered


def _walk_tree(node: Optional[Dict[str, Any]], depth: int = 0) -> List[Dict[str, Any]]:
    if not isinstance(node, dict):
        return []

    items: List[Dict[str, Any]] = []
    role = str(node.get("role", "") or "").strip()
    name = str(node.get("name", "") or "").strip()
    value = str(node.get("value", "") or "").strip()
    description = str(node.get("description", "") or "").strip()
    if role or name or value or description:
        items.append({
            "role": role,
            "name": name,
            "value": value,
            "description": description,
            "depth": depth,
        })

    for child in node.get("children", []) or []:
        items.extend(_walk_tree(child, depth + 1))
    return items


def _summarize_accessibility_tree(tree: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    items = _walk_tree(tree)
    if not items:
        return {
            "inferred_module_count": 0,
            "top_level_regions": [],
            "headings": [],
            "interactive_counts": {},
            "interactive_elements": [],
            "prominent_labels": [],
        }

    top_level_regions: List[Dict[str, str]] = []
    headings: List[str] = []
    prominent_labels: List[str] = []
    interactive_elements: List[Dict[str, Any]] = []
    seen_labels = set()

    interactive_role_groups = {
        "button": "buttons",
        "link": "links",
        "textbox": "inputs",
        "searchbox": "inputs",
        "combobox": "inputs",
        "checkbox": "toggles",
        "radio": "toggles",
        "switch": "toggles",
        "menuitem": "menus",
        "tab": "tabs",
        "dialog": "dialogs",
    }
    interactive_counts: Dict[str, int] = {}
    region_roles = {"main", "navigation", "region", "article", "dialog", "tabpanel", "complementary", "contentinfo", "banner"}

    for item in items:
        role = item["role"]
        name = item["name"]
        depth = int(item["depth"])

        if role in region_roles and depth <= 2:
            top_level_regions.append({
                "role": role,
                "name": name,
            })

        if role == "heading" and name and name not in headings:
            headings.append(name)

        group = interactive_role_groups.get(role)
        if group:
            interactive_counts[group] = interactive_counts.get(group, 0) + 1
            if len(interactive_elements) < 30:
                interactive_elements.append({
                    "role": role,
                    "name": name,
                    "depth": depth,
                })

        label = name or item["value"] or item["description"]
        if label and label not in seen_labels:
            seen_labels.add(label)
            prominent_labels.append(label)

    inferred_module_count = len([region for region in top_level_regions if region.get("role") not in {"banner", "contentinfo"}])
    if inferred_module_count <= 0:
        inferred_module_count = min(max(len(headings), 1), 3)

    return {
        "inferred_module_count": inferred_module_count,
        "top_level_regions": top_level_regions[:12],
        "headings": headings[:12],
        "interactive_counts": interactive_counts,
        "interactive_elements": interactive_elements,
        "prominent_labels": prominent_labels[:20],
    }
