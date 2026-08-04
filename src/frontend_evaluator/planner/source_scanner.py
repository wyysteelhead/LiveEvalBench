"""Source code static analysis for frontend guidelines."""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .rubric_config import get_source_scanner_checks


@dataclass
class CheckResult:
    """Result of a single guideline check."""
    check_id: str
    category: str
    passed: bool
    detail: str
    score: float = field(default=1.0)  # 1.0 = pass, 0.0 = fail
    screenshot: Optional[str] = field(default=None)  # base64 PNG, visual checks only

    def to_dict(self) -> dict:
        d = {
            "check_id": self.check_id,
            "category": self.category,
            "passed": self.passed,
            "detail": self.detail,
            "score": self.score,
        }
        if self.screenshot:
            d["screenshot"] = self.screenshot
        return d


def scan(files: Dict[str, str]) -> List[CheckResult]:
    """Run static analysis checks on source files.

    Args:
        files: Dict mapping filename to file content

    Returns:
        List of CheckResult for each check performed
    """
    all_source = "\n".join(files.values())
    results: List[CheckResult] = []

    check_defs = get_source_scanner_checks()
    heading_levels, has_headings = _extract_heading_levels(all_source)

    for c in check_defs:
        check_type = str(c.get("type", "")).strip()
        check_id = str(c.get("check_id", "")).strip() or "src_unknown"
        category = str(c.get("category", "")).strip() or "source"
        pattern = str(c.get("pattern", ""))
        pass_detail = str(c.get("pass_detail", "Passed"))
        fail_detail = str(c.get("fail_detail", "Failed"))
        skip_if_no_headings = bool(c.get("skip_if_no_headings", False))

        if skip_if_no_headings and not has_headings:
            continue

        if check_type == "regex_absent":
            has_match = bool(re.search(pattern, all_source))
            passed = not has_match
            detail = pass_detail if passed else fail_detail
        elif check_type == "regex_present":
            has_match = bool(re.search(pattern, all_source))
            passed = has_match
            detail = pass_detail if passed else fail_detail
        elif check_type == "regex_count_zero":
            count = len(re.findall(pattern, all_source))
            passed = count == 0
            detail = pass_detail if passed else fail_detail.format(count=count)
        elif check_type == "heading_has_h1":
            passed = 1 in heading_levels
            detail = pass_detail if passed else fail_detail
        elif check_type == "heading_no_skip":
            skips = any(
                heading_levels[i + 1] - heading_levels[i] > 1
                for i in range(len(heading_levels) - 1)
                if heading_levels[i + 1] > heading_levels[i]
            )
            passed = not skips
            detail = pass_detail if passed else fail_detail
        else:
            passed = False
            detail = f"Unknown source scanner check type: {check_type}"

        results.append(CheckResult(
            check_id=check_id,
            category=category,
            passed=passed,
            detail=detail,
            score=1.0 if passed else 0.0,
        ))

    return results


def _extract_heading_levels(all_source: str) -> Tuple[List[int], bool]:
    headings = re.findall(r"<(h[1-6])\b", all_source)
    if not headings:
        return [], False
    levels = [int(h[1]) for h in headings]
    return levels, True
