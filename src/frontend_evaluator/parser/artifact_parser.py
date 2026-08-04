"""Parser for extracting code artifacts from LLM-generated output."""

import re
from typing import Dict
from .exceptions import ParseError


class ArtifactParser:
    """Parses code blocks preceded by filename headers.

    Supports the format:
        # path/to/file.ext
        ```lang
        content here
        ```

    or with ## headers:
        ## path/to/file.ext
        ```
        content here
        ```

    Optional text between the header and code block is ignored.
    """

    # Matches: # filename, ## filename, # 文件1: filename, # File 1: filename
    HEADER_PATTERN = re.compile(
        r'^#{1,2}\s+(?:(?:文件|[Ff]ile)\s*\d+\s*[:：]\s*)?(\S+)\s*$',
        re.MULTILINE,
    )
    # Matches: ```lang\ncontent\n```
    CODE_BLOCK_PATTERN = re.compile(r'```[^\n]*\n(.*?)```', re.DOTALL)
    # Optionally strips a leading <thinking>...</thinking> block.
    LEADING_THINKING_PATTERN = re.compile(
        r'^\s*<thinking>.*?</thinking>\s*',
        re.DOTALL | re.IGNORECASE,
    )

    @classmethod
    def _strip_leading_thinking(cls, text: str) -> str:
        """Strip a leading thinking block so it does not affect parsing."""
        if not text.lstrip().lower().startswith("<thinking>"):
            return text

        match = cls.LEADING_THINKING_PATTERN.match(text)
        if match:
            return text[match.end():]

        # Fallback for malformed input with missing closing tag.
        return re.sub(r'^\s*<thinking>\s*', '', text, count=1, flags=re.IGNORECASE)

    @classmethod
    def parse(cls, text: str) -> Dict[str, str]:
        """Parse text and extract file artifacts.

        Args:
            text: String containing # filename headers followed by code blocks

        Returns:
            Dictionary mapping filenames to their content

        Raises:
            ParseError: If no valid filename headers with code blocks are found
        """
        if not text or not text.strip():
            raise ParseError("Input text is empty")

        normalized_text = cls._strip_leading_thinking(text)

        headers = [
            (m.start(), m.group(1))
            for m in cls.HEADER_PATTERN.finditer(normalized_text)
        ]
        code_blocks = [
            (m.start(), m.group(1).rstrip())
            for m in cls.CODE_BLOCK_PATTERN.finditer(normalized_text)
        ]

        if not headers:
            raise ParseError(
                "No valid code blocks with filename headers found. "
                "Expected format: # filename.ext"
            )

        artifacts = {}
        for i, (header_pos, filename) in enumerate(headers):
            next_header_pos = (
                headers[i + 1][0] if i + 1 < len(headers) else len(normalized_text)
            )

            for block_pos, content in code_blocks:
                if header_pos < block_pos < next_header_pos:
                    if filename in artifacts:
                        raise ParseError(f"Duplicate filename found: {filename}")
                    artifacts[filename] = content
                    break

        if not artifacts:
            raise ParseError(
                "No valid code blocks with filename headers found. "
                "Expected format: # filename.ext"
            )

        return artifacts
