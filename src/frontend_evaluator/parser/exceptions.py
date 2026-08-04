"""Custom exceptions for the artifact parser module."""


class ParseError(Exception):
    """Raised when artifact parsing fails."""

    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)
