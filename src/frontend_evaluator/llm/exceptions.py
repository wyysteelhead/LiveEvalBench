"""LLM provider exceptions."""


class LLMProviderError(Exception):
    """Base exception for LLM provider errors."""
    pass


class UnsupportedProviderError(LLMProviderError):
    """Raised when an unsupported provider is requested."""
    pass


class InvalidModelError(LLMProviderError):
    """Raised when an invalid model is specified for a provider."""
    pass


class ProviderAPIError(LLMProviderError):
    """Raised when a provider API call fails."""
    pass


def handle_provider_error(error: Exception, provider: str) -> str:
    """
    Convert provider-specific errors into user-friendly messages.

    Args:
        error: The exception that occurred
        provider: The provider name (anthropic, openai, google)

    Returns:
        User-friendly error message
    """
    error_msg = str(error).lower()

    # API key errors
    if "api key" in error_msg or "authentication" in error_msg or "unauthorized" in error_msg:
        return f"Invalid or missing API key for {provider}. Please check your .env file."

    # Rate limit errors
    if "rate limit" in error_msg or "quota" in error_msg:
        return f"Rate limit exceeded for {provider}. Please try again later."

    # Model not found errors
    if "model" in error_msg and ("not found" in error_msg or "does not exist" in error_msg):
        return f"Model not found for {provider}. Please check the model name in your .env file."

    # Generic error
    return f"Error calling {provider} API: {str(error)}"
