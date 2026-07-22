class ModelUnavailableError(RuntimeError):
    """Raised when the prediction model is not yet available."""


class InvalidImageError(ValueError):
    """Raised when an uploaded file is invalid for prediction."""
