"""Expected errors raised by knowledge operations."""


class KnowledgeError(Exception):
    """Base class for user-facing knowledge errors."""


class UnsupportedDocumentError(KnowledgeError):
    """Raised when a document format is unsupported."""


class DocumentReadError(KnowledgeError):
    """Raised when a document cannot be read safely."""


class EmptyDocumentError(KnowledgeError):
    """Raised when a document has no indexable content."""


class InvalidQueryError(KnowledgeError):
    """Raised when a search query is invalid."""


class KnowledgeBaseEmptyError(KnowledgeError):
    """Raised when search is attempted before ingestion."""


class IndexStorageError(KnowledgeError):
    """Raised when the persistent index cannot be used."""


class EmbeddingError(KnowledgeError):
    """Raised when local embedding generation or validation fails."""


class ConfigurationError(KnowledgeError):
    """Raised when required application configuration is missing or invalid."""


class LanguageModelError(KnowledgeError):
    """Raised when a language-model provider cannot produce a usable response."""


class ModelOutputTruncatedError(LanguageModelError):
    """Raised when a provider stops before completing structured output."""


class StructuredOutputError(KnowledgeError):
    """Raised when model output violates the grounded-answer contract."""


class InvalidAnswerSchemaError(StructuredOutputError):
    """Raised when model output is not the expected JSON object shape."""


class InvalidRouteSchemaError(StructuredOutputError):
    """Raised when model output is not the expected route JSON shape."""


class InvalidCitationError(StructuredOutputError):
    """Raised when model citations do not match the supplied evidence."""
