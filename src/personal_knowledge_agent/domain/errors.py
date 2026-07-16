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
