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


class MemoryFeatureError(KnowledgeError):
    """Base class for user-facing long-term memory errors."""


class InvalidMemoryError(MemoryFeatureError):
    """Raised when a memory operation receives invalid input."""


class UnsafeMemoryError(MemoryFeatureError):
    """Raised when memory content violates the local safety policy."""


class MemoryConflictError(MemoryFeatureError):
    """Raised when a memory conflicts with an active stable slot."""


class MemoryNotFoundError(MemoryFeatureError):
    """Raised when a requested memory record does not exist."""


class MemoryStorageError(MemoryFeatureError):
    """Raised when persistent memory storage cannot complete an operation."""


class InvalidMemorySchemaError(MemoryFeatureError):
    """Raised when extracted output violates the memory schema."""


class GraphFeatureError(KnowledgeError):
    """Base class for user-facing knowledge-graph errors."""


class InvalidGraphError(GraphFeatureError):
    """Raised when a graph operation receives invalid input."""


class GraphSourceNotFoundError(GraphFeatureError):
    """Raised when the requested source is not in the knowledge index."""


class GraphEmptyError(GraphFeatureError):
    """Raised when the graph database contains no entities."""


class GraphEntityNotFoundError(GraphFeatureError):
    """Raised when a requested entity does not exist in the graph."""


class GraphStorageError(GraphFeatureError):
    """Raised when the persistent graph store cannot complete an operation."""


class InvalidGraphSchemaError(GraphFeatureError):
    """Raised when graph extraction output violates the expected schema."""


class EvaluationFeatureError(KnowledgeError):
    """Base class for user-facing RAG evaluation errors."""


class InvalidEvaluationError(EvaluationFeatureError):
    """Raised when an evaluation input or result is invalid."""


class EvaluationStorageError(EvaluationFeatureError):
    """Raised when an evaluation artifact cannot be persisted safely."""


class InvalidJudgeSchemaError(EvaluationFeatureError):
    """Raised when LLM Judge output violates the expected JSON contract."""
