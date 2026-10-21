from pydantic import BaseModel, Field, model_validator


class ChatRequest(BaseModel):
    """
    Request payload for chat endpoint.
    Supports 'question' (master prompt requirement) as well as 'message' (frontend UI).
    """
    question: str | None = Field(None, description="The user question regarding AAU documents.")
    message: str | None = Field(None, description="Alias for question for frontend compatibility.")
    roles: list[str] = Field(default_factory=lambda: ["public"], description="User access roles.")

    @model_validator(mode="after")
    def validate_content(self) -> "ChatRequest":
        q = (self.question or self.message or "").strip()
        if not q:
            raise ValueError("Either 'question' or 'message' must be provided and non-empty.")
        return self

    @property
    def query(self) -> str:
        return (self.question or self.message or "").strip()


class SourceCitation(BaseModel):
    """Source reference for retrieved evidence."""
    document_id: str = Field(..., description="Unique ID of the document.")
    document_name: str = Field(..., description="Original filename of the document.")
    page: int | None = Field(None, description="Page number if available.")
    
    # Backward compatibility with frontend
    source: str = Field("", description="Display source name.")
    score: float = Field(0.0, description="Relevance score.")
    content: str = Field("", description="Snippet content from source chunk.")


class ChatResponse(BaseModel):
    """
    Response payload for chat endpoint.
    Strictly reports groundedness and refusal status.
    """
    answer: str = Field(..., description="The grounded answer or refusal message.")
    sources: list[SourceCitation] = Field(default_factory=list, description="Validated citations.")
    grounded: bool = Field(..., description="True if the answer is strictly grounded in retrieved evidence.")
    refusal: bool = Field(..., description="True if evidence was insufficient or query was refused.")
    found: bool = Field(True, description="Frontend compatibility flag; equals (not refusal and grounded).")
