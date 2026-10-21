import re
from pathlib import Path

from fastapi import HTTPException, status
from pydantic import BaseModel, Field

from app.core.config import settings


class UserContext(BaseModel):
    user_id: str = "anonymous_user"
    roles: list[str] = Field(default_factory=lambda: ["public"])


def sanitize_filename(filename: str) -> str:
    """
    Sanitize uploaded filename to prevent directory traversal and injection.
    Extracts the pure basename, strips path separators, null bytes, and unsafe characters.
    """
    if not filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Filename cannot be empty.",
        )
    
    # Strip path elements
    clean_name = Path(filename).name
    # Remove null bytes and non-printable characters
    clean_name = clean_name.replace("\x00", "").strip()
    # Replace unsafe characters but keep alphanumeric, dash, underscore, dot
    clean_name = re.sub(r"[^\w\s\.-]", "_", clean_name)
    clean_name = re.sub(r"\s+", "_", clean_name)

    if not clean_name or clean_name in {".", ".."}:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid filename provided.",
        )
    return clean_name


def validate_file_extension(filename: str, allowed_extensions: set[str] | None = None) -> str:
    """Validate that the file has an allowed extension."""
    allowed = allowed_extensions or settings.ALLOWED_EXTENSIONS
    suffix = Path(filename).suffix.lower()
    if suffix not in allowed:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type '{suffix}'. Allowed types: {sorted(allowed)}",
        )
    return suffix


def validate_file_size(size_bytes: int, max_bytes: int | None = None) -> None:
    """Validate file size does not exceed limits and is not zero bytes."""
    limit = max_bytes or settings.MAX_UPLOAD_SIZE_BYTES
    if size_bytes <= 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty (0 bytes).",
        )
    if size_bytes > limit:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File size exceeds maximum permitted limit of {limit // (1024 * 1024)} MB.",
        )


def check_document_access(document_roles: list[str], user_roles: list[str]) -> bool:
    """
    Determine if a user with given roles has permission to access a document.
    'admin' role has access to everything.
    Documents with 'public' role are accessible by everyone.
    """
    if "admin" in user_roles:
        return True
    
    doc_roles_set = set(document_roles)
    if "public" in doc_roles_set:
        return True
    
    # Check role intersection
    return bool(doc_roles_set.intersection(set(user_roles)))


def sanitize_document_text_for_prompt(text: str) -> str:
    """
    Sanitize document text before sending to LLM context to prevent prompt injection.
    Neutralizes system-prompt-override attempts and formats cleanly.
    """
    if not text:
        return ""
    # Strip excessive repeated symbols or adversarial directives
    sanitized = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)
    # Neutralize fake system prompts or control token attempts
    sanitized = re.sub(r"(?i)(system\s*prompt:|system\s*instruction:|ignore\s+all\s+previous\s+instructions)", "[FILTERED INSTRUCTION]", sanitized)
    return sanitized.strip()
