import logging
import re

from groq import APIConnectionError, APIStatusError, Groq, RateLimitError

from app.core.config import settings
from app.core.security import sanitize_document_text_for_prompt
from app.prompts.system_prompt import PRESCRIBED_REFUSAL_MESSAGE, STRICT_SYSTEM_PROMPT
from app.schemas.chat import SourceCitation
from app.services.citation_validator import CitationValidator, get_citation_validator
from app.services.evidence_checker import EvidenceEvaluationResult
from app.services.retriever import RetrievedChunk

logger = logging.getLogger(__name__)

GREETING_PATTERNS = [
    r"^(hi|hello|hey|greetings|good\s+morning|good\s+afternoon|good\s+evening)\b",
    r"^(thanks|thank\s+you|who\s+are\s+you)\b",
]


class AnswerGenerator:
    def __init__(self, citation_validator: CitationValidator | None = None):
        self.citation_validator = citation_validator or get_citation_validator()
        self._groq_client = None
        if settings.GROQ_API_KEY:
            try:
                self._groq_client = Groq(api_key=settings.GROQ_API_KEY)
            except (ValueError, TypeError):
                logger.exception("Could not initialize Groq client")

    def is_greeting(self, text: str) -> bool:
        cleaned = text.strip().lower()
        if len(cleaned.split()) <= 4:
            for pat in GREETING_PATTERNS:
                if re.search(pat, cleaned):
                    return True
        return False

    def build_context_block(self, chunks: list[RetrievedChunk]) -> str:
        """Format retrieved chunks with defensive untrusted data boundaries."""
        blocks = []
        for c in chunks:
            sanitized = sanitize_document_text_for_prompt(c.content)
            page_info = f", Page: {c.page}" if c.page is not None else ""
            blocks.append(f"[Source: {c.document_name}{page_info}]\n{sanitized}")
        return (
            "=== BEGIN RETRIEVED AAU DOCUMENT EVIDENCE ===\n"
            + "\n\n---\n\n".join(blocks)
            + "\n=== END RETRIEVED AAU DOCUMENT EVIDENCE ==="
        )

    def generate_extractive_fallback(
        self, query: str, chunks: list[RetrievedChunk], is_partial: bool, missing_aspects: list[str]
    ) -> str:
        """
        Deterministic, offline extraction when external LLM APIs are not configured.
        Extracts sentences containing relevant keywords from the authorized chunks.
        """
        keywords = [w.lower() for w in re.findall(r"\b\w{3,}\b", query) if w.lower() not in {"what", "when", "where", "which", "how", "who"}]
        relevant_sentences = []
        seen = set()

        for c in chunks:
            # Split into sentences
            sentences = re.split(r"(?<=[.!?])\s+", c.content)
            for s in sentences:
                s_clean = s.strip()
                if not s_clean or s_clean in seen:
                    continue
                s_lower = s_clean.lower()
                matched = sum(1 for kw in keywords if kw in s_lower)
                if matched >= 1:
                    seen.add(s_clean)
                    page_ref = f" (Source: {c.document_name}, Page: {c.page})" if c.page else f" (Source: {c.document_name})"
                    relevant_sentences.append(f"{s_clean}{page_ref}")
                    if len(relevant_sentences) >= 3:
                        break

        if not relevant_sentences:
            return PRESCRIBED_REFUSAL_MESSAGE

        answer = " ".join(relevant_sentences)
        if is_partial and missing_aspects:
            missing_str = ", ".join(missing_aspects)
            answer += f"\n\nNote: Information regarding '{missing_str}' was not found in the available documents."

        return answer

    def generate_answer(
        self,
        query: str,
        evidence: EvidenceEvaluationResult,
    ) -> tuple[str, list[SourceCitation], bool, bool]:
        """
        Generate answer and return (answer_text, citations, is_grounded, is_refusal).
        """
        # 1. Conversational greetings
        if self.is_greeting(query):
            greeting_resp = (
                "Hello! I am AAU Chatbot, an AI knowledge assistant for Addis Ababa University. "
                "How can I assist you with information from AAU documents today?"
            )
            return greeting_resp, [], True, False

        # 2. Strict refusal if evidence is insufficient
        if not evidence.is_sufficient or not evidence.relevant_chunks:
            return PRESCRIBED_REFUSAL_MESSAGE, [], False, True

        context_block = self.build_context_block(evidence.relevant_chunks)

        user_prompt = (
            f"User Question: {query}\n\n"
            f"{context_block}\n\n"
            "Answer the question using ONLY the evidence supplied above. "
            "If the evidence does not sufficiently answer the question, return the prescribed refusal. "
            "Cite the source document and page number whenever available."
        )

        answer_text: str = ""

        # 3. Call LLM (Groq or OpenAI) if configured
        if self._groq_client:
            models_to_try = [settings.GROQ_MODEL, "openai/gpt-oss-120b", "qwen/qwen3.8-27b"]
            seen_models = set()
            models_to_try = [m for m in models_to_try if not (m in seen_models or seen_models.add(m))]

            for mod in models_to_try:
                try:
                    response = self._groq_client.chat.completions.create(
                        model=mod,
                        messages=[
                            {"role": "system", "content": STRICT_SYSTEM_PROMPT},
                            {"role": "user", "content": user_prompt},
                        ],
                        temperature=settings.LLM_TEMPERATURE,
                        timeout=20.0,
                    )
                    answer_text = response.choices[0].message.content or ""
                    if answer_text:
                        break
                except (APIConnectionError, APIStatusError, RateLimitError, TimeoutError) as error:
                    logger.warning("Groq generation with model %s failed: %s", mod, error)

            if not answer_text:
                logger.info("Falling back to deterministic extractive engine.")
                answer_text = self.generate_extractive_fallback(
                    query, evidence.relevant_chunks, evidence.is_partial, evidence.missing_aspects
                )
        else:
            # Deterministic fallback engine
            answer_text = self.generate_extractive_fallback(
                query, evidence.relevant_chunks, evidence.is_partial, evidence.missing_aspects
            )

        # 4. Check if answer is a refusal
        answer_clean = answer_text.strip()
        refusal_phrases = [
            PRESCRIBED_REFUSAL_MESSAGE.lower(),
            "couldn't find sufficient information",
            "not enough information in the provided context",
            "cannot find sufficient information",
            "no information about this in the available",
        ]
        if any(rp in answer_clean.lower() for rp in refusal_phrases):
            return PRESCRIBED_REFUSAL_MESSAGE, [], False, True

        # 5. Citation validation
        citations, is_grounded = self.citation_validator.validate_citations(
            answer_clean, evidence.relevant_chunks
        )

        # If model hallucinated an answer without citation grounding in retrieved chunks, refuse
        if not is_grounded:
            logger.warning("Generated answer lacked grounding in retrieved chunks. Triggering refusal.")
            return PRESCRIBED_REFUSAL_MESSAGE, [], False, True

        return answer_clean, citations, True, False


def get_answer_generator() -> AnswerGenerator:
    return AnswerGenerator()
