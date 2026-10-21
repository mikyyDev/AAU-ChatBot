import logging
import re

from pydantic import BaseModel, Field

from app.core.config import settings
from app.services.retriever import RetrievedChunk

logger = logging.getLogger(__name__)

# Common English question words and filler stop-words
STOP_WORDS: set[str] = {
    "a", "an", "the", "in", "on", "at", "by", "for", "with", "about",
    "against", "between", "into", "through", "during", "before", "after",
    "above", "below", "to", "from", "up", "down", "is", "are", "was",
    "were", "be", "been", "being", "have", "has", "had", "do", "does",
    "did", "can", "could", "should", "would", "will", "shall", "may",
    "might", "must", "what", "which", "who", "whom", "this", "that",
    "these", "those", "am", "it", "its", "tell", "me", "explain", "how",
    "why", "when", "where", "please", "give", "provide", "any",
    "and", "or", "also", "as", "well"
}

TEMPORAL_INDICATORS: set[str] = {
    "january", "february", "march", "april", "may", "june", "july", "august",
    "september", "october", "november", "december", "date", "deadline",
    "semester", "academic year", "schedule", "calendar", "period", "monday",
    "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"
}

NUMERIC_INDICATORS: set[str] = {
    "total", "number", "count", "fee", "cost", "amount", "percent", "%",
    "gpa", "credits", "hours", "rate", "statistics"
}


class EvidenceEvaluationResult(BaseModel):
    is_sufficient: bool = Field(..., description="Whether evidence is sufficient to answer the question.")
    is_partial: bool = Field(False, description="Whether evidence supports only part of the question.")
    confidence_score: float = Field(0.0, description="Overall confidence score for groundedness.")
    supported_aspects: list[str] = Field(default_factory=list)
    missing_aspects: list[str] = Field(default_factory=list)
    relevant_chunks: list[RetrievedChunk] = Field(default_factory=list)
    reason: str = Field("", description="Diagnostic explanation of the evidence determination.")


def get_stem(word: str) -> str:
    """Extract root prefix for robust inflectional matching."""
    w = word.lower()
    for suffix in ["ments", "ment", "ing", "tion", "ed", "es", "s"]:
        if len(w) > len(suffix) + 3 and w.endswith(suffix):
            return w[:-len(suffix)]
    return w


def extract_keywords(query: str) -> list[str]:
    """Extract substantive keywords from user query, stripping punctuation and stop-words."""
    tokens = re.findall(r"\b[a-zA-Z0-9_\-']+\b", query.lower())
    keywords = [t for t in tokens if t not in STOP_WORDS and len(t) > 2]
    return keywords


class EvidenceChecker:
    """
    Independent evidence validation service.
    Ensures that high retrieval similarity alone does not grant an answer,
    and rigorously verifies that factual predicates required by the query exist in the context.
    """

    def __init__(self, score_threshold: float | None = None):
        self.score_threshold = score_threshold or settings.MIN_RELEVANCE_SCORE

    def evaluate_evidence(
        self,
        query: str,
        retrieved_chunks: list[RetrievedChunk],
    ) -> EvidenceEvaluationResult:
        # 1. Empty retrieval case
        if not retrieved_chunks:
            return EvidenceEvaluationResult(
                is_sufficient=False,
                confidence_score=0.0,
                reason="No relevant document chunks found in the AAU knowledge base.",
            )

        query_lower = query.lower().strip()
        keywords = extract_keywords(query)

        # 2. Score threshold check
        top_chunk = retrieved_chunks[0]
        if top_chunk.score < self.score_threshold:
            logger.info(f"Top chunk score {top_chunk.score} is below threshold {self.score_threshold}")
            return EvidenceEvaluationResult(
                is_sufficient=False,
                confidence_score=top_chunk.score,
                reason=f"Retrieval confidence ({top_chunk.score}) is below the required threshold ({self.score_threshold}).",
            )

        # Combine text from retrieved chunks for evidence evaluation
        combined_text = " ".join(chunk.content.lower() for chunk in retrieved_chunks)

        # 3. Keyword / Entity overlap check with stemming support
        if not keywords:
            # Short generic query without distinct keywords
            matched_keywords = []
            coverage_ratio = 1.0
        else:
            matched_keywords = [
                kw for kw in keywords
                if kw in combined_text or get_stem(kw) in combined_text
            ]
            coverage_ratio = len(matched_keywords) / len(keywords)

        missing_aspects = [kw for kw in keywords if kw not in matched_keywords]

        # Check for partial evidence
        is_partial = False
        if len(matched_keywords) >= 2 and len(missing_aspects) >= 1:
            is_partial = True

        # If keyword coverage is too low, the retrieved chunks do not address the question
        if not is_partial and coverage_ratio < 0.35 and len(matched_keywords) < 2:
            return EvidenceEvaluationResult(
                is_sufficient=False,
                confidence_score=round(top_chunk.score * coverage_ratio, 3),
                supported_aspects=matched_keywords,
                missing_aspects=missing_aspects,
                reason="Retrieved documents do not contain the core entities or concepts requested.",
            )

        # 4. Factual Predicate Check
        # Check if the specific predicate requested (e.g. deadline, date, number, requirement) exists in context
        is_when_query = any(w in query_lower for w in ["when", "deadline", "date", "schedule"])
        is_count_query = any(w in query_lower for w in ["how many", "number of", "fee", "cost", "statistic"])

        has_temporal_evidence = any(t in combined_text for t in TEMPORAL_INDICATORS) or bool(re.search(r"\b20\d{2}\b", combined_text))
        has_numeric_evidence = bool(re.search(r"\b\d+\b", combined_text)) or any(n in combined_text for n in NUMERIC_INDICATORS)

        if is_when_query and not has_temporal_evidence:
            if is_partial:
                missing_aspects.append("date / deadline")
            else:
                logger.info("Question asks 'when/deadline' but context contains no dates or deadlines.")
                return EvidenceEvaluationResult(
                    is_sufficient=False,
                    confidence_score=0.2,
                    reason="The context mentions the topic but does not contain date, deadline, or schedule information.",
                    supported_aspects=matched_keywords,
                    missing_aspects=["date / deadline"],
                    relevant_chunks=retrieved_chunks,
                )

        if is_count_query and not has_numeric_evidence:
            if is_partial:
                missing_aspects.append("numerical or fee details")
            else:
                logger.info("Question asks for counts/fees/numbers but context has no numeric data.")
                return EvidenceEvaluationResult(
                    is_sufficient=False,
                    confidence_score=0.2,
                    reason="The context mentions the topic but lacks the specific numbers, fees, or statistics requested.",
                    supported_aspects=matched_keywords,
                    missing_aspects=["numerical or fee details"],
                    relevant_chunks=retrieved_chunks,
                )

        # 5. Check for Partial Evidence
        # If question contains multiple clauses/conjunctions ("and", "also", "as well as")
        is_partial = False
        missing_aspects = [kw for kw in keywords if kw not in matched_keywords]
        if missing_aspects and len(missing_aspects) >= 2 and len(matched_keywords) >= 1:
            is_partial = True

        # Confidence calculation
        confidence = round(0.5 * top_chunk.score + 0.5 * coverage_ratio, 3)

        # Filter chunks that actually contain at least one matched keyword
        filtered_chunks = [
            chunk for chunk in retrieved_chunks
            if any(kw in chunk.content.lower() for kw in matched_keywords) or not keywords
        ]
        if not filtered_chunks:
            filtered_chunks = retrieved_chunks[:2]

        return EvidenceEvaluationResult(
            is_sufficient=True,
            is_partial=is_partial,
            confidence_score=confidence,
            supported_aspects=matched_keywords,
            missing_aspects=missing_aspects,
            relevant_chunks=filtered_chunks,
            reason="Sufficient grounding evidence identified in context.",
        )


def get_evidence_checker() -> EvidenceChecker:
    return EvidenceChecker()
