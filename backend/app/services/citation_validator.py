import logging
import re
from typing import List, Tuple
from app.schemas.chat import SourceCitation
from app.services.retriever import RetrievedChunk

logger = logging.getLogger(__name__)


class CitationValidator:
    """
    Validates that citations attached to answers correspond strictly to
    chunks that were actually retrieved and contain verifiable content.
    Prevents citation fabrication and hallucinated page numbers.
    """

    def validate_citations(
        self,
        answer_text: str,
        retrieved_chunks: List[RetrievedChunk],
    ) -> Tuple[List[SourceCitation], bool]:
        """
        Extract and validate citations from the answer and retrieved chunks.
        Returns (validated_citations, is_grounded).
        """
        if not retrieved_chunks:
            return [], False

        # Map of valid document names and IDs in retrieved context
        doc_map = {c.document_name.lower(): c for c in retrieved_chunks}
        id_map = {c.document_id: c for c in retrieved_chunks}

        valid_citations: List[SourceCitation] = []
        seen_keys = set()

        # Check for inline citations mentioned in answer (e.g. "[Source: filename.pdf, Page: 2]" or "【filename.docx】")
        inline_refs = re.findall(
            r"[\[\(\u3010](?:Source:\s*)?([a-zA-Z0-9_\-\.]+)(?:,\s*Page:\s*(\d+))?[\]\)\u3011]",
            answer_text,
            flags=re.IGNORECASE,
        )

        for ref_name, ref_page_str in inline_refs:
            ref_name_lower = ref_name.lower()
            matching_chunk = doc_map.get(ref_name_lower)
            if not matching_chunk:
                # Document name not in retrieved chunks — hallucination!
                logger.warning(f"Detected hallucinated document citation in answer: {ref_name}")
                continue

            claimed_page = int(ref_page_str) if ref_page_str else None
            # Validate page number if claimed
            final_page = None
            if claimed_page is not None:
                if matching_chunk.page is not None and matching_chunk.page == claimed_page:
                    final_page = claimed_page
                elif matching_chunk.page is None:
                    # Document has no page metadata — omit rather than invent
                    final_page = None
                else:
                    final_page = matching_chunk.page
            else:
                final_page = matching_chunk.page

            key = (matching_chunk.document_id, final_page)
            if key not in seen_keys:
                seen_keys.add(key)
                valid_citations.append(
                    SourceCitation(
                        document_id=matching_chunk.document_id,
                        document_name=matching_chunk.document_name,
                        page=final_page,
                        source=matching_chunk.document_name,
                        score=matching_chunk.score,
                        content=matching_chunk.content[:200] + ("..." if len(matching_chunk.content) > 200 else ""),
                    )
                )

        # If no inline citations were explicitly parsed, build citations from the chunks
        # whose vocabulary was actually utilized in the generated answer
        if not valid_citations:
            answer_words = set(re.findall(r"\b[a-zA-Z0-9_\-]{4,}\b", answer_text.lower()))
            for chunk in retrieved_chunks:
                chunk_words = set(re.findall(r"\b[a-zA-Z0-9_\-]{4,}\b", chunk.content.lower()))
                overlap = answer_words.intersection(chunk_words)
                
                # If chunk content shares substantial terminology with answer, it supported the claim
                if len(overlap) >= 3:
                    key = (chunk.document_id, chunk.page)
                    if key not in seen_keys:
                        seen_keys.add(key)
                        valid_citations.append(
                            SourceCitation(
                                document_id=chunk.document_id,
                                document_name=chunk.document_name,
                                page=chunk.page,
                                source=chunk.document_name,
                                score=chunk.score,
                                content=chunk.content[:200] + ("..." if len(chunk.content) > 200 else ""),
                            )
                        )

        # Grounding check: answer must be supported by at least one valid citation
        is_grounded = len(valid_citations) > 0
        return valid_citations, is_grounded


def get_citation_validator() -> CitationValidator:
    return CitationValidator()
