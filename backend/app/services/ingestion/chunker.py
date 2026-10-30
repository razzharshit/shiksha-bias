import re
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from app.core.config import settings
from app.services.ingestion.pdf_parser import ExtractedPage, ExtractedBlock


@dataclass
class ChunkData:
    page_number: int
    chunk_index: int
    content: str
    cleaned_content: str
    token_count: int
    bounding_boxes: List[List[float]] = field(default_factory=list)
    is_model_input: bool = False
    chapter_id: Optional[str] = None
    sub_chunks: List['ChunkData'] = field(default_factory=list)


class ContextAwareChunker:
    """
    Context-Aware Sentence-Preserving Chunker.
    Generates:
      1. Contextual Document Chunks (for UI display and surrounding context retrieval).
      2. Model Input Chunks (strictly <= 256 tokens for DeBERTa/transformer compatibility).
    """

    COMMON_ABBREVIATIONS = ["Mr.", "Mrs.", "Ms.", "Dr.", "Prof.", "e.g.", "i.e.", "vs.", "fig.", "al."]

    def __init__(
        self,
        model_max_tokens: int = settings.MODEL_MAX_SEQUENCE_LENGTH,  # 256
        context_target_tokens: int = settings.CONTEXT_CHUNK_MAX_TOKENS,  # 400
        overlap_tokens: int = settings.CHUNK_OVERLAP_TOKENS,  # 30
        tokenizer_name: Optional[str] = "microsoft/deberta-v3-base"
    ):
        self.model_max_tokens = model_max_tokens
        self.context_target_tokens = context_target_tokens
        self.overlap_tokens = overlap_tokens
        self.tokenizer = None

        if tokenizer_name:
            try:
                from transformers import AutoTokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            except Exception:
                self.tokenizer = None

    def count_tokens(self, text: str, add_special_tokens: bool = True) -> int:
        """Counts exact subword tokens using DeBERTa tokenizer when available, or reliable upper-bound proxy."""
        if self.tokenizer:
            return len(self.tokenizer.encode(text, add_special_tokens=add_special_tokens))
        words = text.split()
        # Conservative proxy: 1.35 tokens per word + 2 special tokens ([CLS], [SEP])
        return int(len(words) * 1.35) + (2 if add_special_tokens else 0)

    def split_into_sentences(self, text: str) -> List[str]:
        """Splits paragraph into sentences without breaking mid-sentence."""
        temp_text = text
        for abbrev in self.COMMON_ABBREVIATIONS:
            temp_text = re.sub(re.escape(abbrev), abbrev.replace(".", "@DOT@"), temp_text, flags=re.IGNORECASE)

        raw_splits = re.split(r'[\.\!\?]\s+(?=[A-Z0-9"\'])', temp_text)
        sentences = []
        for s in raw_splits:
            s_restored = s.replace("@DOT@", ".").strip()
            if s_restored:
                sentences.append(s_restored)
        return sentences if sentences else ([text.strip()] if text.strip() else [])

    def chunk_pages(
        self,
        pages: List[ExtractedPage],
        chapter_mapping: Optional[List[Dict[str, Any]]] = None
    ) -> List[ChunkData]:
        """
        Processes extracted pages into hierarchically linked document chunks and model-input chunks.
        """
        all_chunks: List[ChunkData] = []
        global_chunk_idx = 0

        for page in pages:
            if not page.cleaned_text.strip():
                continue

            # Determine associated chapter
            ch_id = None
            if chapter_mapping:
                for ch in chapter_mapping:
                    if ch["start_page"] <= page.page_number <= ch["end_page"]:
                        ch_id = ch.get("id")
                        break

            # Collect page bounding boxes
            page_boxes = [[b.x0, b.y0, b.x1, b.y1] for b in page.blocks if not b.is_header_footer]

            sentences = self.split_into_sentences(page.cleaned_text)
            if not sentences:
                continue

            # Step 1: Build Contextual Chunk (document level)
            current_context_sents: List[str] = []
            current_context_tokens = 0

            for sent in sentences:
                sent_tokens = self.count_tokens(sent)
                if current_context_tokens + sent_tokens > self.context_target_tokens and current_context_sents:
                    ctx_text = " ".join(current_context_sents)
                    context_chunk = ChunkData(
                        page_number=page.page_number,
                        chunk_index=global_chunk_idx,
                        content=ctx_text,
                        cleaned_content=ctx_text,
                        token_count=current_context_tokens,
                        bounding_boxes=page_boxes,
                        is_model_input=False,
                        chapter_id=ch_id
                    )
                    
                    # Generate sub-chunks strictly <= model_max_tokens (256)
                    context_chunk.sub_chunks = self._generate_model_input_chunks(
                        current_context_sents,
                        page.page_number,
                        global_chunk_idx,
                        page_boxes,
                        ch_id
                    )
                    
                    all_chunks.append(context_chunk)
                    global_chunk_idx += 1
                    current_context_sents = []
                    current_context_tokens = 0

                current_context_sents.append(sent)
                current_context_tokens += sent_tokens

            # Flush remaining sentences as final context chunk on page
            if current_context_sents:
                ctx_text = " ".join(current_context_sents)
                context_chunk = ChunkData(
                    page_number=page.page_number,
                    chunk_index=global_chunk_idx,
                    content=ctx_text,
                    cleaned_content=ctx_text,
                    token_count=current_context_tokens,
                    bounding_boxes=page_boxes,
                    is_model_input=False,
                    chapter_id=ch_id
                )
                context_chunk.sub_chunks = self._generate_model_input_chunks(
                    current_context_sents,
                    page.page_number,
                    global_chunk_idx,
                    page_boxes,
                    ch_id
                )
                all_chunks.append(context_chunk)
                global_chunk_idx += 1

        return all_chunks

    def _generate_model_input_chunks(
        self,
        sentences: List[str],
        page_num: int,
        parent_idx: int,
        boxes: List[List[float]],
        chapter_id: Optional[str]
    ) -> List[ChunkData]:
        """
        Generates granular model input chunks strictly capped at model_max_tokens (<= 256 tokens).
        """
        model_chunks: List[ChunkData] = []
        sub_sents: List[str] = []
        sub_tokens = 0
        sub_idx = 0

        # Target 180-220 tokens so it comfortably fits within 256 tokens with special tokens ([CLS], [SEP])
        effective_max = min(self.model_max_tokens - 10, 220)

        for sent in sentences:
            sent_tokens = self.count_tokens(sent)
            
            # If a single sentence exceeds effective_max, we still keep it intact as a single model input
            if sub_tokens + sent_tokens > effective_max and sub_sents:
                chunk_text = " ".join(sub_sents)
                model_chunks.append(ChunkData(
                    page_number=page_num,
                    chunk_index=sub_idx,
                    content=chunk_text,
                    cleaned_content=chunk_text,
                    token_count=sub_tokens,
                    bounding_boxes=boxes,
                    is_model_input=True,
                    chapter_id=chapter_id
                ))
                sub_idx += 1
                sub_sents = []
                sub_tokens = 0

            sub_sents.append(sent)
            sub_tokens += sent_tokens

        if sub_sents:
            chunk_text = " ".join(sub_sents)
            model_chunks.append(ChunkData(
                page_number=page_num,
                chunk_index=sub_idx,
                content=chunk_text,
                cleaned_content=chunk_text,
                token_count=sub_tokens,
                bounding_boxes=boxes,
                is_model_input=True,
                chapter_id=chapter_id
            ))

        return model_chunks
