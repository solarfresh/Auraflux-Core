import re
from typing import List


def estimate_tokens(text: str) -> int:
    """
    Estimates token counts for bilingual (English & CJK) text:
    - CJK characters: 1 Char ≈ 1 Token
    - Non-CJK/English words: 1 Word ≈ 1.3 Tokens
    """
    if not text:
        return 0

    # Matches all CJK characters including punctuation
    cjk_chars = re.findall(r'[\u4e00-\u9fff\u3000-\u303f\uff00-\uffef]', text)
    cjk_count = len(cjk_chars)

    # Removes CJK characters to count remaining English words
    non_cjk_text = re.sub(r'[\u4e00-\u9fff\u3000-\u303f\uff00-\uffef]', ' ', text)
    english_words = non_cjk_text.split()
    english_token_count = int(len(english_words) * 1.3)

    return cjk_count + english_token_count

def split_paragraphs_bilingual(text: str) -> List[str]:
    """
    Splits text into structural paragraphs with bilingual support:
    - Pure English text splits strictly on double newlines (`\\n\\n`), converting single internal newlines to spaces.
    - CJK text converts single newlines after CJK sentence-ending punctuation (`。！？`) into paragraph breaks.
    """
    if not text:
        return []

    # Check for CJK characters
    has_cjk = bool(re.search(r'[\u4e00-\u9fff]', text))

    if has_cjk:
        # Convert CJK sentence-ending punctuation followed by single newlines into double newlines
        text = re.sub(r'([。！？])\n', r'\1\n\n', text)
        raw_paragraphs = re.split(r'\n\s*\n', text)

        cleaned_paragraphs = []
        for p in raw_paragraphs:
            p_clean = p.strip()
            if not p_clean:
                continue
            # Strip internal single newlines inside CJK paragraphs
            p_clean = re.sub(r'\s*\n\s*', '', p_clean)
            cleaned_paragraphs.append(p_clean)
        return cleaned_paragraphs

    else:
        # For Western/English text, split exclusively on double newlines
        raw_paragraphs = re.split(r'\n\s*\n', text)

        cleaned_paragraphs = []
        for p in raw_paragraphs:
            p_clean = p.strip()
            if not p_clean:
                continue
            # Replace single internal newlines with spaces to avoid splitting sentences
            p_clean = re.sub(r'\s*\n\s*', ' ', p_clean)
            cleaned_paragraphs.append(p_clean)
        return cleaned_paragraphs
