import re
from collections import Counter

SECTION_PATTERNS = {
    "objectives": r"\b(objective|aim|research question|hypothes[ie]s?)\b",
    "methodology": r"\b(methodology|methods?|study design|experimental design|sampling|participants)\b",
    "evaluation": r"\b(evaluate|evaluation|metric|measure|validation|baseline|benchmark)\b",
    "timeline": r"\b(timeline|milestone|work package|month \d+|year \d+)\b",
    "ethics": r"\b(ethic|consent|privacy|confidential|data protection|irb|institutional review)\b",
    "risk": r"\b(risk|limitation|mitigation|contingency)\b",
    "impact": r"\b(impact|outcome|deliverable|disseminat|knowledge transfer)\b",
    "reproducibility": r"\b(open data|open source|reproducib|replicat|preregister|version control)\b",
}
VAGUE_TERMS = {
    "innovative", "groundbreaking", "revolutionary", "significant", "various",
    "effective", "efficient", "advanced", "novel", "robust"
}

def word_tokens(text):
    return re.findall(r"\b[\w'-]+\b", (text or "").lower())

def quality_features(text):
    text = text or ""
    tokens = word_tokens(text)
    words = len(tokens)
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s]
    sentences_n = max(len(sentences), 1)
    avg_sentence_words = words / sentences_n
    unique_ratio = len(set(tokens)) / max(words, 1)
    question_mark_count = text.count("?")
    vague_count = sum(tokens.count(term) for term in VAGUE_TERMS)
    section_hits = {name: int(bool(re.search(pattern, text, re.I))) for name, pattern in SECTION_PATTERNS.items()}
    section_coverage = sum(section_hits.values()) / len(section_hits)
    # Transparent heuristic indicators, not validated measurements.
    clarity = 100 * (
        0.45 * min(unique_ratio / 0.55, 1.0) +
        0.35 * max(0.0, 1.0 - abs(avg_sentence_words - 22) / 45) +
        0.20 * max(0.0, 1.0 - vague_count / max(words * 0.03, 1))
    )
    clarity = max(0.0, min(100.0, clarity))
    methodology = 100 * (
        0.25 * section_hits["objectives"] +
        0.30 * section_hits["methodology"] +
        0.20 * section_hits["evaluation"] +
        0.10 * section_hits["timeline"] +
        0.05 * section_hits["ethics"] +
        0.10 * section_hits["risk"]
    )
    return {
        "word_count": words,
        "sentence_count": len(sentences),
        "avg_sentence_words": round(avg_sentence_words, 2),
        "lexical_diversity": round(unique_ratio, 3),
        "vague_term_count": vague_count,
        "clarity_heuristic": round(clarity, 1),
        "methodology_completeness_heuristic": round(methodology, 1),
        "section_coverage_heuristic": round(section_coverage * 100, 1),
        **{f"section_{k}": v for k, v in section_hits.items()},
    }

def alignment_score(proposal, call_text):
    """TF-IDF cosine similarity baseline; not a validated measure of funder fit."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    proposal, call_text = (proposal or "").strip(), (call_text or "").strip()
    if not proposal or not call_text:
        return None
    vec = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", max_features=20000)
    matrix = vec.fit_transform([proposal, call_text])
    return round(float(cosine_similarity(matrix[0:1], matrix[1:2])[0, 0] * 100), 1)
