import math
import re
from collections import Counter
from dataclasses import dataclass

from .models import Evidence, SearchResult

STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "do", "for", "from", "has",
    "have", "how", "i", "if", "in", "is", "it", "its", "my", "of", "on", "or", "our",
    "please", "so", "that", "the", "their", "this", "to", "was", "we", "what", "when",
    "where", "which", "who", "will", "with", "you", "your", "not", "no", "all", "also",
    "than", "then", "there", "they", "them", "must", "should", "may",
}

SUSPICIOUS_PATTERN = re.compile(
    r"ignore (all )?(prior|previous) instructions|reveal (the )?(system prompt|api keys?)",
    re.IGNORECASE,
)
CITATION_PATTERN = re.compile(r"\[([a-z0-9][a-z0-9-]*)\]")


class CitationError(Exception):
    pass


def tokenize(text: str) -> list[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    tokens = []
    for word in words:
        if word in STOP_WORDS or len(word) < 2:
            continue
        if len(word) > 3 and word.endswith("s"):
            word = word[:-1]
        tokens.append(word)
    return tokens


def is_suspicious(text: str) -> bool:
    return SUSPICIOUS_PATTERN.search(text) is not None


def strip_suspicious(text: str) -> tuple[str, int]:
    sentences = re.split(r"(?<=[.!?])\s+", text)
    kept = [s for s in sentences if not is_suspicious(s)]
    return " ".join(kept), len(sentences) - len(kept)


@dataclass
class Chunk:
    source_id: str
    text: str


def build_chunks(policies: dict[str, str]) -> list[Chunk]:
    chunks = []
    for source_id, text in policies.items():
        blocks = [" ".join(b.split()) for b in re.split(r"\n\s*\n", text) if b.strip()]
        title = blocks[0].lstrip("# ").strip() if blocks else ""
        for block in blocks:
            if block.startswith("#") or block.startswith("Source ID:"):
                continue
            chunks.append(Chunk(source_id=source_id, text=f"{title}. {block}"))
    return chunks


class PolicyIndex:
    def __init__(self, policies: dict[str, str]):
        self.chunks = build_chunks(policies)
        counters = [Counter(tokenize(c.text)) for c in self.chunks]
        count = len(counters)
        document_frequency: Counter[str] = Counter()
        for counter in counters:
            document_frequency.update(counter.keys())
        self._count = count
        self._idf = {
            term: math.log((1 + count) / (1 + df)) + 1 for term, df in document_frequency.items()
        }
        self._vectors = [self._weigh(counter) for counter in counters]

    def _weigh(self, counter: Counter[str]) -> dict[str, float]:
        # Unknown words get the highest weight so off-topic questions score low.
        unknown = math.log(1 + self._count) + 1
        return {t: (1 + math.log(n)) * self._idf.get(t, unknown) for t, n in counter.items()}

    @staticmethod
    def _cosine(left: dict[str, float], right: dict[str, float]) -> float:
        dot = sum(w * right.get(t, 0.0) for t, w in left.items())
        norm = math.sqrt(sum(w * w for w in left.values())) * math.sqrt(
            sum(w * w for w in right.values())
        )
        return dot / norm if norm else 0.0

    def search(self, query: str, top_k: int, min_score: float) -> SearchResult:
        query_vector = self._weigh(Counter(tokenize(query)))
        best_by_source: dict[str, Evidence] = {}
        best_score = 0.0
        removed = 0
        for chunk, vector in zip(self.chunks, self._vectors):
            score = self._cosine(query_vector, vector)
            if is_suspicious(chunk.text):
                if score >= min_score:
                    removed += 1
                continue
            best_score = max(best_score, score)
            current = best_by_source.get(chunk.source_id)
            if current is None or score > current.score:
                best_by_source[chunk.source_id] = Evidence(
                    source_id=chunk.source_id, text=chunk.text, score=round(score, 4)
                )
        accepted = [e for e in best_by_source.values() if e.score >= min_score]
        accepted.sort(key=lambda e: e.score, reverse=True)
        return SearchResult(
            query=query,
            evidence=accepted[:top_k],
            best_score=round(best_score, 4),
            removed_suspicious=removed,
        )


def extract_citations(text: str) -> list[str]:
    seen: list[str] = []
    for source_id in CITATION_PATTERN.findall(text):
        if source_id not in seen:
            seen.append(source_id)
    return seen


def validate_citations(text: str, allowed: set[str]) -> list[str]:
    citations = extract_citations(text)
    unknown = [c for c in citations if c not in allowed]
    if unknown:
        raise CitationError(f"citations not found in accepted evidence: {unknown}")
    return citations
