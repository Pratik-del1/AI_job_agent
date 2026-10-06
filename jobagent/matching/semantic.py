"""Semantic similarity between the candidate and a job, bounded 0-1.

The embedding model is a setting. Job text is cleaned and chunked before it
gets here, so nothing is silently truncated by the model's token limit.
"""

from typing import Optional, Protocol

import numpy as np

from jobagent.resume.schema import ResumeProfile


class Embedder(Protocol):
    def encode(self, texts: list[str]) -> np.ndarray:
        """Unit-length vectors, one row per text."""


class SentenceTransformerEmbedder:
    """Lazy wrapper around a sentence-transformers model."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = None
        self._cache: dict[str, np.ndarray] = {}

    def encode(self, texts: list[str]) -> np.ndarray:
        missing = [
            text
            for text in dict.fromkeys(texts)
            if text not in self._cache
        ]

        if missing:
            if self._model is None:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(self.model_name)

            vectors = self._model.encode(
                missing,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            for text, vector in zip(missing, vectors):
                self._cache[text] = np.asarray(vector, dtype=np.float32)

        return np.vstack([self._cache[text] for text in texts])


def rescale(value: float, floor: float, ceiling: float) -> float:
    """Map a raw cosine similarity onto 0-1 between two fixed anchors.

    The anchors are constants, not per-batch statistics, so a job's score
    does not depend on which other jobs were fetched with it.
    """

    if ceiling <= floor:
        raise ValueError("ceiling must be greater than floor")

    return float(min(1.0, max(0.0, (value - floor) / (ceiling - floor))))


def candidate_chunks(profile: ResumeProfile) -> list[str]:
    """The candidate as separate facets: summary, skills, each role and
    each project."""

    chunks = []

    if profile.summary:
        chunks.append(profile.summary)

    if profile.skills:
        chunks.append(
            "Skills: " + ", ".join(skill.name for skill in profile.skills)
        )

    for item in profile.experience:
        parts = [
            " at ".join(part for part in (item.title, item.company) if part),
            *item.highlights,
        ]
        if item.technologies:
            parts.append("Technologies: " + ", ".join(item.technologies))
        chunks.append("\n".join(part for part in parts if part))

    for project in profile.projects:
        parts = [project.name, project.description, *project.highlights]
        if project.technologies:
            parts.append("Technologies: " + ", ".join(project.technologies))
        chunks.append("\n".join(part for part in parts if part))

    return [chunk for chunk in chunks if chunk.strip()]


def semantic_score(
    candidate_vectors: Optional[np.ndarray],
    job_chunks: list[str],
    embedder: Embedder,
    top_k: int,
    floor: float,
    ceiling: float,
):
    """(score, raw similarity), or (None, None) when there is nothing to
    compare.

    Each job chunk is scored by its best-matching candidate facet; the raw
    value is the mean of the ``top_k`` best job chunks.
    """

    if (
        candidate_vectors is None
        or len(candidate_vectors) == 0
        or not job_chunks
    ):
        return None, None

    job_vectors = embedder.encode(job_chunks)

    best_per_chunk = (job_vectors @ candidate_vectors.T).max(axis=1)
    top = np.sort(best_per_chunk)[::-1][:max(1, top_k)]
    raw = float(top.mean())

    return rescale(raw, floor, ceiling), raw


def title_similarity_fn(embedder: Embedder, floor: float, ceiling: float):
    """Similarity of a job title to the closest target role name, 0-1."""

    def similarity(title: str, roles: list[str]) -> Optional[float]:
        roles = [role for role in roles if role]
        if not title or not roles:
            return None

        vectors = embedder.encode([title, *roles])
        raw = float((vectors[1:] @ vectors[0]).max())

        return rescale(raw, floor, ceiling)

    return similarity
