from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from sentinel_ai.schemas import RetrievalResult


@dataclass(frozen=True)
class Document:
    title: str
    text: str


class LocalPolicyRetriever:
    def __init__(self, documents: list[Document]) -> None:
        self.documents = documents
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self.matrix = self.vectorizer.fit_transform([doc.text for doc in documents])

    @classmethod
    def from_markdown(cls, path: Path) -> "LocalPolicyRetriever":
        text = path.read_text(encoding="utf-8")
        sections: list[Document] = []
        for raw_section in text.split("\n## "):
            section = raw_section.strip()
            if not section:
                continue
            lines = section.splitlines()
            title = lines[0].replace("#", "").strip()
            body = "\n".join(lines[1:]).strip()
            sections.append(Document(title=title, text=body or title))
        return cls(sections)

    def search(self, query: str, top_k: int = 3) -> list[RetrievalResult]:
        query_vector = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vector, self.matrix).ravel()
        ranked = scores.argsort()[::-1][:top_k]
        return [
            RetrievalResult(
                title=self.documents[index].title,
                score=round(float(scores[index]), 4),
                text=self.documents[index].text,
            )
            for index in ranked
        ]
