from pathlib import Path

from sentinel_ai.rag.vector_store import LocalPolicyRetriever


def test_retriever_returns_ranked_context(tmp_path: Path) -> None:
    corpus = tmp_path / "policy.md"
    corpus.write_text(
        "# Policies\n\n## Device Trust\nLow device trust requires review.\n\n"
        "## Drift\nFeature drift requires investigation.\n",
        encoding="utf-8",
    )
    retriever = LocalPolicyRetriever.from_markdown(corpus)

    results = retriever.search("low device trust", top_k=1)

    assert results[0].title == "Device Trust"
    assert results[0].score > 0
