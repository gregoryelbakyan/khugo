import pytest

from khugo.data import TrainingDocument


def test_document_requires_source_metadata() -> None:
    with pytest.raises(ValueError, match="requires source metadata"):
        TrainingDocument.from_mapping({"id": "untracked", "text": "Բարեւ"})


def test_document_accepts_legacy_single_source_and_serializes_sources() -> None:
    document = TrainingDocument.from_mapping(
        {
            "id": "tracked",
            "text": "Բարեւ",
            "source": {"name": "Corpus", "license": "CC0-1.0"},
        }
    )

    assert document.to_mapping()["sources"] == [{"name": "Corpus", "license": "CC0-1.0"}]
