from khugo.data import SourceMetadata, TrainingDocument
from khugo.data.filter import FilterConfig, assess_document


def _document(text: str) -> TrainingDocument:
    return TrainingDocument(id="document", text=text, sources=(SourceMetadata("Test", "CC0-1.0"),))


def test_filter_rejects_short_documents() -> None:
    decision = assess_document(_document("կարճ"), FilterConfig(min_characters=10))

    assert not decision.accepted
    assert decision.reason == "too_short"


def test_filter_rejects_symbol_heavy_documents() -> None:
    decision = assess_document(_document("!!!! 1234 !!!!"), FilterConfig(min_characters=5))

    assert not decision.accepted
    assert decision.reason == "low_alphabetic_ratio"
