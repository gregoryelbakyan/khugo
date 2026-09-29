from khugo.data import SourceMetadata, TrainingDocument
from khugo.data.normalize import normalize_document, normalize_text


def test_normalize_text_preserves_armenian_and_normalizes_whitespace() -> None:
    assert normalize_text("  Բարեւ\r\n\tաշխարհ  ") == "Բարեւ\nաշխարհ"


def test_normalize_document_preserves_source_metadata() -> None:
    source = SourceMetadata(name="Corpus", license="CC BY 4.0", url="https://example.org")
    document = TrainingDocument(id="one", text="  Բարեւ  ", sources=(source,))

    normalized = normalize_document(document)

    assert normalized.text == "Բարեւ"
    assert normalized.sources == (source,)
