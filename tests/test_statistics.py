from khugo.data import SourceMetadata, TrainingDocument
from khugo.data.statistics import compute_statistics


def test_statistics_reports_sources_and_licenses() -> None:
    documents = [
        TrainingDocument(
            id="one",
            text="Բարեւ աշխարհ",
            sources=(SourceMetadata("Corpus", "CC BY 4.0"),),
        )
    ]

    result = compute_statistics(documents)

    assert result.document_count == 1
    assert result.character_count == len("Բարեւ աշխարհ")
    assert result.documents_by_license == {"CC BY 4.0": 1}
