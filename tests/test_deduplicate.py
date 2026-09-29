from khugo.data import SourceMetadata, TrainingDocument
from khugo.data.deduplicate import deduplicate_documents


def test_deduplication_retains_metadata_for_all_sources() -> None:
    first = TrainingDocument(
        id="first",
        text="Բարեւ աշխարհ սա փորձնական նախադասություն է",
        sources=(SourceMetadata("Corpus A", "CC BY 4.0"),),
    )
    duplicate = TrainingDocument(
        id="second",
        text="ԲԱՐԵՒ ԱՇԽԱՐՀ ՍԱ ՓՈՐՁՆԱԿԱՆ ՆԱԽԱԴԱՍՈՒԹՅՈՒՆ Է",
        sources=(SourceMetadata("Corpus B", "CC0-1.0"),),
    )

    result = deduplicate_documents([first, duplicate])

    assert len(result) == 1
    assert [source.name for source in result[0].sources] == ["Corpus A", "Corpus B"]
    assert result[0].metadata["deduplicated_ids"] == ["second"]
