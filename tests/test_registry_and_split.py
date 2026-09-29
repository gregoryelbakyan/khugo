from pathlib import Path

import pytest

from khugo.data import SourceMetadata, TrainingDocument
from khugo.data.registry import SourceRecord, load_source_registry
from khugo.data.split import train_validation_split
from khugo.data.validation import assess_training_eligibility


def _document(document_id: str, language: str = "hyw") -> TrainingDocument:
    return TrainingDocument(
        id=document_id,
        text="Արեւմտահայերէն փորձնական նախադասութիւն մըն է։",
        sources=(SourceMetadata("Corpus", "CC0-1.0", source_id="approved"),),
        language=language,
    )


def _registry() -> dict[str, SourceRecord]:
    return {
        "approved": SourceRecord(
            source_id="approved", source_name="Corpus", url="https://example.org", author=None,
            license="CC0-1.0", copyright_status="public-domain", retrieved_at=None,
            training_allowed=True, redistribution_allowed=True,
        )
    }


def test_eligibility_rejects_eastern_armenian_and_unregistered_sources() -> None:
    assert assess_training_eligibility(_document("hye", "hye"), _registry()).reason == "language_not_hyw"
    unregistered = TrainingDocument(
        id="unregistered", text="Արեւմտահայերէն փորձնական նախադասութիւն մըն է։",
        sources=(SourceMetadata("Unknown", "unknown", source_id="unknown"),), language="hyw",
    )
    assert assess_training_eligibility(unregistered, _registry()).reason == "unregistered_source"


def test_eligibility_rejects_license_mismatch() -> None:
    mismatched = TrainingDocument(
        id="mismatch", text="Արեւմտահայերէն փորձնական նախադասութիւն մըն է։",
        sources=(SourceMetadata("Corpus", "CC BY 4.0", source_id="approved"),), language="hyw",
    )
    assert assess_training_eligibility(mismatched, _registry()).reason == "license_mismatch"


def test_split_is_stable_for_a_fixed_seed() -> None:
    documents = [_document(str(index)) for index in range(20)]
    assert train_validation_split(documents, validation_ratio=0.2, seed=42) == train_validation_split(
        documents, validation_ratio=0.2, seed=42
    )


def test_registry_rejects_duplicate_ids(tmp_path: Path) -> None:
    registry = tmp_path / "sources.yaml"
    registry.write_text(
        "sources:\n  - source_id: same\n    source_name: A\n    url: https://a\n    license: CC0\n"
        "    copyright_status: public-domain\n    training_allowed: true\n    redistribution_allowed: true\n"
        "  - source_id: same\n    source_name: B\n    url: https://b\n    license: CC0\n"
        "    copyright_status: public-domain\n    training_allowed: true\n    redistribution_allowed: true\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate"):
        load_source_registry(registry)
