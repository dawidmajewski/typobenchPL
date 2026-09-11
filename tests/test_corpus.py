from __future__ import annotations

import io
import json

import pytest

from typobenchpl.corpus import CorpusError, score_stream, verify_cases


def test_builtin_verification_corpus_passes() -> None:
    summary = verify_cases()

    assert summary.successful
    assert summary.total > 0
    assert summary.passed == summary.total


def test_scores_exact_number_of_records() -> None:
    records = [
        {"id": "1", "text": "Poprawnie."},
        {"id": "2", "text": "Źle,, tutaj."},
        {"id": "3", "text": "Też poprawnie."},
        {"id": "4", "text": "Źle;; tutaj."},
        {"id": "ignored", "text": "Źle,, lecz rekord jest poza limitem."},
    ]
    stream = io.StringIO("\n".join(json.dumps(record) for record in records))

    summary = score_stream(stream, runs=4)

    assert summary.score == 50.0
    assert summary.passed == 2
    assert summary.failed == 2
    assert summary.issues_by_rule == {"G010": 1, "G011": 1}


def test_score_fails_when_there_are_too_few_records() -> None:
    stream = io.StringIO('{"text":"Poprawnie."}\n')

    with pytest.raises(CorpusError, match="expected 2 texts, found 1"):
        score_stream(stream, runs=2)


def test_score_rejects_invalid_record() -> None:
    stream = io.StringIO('{"value":"Brak pola text"}\n')

    with pytest.raises(CorpusError, match="'text' must be a string"):
        score_stream(stream, runs=1)
