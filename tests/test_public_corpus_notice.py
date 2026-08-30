"""The shipped public corpora and their legal notice move as one artifact."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from tests import measurement_provenance as measurement

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from open_surface import OPEN_TREE_CATEGORIES  # noqa: E402
from public_corpus_notice import (  # noqa: E402
    ATTRIBUTIONS,
    PUBLIC_CORPUS_ROOTS,
    PublicCorpusNoticeError,
    assert_notice_current,
    build,
    inventory,
    notice_path,
)


def test_inventory_rederives_the_reviewed_scope() -> None:
    observed = inventory(ROOT)

    assert observed.total == 138
    assert observed.public_domain == 67
    assert observed.attribution_required == 71
    assert observed.attribution_counts == {
        "github:Textualize/rich": 15,
        "github:expressjs/express": 12,
        "github:nlohmann/json": 11,
        "github:pytest-dev/pytest": 11,
        "github:encode/httpx": 11,
        "github:golang/example": 11,
    }


def test_notice_is_a_fresh_render_of_the_fixture_obligations() -> None:
    assert_notice_current(ROOT)
    assert notice_path(ROOT).read_text(encoding="utf-8") == build(ROOT)


def test_shipped_public_corpus_bytes_match_the_measured_baseline() -> None:
    baseline_path = ROOT / "tests" / "baselines" / "public_corpus_baseline.json"
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    expected = baseline["manifest"]["corpora"]

    for relative_root in PUBLIC_CORPUS_ROOTS:
        corpus = relative_root.rstrip("/")
        digest, count = measurement._corpus_digest(corpus)
        assert digest == expected[corpus]["digest"], corpus
        assert count == expected[corpus]["files"], corpus


def test_generated_corpus_claims_survive_without_covering_public_text() -> None:
    notice = build(ROOT)

    assert "They contain no third-party text and no real personal data." in notice
    assert "There is no upstream and no third-party attribution is owed." in notice
    assert "PUBLIC CORPORA" in notice
    assert all(item.license_text in notice for item in ATTRIBUTIONS)


def test_open_surface_admits_every_reviewed_public_fixture() -> None:
    corpora = next(item for item in OPEN_TREE_CATEGORIES if item.name == "corpora")

    assert set(PUBLIC_CORPUS_ROOTS) <= set(corpora.roots)
    for relative_root in PUBLIC_CORPUS_ROOTS:
        paths = sorted((ROOT / relative_root).glob("*.json"))
        assert paths, relative_root
        assert all(corpora.matches(path.relative_to(ROOT).as_posix()) for path in paths)


def test_an_unknown_source_is_refused_before_it_can_ship(tmp_path: Path) -> None:
    """Known-dirty control: a seventh project must fail, never disappear."""
    relative_root = "benchmarks/fixtures/public_code_v1/"
    corpus = tmp_path / relative_root
    corpus.mkdir(parents=True)
    fixture = {
        "fixture_id": "unknown-source",
        "provenance": {
            "source": "github:unknown/project",
            "license": "MIT",
            "external": {
                "publisher": "unknown/project",
                "source_url": "https://github.com/unknown/project/blob/abc/file.py",
            },
        },
    }
    (corpus / "unknown-source.json").write_text(json.dumps(fixture), encoding="utf-8")

    with pytest.raises(PublicCorpusNoticeError, match="un-attributed public source"):
        inventory(tmp_path, roots=(relative_root,))
