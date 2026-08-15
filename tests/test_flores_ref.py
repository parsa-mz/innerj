"""Guards for the FLORES stub/rehydrate round trip.

The release path removes FLORES text from the records and rebuilds it on demand. Two
things have to hold or the released dataset is not the one the paper reports on:

* the round trip is **byte-exact**, so a rebuilt file is the file the numbers came from;
* a stub carries **no FLORES text**, which is the whole point of the exercise.

These run in either tree, because the development tree holds full records and a release
tree holds stubs, and the interesting assertion differs. Both need the FLORES snapshot,
so both skip when it is not cached rather than passing vacuously.
"""

from __future__ import annotations

import copy
import hashlib
import json

import pytest

from innerj import config
from innerj.tasks import flores_ref as fr

#: The primary released dataset and its published SHA-256 prefix, as pinned in the
#: paper's artifact appendix. If a rebuild does not reproduce this, the snapshot is not
#: the revision the dataset was built from and no downstream number is comparable.
PRIMARY = "language/Qwen3.6-27B_matched_n400_s0.jsonl"
PRIMARY_SHA = "00623f40"
PRIMARY_RECORDS = 1954


def _frame():
    from innerj.tasks.language import LANGUAGE_NAMES, load_parallel

    try:
        return load_parallel(sorted(LANGUAGE_NAMES))
    except FileNotFoundError as exc:
        pytest.skip(f"FLORES not cached: {exc}")


def _primary():
    path = config.DATA_ROOT / PRIMARY
    if not path.is_file():
        pytest.skip(f"{path} not present")
    text = path.read_text()
    rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    return text, rows


def _dump(records):
    return "\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n"


def test_the_dataset_reserialises_unchanged():
    """Byte-exactness downstream rests on this: the dump is the file."""
    text, rows = _primary()
    assert len(rows) == PRIMARY_RECORDS
    assert _dump(rows) == text


def test_the_round_trip_is_exact_and_lands_on_the_published_hash():
    """Whichever form is on disk, the *full* dataset must hash to ``PRIMARY_SHA``."""
    text, rows = _primary()
    frame = _frame()

    if fr.PLACEHOLDER in text:  # a release tree: stubs on disk
        full = _dump(fr.rehydrate(copy.deepcopy(rows), frame))
        assert fr.PLACEHOLDER not in full, "rehydration left a placeholder behind"
    else:  # a development tree: full records on disk
        stub_text = _dump(fr.stub(copy.deepcopy(rows), frame))
        assert fr.PLACEHOLDER in stub_text
        longest = max((fr.passage_of(r["context"]) for r in rows), key=len)
        assert longest[:60] not in stub_text, "a passage survived stubbing"
        stubbed = [json.loads(x) for x in stub_text.splitlines()]
        full = _dump(fr.rehydrate(stubbed, frame))
        assert full == text, "rehydration is not byte-exact"

    assert hashlib.sha256(full.encode()).hexdigest().startswith(PRIMARY_SHA)


def test_rehydrating_something_without_provenance_raises():
    """A missing reference must be fatal, not a silent pass-through of placeholders."""
    bare = {"id": "x", "condition": "report", "context": fr.PLACEHOLDER, "meta": {}}
    with pytest.raises(ValueError, match="not a stub"):
        fr.rehydrate([bare], object())


def test_an_ambiguous_or_absent_passage_raises():
    """Provenance is refused unless it is unique -- a guess would rebuild wrong text."""
    with pytest.raises(ValueError, match="not found in FLORES"):
        fr.resolve("nothing like a FLORES sentence", "Korean", {})
    with pytest.raises(ValueError, match="ambiguous"):
        fr.resolve("x", "Korean", {"x": [("ko", 1), ("ko", 900)]})
    with pytest.raises(ValueError, match="not a FLORES language"):
        fr.resolve("x", "Klingon", {"x": [("ko", 1)]})
