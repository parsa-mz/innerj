"""Release the benchmark without redistributing FLORES text.

Every prompt quotes a FLORES-200 passage verbatim, so a released record file is a
CC BY-SA 4.0 derivative of FLORES. A **stub** file carries only the fields this project
authored plus a *reference* to the passage -- language code, row offset, span -- and
:func:`rehydrate` rebuilds the byte-identical original from FLORES at the pinned
revision. Two things are stubbed, and nothing else:

* the passage inside ``context``, replaced by :data:`PLACEHOLDER`;
* in the ``automatic`` arm only, ``gold_answer`` and ``candidate_answers``, which are
  first words of the true continuation in each candidate language.

Token ids stay: they are integers, and keeping them is what makes rehydration
independent of the tokenizer, so a rebuild cannot drift with a ``transformers``
release the way regenerating from scratch can.

The round trip is exact and must be checked rather than assumed --
``tests/test_flores_ref.py`` pins the rebuilt primary dataset to its published
SHA-256.
"""

from __future__ import annotations

from typing import Any

from innerj.tasks.language import LANGUAGE_NAMES

#: Marks where a FLORES passage was removed. Not a legal token in any prompt, so
#: its presence in a file is exactly the test for "still a stub".
PLACEHOLDER = "{{FLORES}}"

#: The ``meta`` key holding provenance. ``meta`` already carries ``table`` and
#: ``dummy``, so this is additive and removing it restores the original mapping --
#: insertion order puts it last, which is what keeps the round trip byte-exact.
META_KEY = "flores"

_TABLE_PREFIX = "Reference table:"
_NAME_TO_CODE = {name: code for code, name in LANGUAGE_NAMES.items()}


def passage_of(context: str) -> str:
    """The FLORES passage inside ``context``.

    Under a matched legend the operator table is prepended and separated by a blank
    line; the table itself is newline-joined and so contains none. Without one the
    context *is* the passage.
    """
    if context.startswith(_TABLE_PREFIX):
        return context.split("\n\n", 1)[1]
    return context


def span_index(frame: Any, span: int) -> dict[str, list[tuple[str, int]]]:
    """Map every joined ``span``-sentence window to the (code, offset) producing it.

    Built once per rebuild. Offsets start at 1 and stop ``span + 1`` short of the end,
    matching the generator, which holds a sentence back as the true continuation.
    """
    out: dict[str, list[tuple[str, int]]] = {}
    for code in LANGUAGE_NAMES:
        if code not in frame.columns:
            continue
        col = frame[code].tolist()
        for start in range(1, len(col) - span):
            out.setdefault(" ".join(col[start : start + span]), []).append(
                (code, start)
            )
    return out


def resolve(context: str, latent_value: str, index: dict) -> tuple[str, int]:
    """Locate a record's passage in FLORES, refusing anything but a unique hit.

    The gold language pins which column to accept, so a window that happens to repeat
    across languages cannot silently resolve to the wrong one. An ambiguous or absent
    hit raises: a stub whose provenance is a guess would rebuild the wrong text.
    """
    passage = passage_of(context)
    want = _NAME_TO_CODE.get(latent_value)
    if want is None:
        raise ValueError(f"{latent_value!r} is not a FLORES language in this benchmark")
    hits = [h for h in index.get(passage, []) if h[0] == want]
    if not hits:
        raise ValueError(
            f"passage not found in FLORES under {want!r}; the snapshot is not the "
            f"revision this dataset was built from"
        )
    if len(hits) > 1:
        offsets = [h[1] for h in hits]
        raise ValueError(f"passage is ambiguous in {want!r}: offsets {offsets}")
    return hits[0]


def _choices_of(group: list[dict]) -> list[str] | None:
    """Candidate language codes in generation order, read off the ``report`` arm.

    ``report``'s candidates *are* the language names in that order, so the order the
    automatic arm's continuation words follow is recoverable without a tokenizer.
    """
    for r in group:
        if r.get("condition") == "report":
            names = r.get("candidate_answers") or []
            codes = [_NAME_TO_CODE.get(n) for n in names]
            if all(codes):
                return codes  # type: ignore[return-value]
    return None


def stub(records: list[dict], frame: Any, *, span: int = 3) -> list[dict]:
    """Replace FLORES text with references, in place, returning the same records.

    Records are grouped by semantic instance because the passage is shared across an
    instance's arms and only resolved once.
    """
    index = span_index(frame, span)
    groups: dict[str, list[dict]] = {}
    for r in records:
        groups.setdefault(r["semantic_instance_id"], []).append(r)

    for group in groups.values():
        code, start = resolve(group[0]["context"], group[0]["latent_value"], index)
        choices = _choices_of(group)
        for r in group:
            ref = {"code": code, "start": start, "span": span}
            if r.get("condition") == "automatic":
                if choices is None:
                    raise ValueError(
                        f"{r['id']}: the automatic arm's candidates are FLORES words, "
                        f"and no report arm is present to recover their order from"
                    )
                ref["choices"] = choices
                r["gold_answer"] = PLACEHOLDER
                r["candidate_answers"] = [PLACEHOLDER] * len(r["candidate_answers"])
            r["context"] = r["context"].replace(passage_of(r["context"]), PLACEHOLDER)
            r.setdefault("meta", {})[META_KEY] = ref
    return records


def rehydrate(records: list[dict], frame: Any) -> list[dict]:
    """Rebuild the original records from FLORES, in place, returning the same records.

    Reverses :func:`stub` exactly, including dropping the provenance key again, so the
    result is byte-identical to the file the stub was made from.
    """
    for r in records:
        ref = (r.get("meta") or {}).pop(META_KEY, None)
        if ref is None:
            raise ValueError(f"{r.get('id')}: no {META_KEY!r} provenance; not a stub")
        code, start, span = ref["code"], ref["start"], ref["span"]
        if code not in frame.columns:
            raise ValueError(f"FLORES snapshot has no {code!r} column")
        passage = " ".join(frame[code].tolist()[start : start + span])
        r["context"] = r["context"].replace(PLACEHOLDER, passage)
        if r.get("condition") == "automatic":
            row = frame.iloc[start + span]
            words = [str(row[c]).split()[0] for c in ref["choices"]]
            r["candidate_answers"] = words
            r["gold_answer"] = words[ref["choices"].index(code)]
    return records
