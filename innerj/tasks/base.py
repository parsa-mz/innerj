"""Dataset schema for JGateBench.

One *semantic instance* is a latent world paired with conditions differing only in what
the
model must do with ``z``: ``AUTOMATIC`` (used, never exposed), ``REPORT`` (stated),
``FLEXIBLE`` (passed into a prompted operator) and ``CONTROL`` (matched form, does not
need
``z``). Conditions sharing a ``semantic_instance_id`` share their context verbatim, so
every
contrast is within-instance and paired.

Two invariants are enforced at construction because each has silently invalidated a
pipeline: **the label never appears in the context**, a concept present in the prompt
reading at rank ~1; and **labels are single-token in continuation form**, the lens being
token-indexed.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path


class Condition(StrEnum):
    """Which downstream use of ``z`` the prompt demands."""

    AUTOMATIC = "automatic"
    REPORT = "report"
    FLEXIBLE = "flexible"
    CONTROL = "control"
    #: The operator is present but the value it applies to is *given* in the prompt,
    #: so no inference is required. Crossing this with ``report`` isolates
    #: latent-variable demand from the compositional work of applying an operator:
    #: ``(flexible - supplied) - (report - control)``.
    SUPPLIED = "supplied"


#: The two conditions whose contrast defines workspace entry.
CONTRAST = (Condition.AUTOMATIC, Condition.FLEXIBLE)


@dataclass
class Record:
    """One (semantic instance, condition) pair: a prompt plus exact ground truth.

    Attributes:
        latent_value: The value of ``z``, never present in ``context``.
        latent_token_id: Continuation-form token id of ``latent_value`` -- the direction
            whose entry we measure.
        control_token_ids: Frequency-matched tokens for the ``M_z`` contrast.
        gold_answer: The correct output *for this condition*, which differs across
            conditions
            even though ``z`` does not.
        candidate_answers: The forced-choice set; accuracy is never open-vocabulary
            argmax.
        query_position: Position of the retrieval cue, where entry is measured. A
            passive
            sentence-final read understates persistence badly enough that one project
            retracted
            a headline over it.
    """

    id: str
    family: str
    condition: Condition
    semantic_instance_id: str
    template_id: str
    context: str
    instruction: str
    latent_name: str
    latent_value: str
    latent_token_id: int
    control_token_ids: list[int]
    gold_answer: str
    candidate_answers: list[str]
    candidate_token_ids: list[int]
    operator_family: str | None = None
    operator_id: str | None = None
    query_position: int | None = None
    answer_position: int | None = None
    meta: dict = field(default_factory=dict)

    @property
    def prompt(self) -> str:
        return f"{self.context}\n\n{self.instruction}"

    def to_json(self) -> str:
        d = asdict(self)
        d["condition"] = str(self.condition)
        return json.dumps(d, ensure_ascii=False)


def check_label_absent(context: str, label: str) -> None:
    """Raise if ``label`` appears in ``context``: the prompt-copy ceiling, invisible in
        the output because every condition then reads the concept at rank ~1.
    """
    if label.lower() in context.lower():
        raise ValueError(
            f"latent label {label!r} appears in the context, which pins its "
            f"rank at ~1 at its own position and destroys the contrast"
        )


def check_label_symmetry(group: dict[Condition, Record]) -> None:
    """Raise unless every condition of an instance mentions the same labels.

    A flexible-arm table naturally spells out ``"Spanish -> 7"``, so the contrast
    measures
    *whether the label was printed* and returns a large, clean, artifactual effect in
    exactly the
    predicted direction. Either every arm names the label or none does.
    """
    presence: dict[bool, list[Condition]] = {True: [], False: []}
    for condition, record in group.items():
        label = record.latent_value.lower()
        presence[label in record.prompt.lower()].append(condition)
    if presence[True] and presence[False]:
        raise ValueError(
            f"latent label appears in {[str(c) for c in presence[True]]} but not "
            f"{[str(c) for c in presence[False]]}. This asymmetry manufactures an "
            f"entry effect; make the label present in all arms or none."
        )


def write_jsonl(records: Iterable[Record], path: str | Path) -> int:
    """Write records as JSONL, creating parent directories. Returns the count."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(record.to_json() + "\n")
            n += 1
    return n


def read_jsonl(path: str | Path) -> Iterator[Record]:
    """Read records back, restoring the ``Condition`` enum."""
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            d = json.loads(line)
            d["condition"] = Condition(d["condition"])
            yield Record(**d)


def group_by_instance(records: Iterable[Record]) -> dict[str, dict[Condition, Record]]:
    """Index records by semantic instance, then condition, for the paired statistics."""
    out: dict[str, dict[Condition, Record]] = {}
    for record in records:
        out.setdefault(record.semantic_instance_id, {})[record.condition] = record
    return out


def complete_instances(
    records: Iterable[Record], required: Iterable[Condition] = CONTRAST
) -> dict[str, dict[Condition, Record]]:
    """Keep only instances present in every ``required`` condition; an incomplete one
        would unbalance a comparison that is only meaningful paired.
    """
    required = set(required)
    grouped = group_by_instance(records)
    return {k: v for k, v in grouped.items() if required <= set(v)}
