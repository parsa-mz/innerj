"""Dataset schema for JGateBench.

One *semantic instance* is a latent world (a passage, a program, a grid walk)
paired with the four conditions that differ only in what the model is asked to
do with the latent variable ``z``:

* ``AUTOMATIC``  -- ``z`` must be used, but never exposed;
* ``REPORT``     -- ``z`` must be stated;
* ``FLEXIBLE``   -- ``z`` must be passed into an operator the prompt defines;
* ``CONTROL``    -- an instruction of matched form that does not need ``z``.

Conditions sharing a ``semantic_instance_id`` share their context verbatim, so
the automatic/flexible contrast is a within-instance comparison and the
statistics are paired.

Two invariants are enforced at construction rather than checked later, because
each has silently invalidated a pipeline before:

* **The label never appears in the context.** A concept present in the prompt
  reads at rank ~1 at its own position, which saturates the readout and makes
  every condition look identical.
* **Labels are single-token in the continuation form.** The J-lens is
  token-indexed, so a multi-token label has no single direction to measure.
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
        latent_value: The value of ``z``, e.g. ``"Spanish"``. Never present in
            ``context``.
        latent_token_id: Continuation-form token id of ``latent_value``. This is
            the direction whose workspace entry we measure.
        control_token_ids: Frequency-matched tokens for the ``M_z`` contrast.
        gold_answer: The correct output *for this condition*. Differs across
            conditions even though ``z`` does not.
        candidate_answers: The forced-choice set. Behavioural accuracy is scored
            over this set, never open-vocabulary argmax.
        query_position: Token position of the retrieval cue. Entry is measured
            here: a passive read at a sentence-final position understates
            persistence badly enough that one project retracted a headline over
            it. You cannot test a workspace without querying it.
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
    """Raise if ``label`` appears in ``context``, case-insensitively.

    This is the prompt-copy ceiling. It is checked as a hard invariant because
    the failure is invisible in the output: every condition reads the concept at
    rank ~1 and the contrast collapses to zero.
    """
    if label.lower() in context.lower():
        raise ValueError(
            f"latent label {label!r} appears in the context, which pins its "
            f"rank at ~1 at its own position and destroys the contrast"
        )


def check_label_symmetry(group: dict[Condition, Record]) -> None:
    """Raise unless every condition of an instance mentions the same labels.

    The failure this catches is worse than a null result. A flexible-condition
    lookup table naturally spells out ``"Spanish -> 7"``, so the gold label sits
    in that prompt and not in the automatic one. The contrast then measures
    *whether the label was printed*, not whether the model wrote it to the
    workspace -- and it yields a large, clean, entirely artifactual effect in
    exactly the predicted direction.

    Symmetry is therefore an invariant, not a preference: either every arm names
    the label or none does.
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
    """Index records by semantic instance, then condition.

    The paired statistics need this: an automatic/flexible effect is a
    within-instance difference, and treating the two arms as independent samples
    gives an interval several times too narrow.
    """
    out: dict[str, dict[Condition, Record]] = {}
    for record in records:
        out.setdefault(record.semantic_instance_id, {})[record.condition] = record
    return out


def complete_instances(
    records: Iterable[Record], required: Iterable[Condition] = CONTRAST
) -> dict[str, dict[Condition, Record]]:
    """Keep only instances present in every ``required`` condition.

    An incomplete instance would enter one arm and not the other, unbalancing a
    comparison that is only meaningful paired.
    """
    required = set(required)
    grouped = group_by_instance(records)
    return {k: v for k, v in grouped.items() if required <= set(v)}
