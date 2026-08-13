"""Object tracking: the second family, and the first test that the result is not
about language.

``z`` is the item a person holds after a sequence of swaps. Almost everything about the
surface changes -- narrative rather than passage, state tracking rather than
recognition, a
progressively updated value rather than a static property -- while the four-condition
structure is identical, so a window and gather that replicate here are not facts about
language.

**Numeric latents are unscoreable on this tokenizer**: no digit is single-token in
continuation form on Qwen (``" 7"`` is two tokens), which is why program variables were
abandoned. **The answer must be derived, not copied** -- every item is named up front,
so
contamination applies identically to every candidate and cannot favour one, but the
tracked
person's final item must differ from their initial one.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from innerj.analysis.readout import single_token_id, single_token_subset
from innerj.tasks.base import Condition, Record, check_label_symmetry

#: Items, split by whether they are edible. The split drives the automatic
#: condition, which must use the tracked item without naming it.
EDIBLE = [
    "apple", "plum", "peach", "melon", "bread", "cheese", "lemon", "grape",
    "carrot", "onion",
]
INEDIBLE = [
    "hammer", "candle", "mirror", "pencil", "ribbon", "kettle", "shovel",
    "anchor", "helmet", "ladder",
]

NAMES = [
    "Alice", "Bob", "Clara", "David", "Erin", "Frank", "Grace", "Hugo",
    "Iris", "Jonas",
]

OPERATOR_SYMBOLS = [
    "K", "Q", "R", "T", "X", "Z", "M", "V", "W", "B", "F", "G", "H", "J", "P",
]


@dataclass
class TrackingConfig:
    """Generation settings.

    Attributes:
        n_instances: Semantic instances; each yields up to 4 records.
        n_people: People, hence items in play, hence the candidate-set size.
        n_swaps: Tracking depth, traded against reliability -- the family is useless if
            low-load accuracy sits at floor.
        matched_legend: Carry the operator table in every condition.
        seed: Names, items, swaps and tables.
    """

    n_instances: int = 400
    n_people: int = 4
    n_swaps: int = 3
    matched_legend: bool = True
    seed: int = 0


def usable_items(tokenizer: Any) -> tuple[list[str], list[str]]:
    """Single-token edible and inedible items for this tokenizer."""
    return (
        single_token_subset(tokenizer, EDIBLE),
        single_token_subset(tokenizer, INEDIBLE),
    )


def usable_symbols(tokenizer: Any) -> list[str]:
    return single_token_subset(tokenizer, OPERATOR_SYMBOLS)


def generate(tokenizer: Any, config: TrackingConfig | None = None) -> list[Record]:
    """Build matched automatic/report/flexible/control records."""
    config = config or TrackingConfig()
    rng = random.Random(config.seed)

    edible, inedible = usable_items(tokenizer)
    if min(len(edible), len(inedible)) < 2:
        raise ValueError(
            f"need at least 2 single-token items of each kind, have "
            f"{len(edible)} edible and {len(inedible)} inedible"
        )
    symbols_pool = usable_symbols(tokenizer)
    if len(symbols_pool) < config.n_people:
        raise ValueError(
            f"only {len(symbols_pool)} single-token symbols; need {config.n_people}"
        )
    names_pool = single_token_subset(tokenizer, NAMES)
    if len(names_pool) < config.n_people:
        raise ValueError(
            f"only {len(names_pool)} single-token names; need {config.n_people}"
        )
    yes_no = ["Yes", "No"]
    for word in yes_no:
        single_token_id(tokenizer, word)

    records: list[Record] = []
    for i in range(config.n_instances):
        # A mixed basket, so the edible/inedible question is not degenerate.
        n_edible = config.n_people // 2
        items = rng.sample(edible, n_edible) + rng.sample(
            inedible, config.n_people - n_edible
        )
        rng.shuffle(items)
        people = rng.sample(names_pool, config.n_people)
        holder = dict(zip(people, items, strict=True))
        subject = people[0]
        initial = holder[subject]

        lines = [f"{person} has the {holder[person]}." for person in people]
        for _ in range(config.n_swaps):
            a, b = rng.sample(people, 2)
            lines.append(f"{a} and {b} swap items.")
            holder[a], holder[b] = holder[b], holder[a]
        final = holder[subject]

        # The answer must be computed, not read off the opening lines.
        if final == initial:
            continue

        context = "\n".join(lines)
        instance_id = f"track_{i:06d}"
        gold_token = single_token_id(tokenizer, final)
        controls = [single_token_id(tokenizer, it) for it in items if it != final]

        symbols = rng.sample(symbols_pool, config.n_people)
        table = dict(zip(items, symbols, strict=True))
        table_text = "\n".join(f"{k} -> {v}" for k, v in table.items())
        suffix = "matched" if config.matched_legend else "unmatched"

        if config.matched_legend:
            context = f"Reference table:\n{table_text}\n\n{context}"
            flexible_instruction = (
                f"What symbol corresponds to the item {subject} has at the end?"
                f"\nAnswer:"
            )
        else:
            flexible_instruction = (
                f"Use this table:\n{table_text}\n"
                f"What symbol corresponds to the item {subject} has at the end?"
                f"\nAnswer:"
            )

        shared = dict(
            family="tracking",
            semantic_instance_id=instance_id,
            context=context,
            latent_name=f"item_of_{subject}",
            latent_value=final,
            latent_token_id=gold_token,
            control_token_ids=controls,
        )
        group: dict[Condition, Record] = {}

        # Condition A: automatic. Answering needs the tracked item, and the answer
        # is a property of it rather than its name.
        group[Condition.AUTOMATIC] = Record(
            id=f"{instance_id}_automatic",
            condition=Condition.AUTOMATIC,
            template_id=f"track_edible_{suffix}",
            instruction=(
                f"Is the item {subject} has at the end edible?\nAnswer:"
            ),
            gold_answer="Yes" if final in edible else "No",
            candidate_answers=yes_no,
            candidate_token_ids=[single_token_id(tokenizer, w) for w in yes_no],
            operator_family="predicate",
            operator_id="edible",
            **shared,
        )

        group[Condition.REPORT] = Record(
            id=f"{instance_id}_report",
            condition=Condition.REPORT,
            template_id=f"track_report_{suffix}",
            instruction=f"What item does {subject} have at the end?\nAnswer:",
            gold_answer=final,
            candidate_answers=items,
            candidate_token_ids=[single_token_id(tokenizer, it) for it in items],
            operator_family="report",
            operator_id="report_00",
            **shared,
        )

        group[Condition.FLEXIBLE] = Record(
            id=f"{instance_id}_flexible",
            condition=Condition.FLEXIBLE,
            template_id=f"track_lookup_{suffix}",
            instruction=flexible_instruction,
            gold_answer=table[final],
            candidate_answers=symbols,
            candidate_token_ids=[single_token_id(tokenizer, s) for s in symbols],
            operator_family="lookup",
            operator_id=f"lookup_{rng.randrange(10_000):04d}",
            meta={"table": table, "subject": subject},
            **shared,
        )

        # Condition D: matched control. Same shape, single-token answer, and the
        # tracked state is irrelevant to it.
        absent = rng.choice([n for n in names_pool if n not in people])
        asked = rng.choice([people[-1], absent])
        group[Condition.CONTROL] = Record(
            id=f"{instance_id}_control",
            condition=Condition.CONTROL,
            template_id=f"track_control_{suffix}",
            instruction=f"Is {asked} mentioned in the text?\nAnswer:",
            gold_answer="Yes" if asked in people else "No",
            candidate_answers=yes_no,
            candidate_token_ids=[single_token_id(tokenizer, w) for w in yes_no],
            operator_family="control",
            operator_id="control_00",
            **shared,
        )

        if config.matched_legend:
            check_label_symmetry(group)
        records.extend(group.values())

    return records
