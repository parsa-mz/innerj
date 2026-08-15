"""Language identity: the family that most directly extends the source paper.

``z`` is the language of a passage, taken from FLORES-200, which is *N-way parallel*.
Two
things follow. **The counterfactual is exact**: a Spanish passage and its German
counterpart
are the same sentences, so patching between them varies the latent variable and nothing
else. **The automatic condition gets a real forced choice**: the continuation sentence
also
exists in every language, so the candidate set is its first token in each candidate
language
-- no judge, and distractors matched by construction.
"""

from __future__ import annotations

import glob
import random
from dataclasses import dataclass
from typing import Any

import pandas as pd

from innerj.analysis.readout import single_token_id, single_token_subset
from innerj.tasks.base import (
    Condition,
    Record,
    check_label_absent,
    check_label_symmetry,
)

FLORES_REPO = "haoranxu/FLORES-200"

#: FLORES code -> English language name. The name must be single-token in
#: continuation form for the tokenizer in use; :func:`usable_languages` filters
#: this list per checkpoint rather than assuming.
LANGUAGE_NAMES: dict[str, str] = {
    "es": "Spanish",
    "de": "German",
    "fr": "French",
    "it": "Italian",
    "pt": "Portuguese",
    "nl": "Dutch",
    "ru": "Russian",
    "pl": "Polish",
    "tr": "Turkish",
    "sv": "Swedish",
    "da": "Danish",
    "fi": "Finnish",
    "cs": "Czech",
    "el": "Greek",
    "he": "Hebrew",
    "ar": "Arabic",
    "hi": "Hindi",
    "ja": "Japanese",
    "ko": "Korean",
    "vi": "Vietnamese",
}

#: Candidate symbols for flexible-operator outputs. They carry no semantic
#: relation to any language name, so a correct answer cannot come from an
#: association shortcut that bypasses the latent variable. Single-token-ness is
#: *not* assumed: on Qwen's vocabulary `" 6"` is two tokens while `" 7"` is one,
#: so the usable subset is filtered per tokenizer by :func:`usable_symbols`.
OPERATOR_SYMBOLS = [
    "7", "3", "9", "4", "2", "8", "5", "6", "1",
    "K", "Q", "R", "T", "X", "Z", "M", "V", "W",
]

#: Instruction per arm, keyed by ``matched_length``. See
#: :class:`LanguageConfig` for why the second set exists.
#:
#: The ``False`` set is what every published number was measured on. Its arms are
#: 13/12/14/12 tokens on the Qwen tokenizer.
#:
#: The ``True`` set is 14 tokens in every arm and shares the 9-token tail
#: ``". Use the passage above.\nAnswer:"``, so a span read from the end covers the
#: same tokens in each arm. Three constraints shaped it, and two designs that
#: satisfied the first were rejected on the others:
#:
#: * equal length **and** a long shared tail --- so a span measure is comparable;
#: * no arm at ceiling or floor (trap 8) --- a 12-token variant put flexible at
#:   exactly 1.000, which makes every correlation with behaviour zero by
#:   construction;
#: * the demands must survive --- a 16-token "slot" design (``Task: symbol`` in a
#:   fixed frame) differs at a *single* token position, which is optimal for the
#:   measurement, and the model cannot execute it: the control arm scored 0 of 40.
#:   Terseness that reads as a label rather than a question destroys the task.
#:
#: Behaviour on 120 instances, matched against the default set in the same run:
#: automatic 0.550 (vs 0.575), report 0.975 (0.975), flexible 0.933 (0.942),
#: control 0.958 (0.925). The flexible-minus-control accuracy gap that Stage 1
#: depends on stays within noise of zero.
INSTRUCTIONS: dict[bool, dict[str, str]] = {
    False: {
        "automatic": "Continue the passage. Write the next sentence.\nAnswer:",
        "report": "What language is the passage written in?\nAnswer:",
        "flexible": "What symbol corresponds to the language of the passage?\nAnswer:",
        "control": "Does the passage contain a question mark?\nAnswer:",
        # {dummy} is a candidate language other than the passage's, so the operator
        # is exercised without the model inferring anything from the passage. It must
        # not be the gold language: naming that would put the concept in the prompt
        # at rank ~1 by copying (trap 2) and destroy the measurement.
        "supplied": "Treat the language as {dummy}. Which symbol is it?\nAnswer:",
    },
    True: {
        "automatic": "Write the next sentence. Use the passage above.\nAnswer:",
        "report": "Name the written language. Use the passage above.\nAnswer:",
        "flexible": "Give the language symbol. Use the passage above.\nAnswer:",
        "control": "Report any question mark. Use the passage above.\nAnswer:",
        "supplied": "Given the language {dummy}. Give the language symbol.\nAnswer:",
    },
}


def _snapshot_dir() -> str:
    """Locate the downloaded FLORES snapshot in the HuggingFace cache.

    Resolved through ``huggingface_hub``, so it finds the standard cache wherever that
    is. It was once an absolute glob into one machine's shared cache, and ran nowhere
    else.
    """
    from huggingface_hub import snapshot_download

    try:
        return snapshot_download(
            FLORES_REPO, repo_type="dataset", local_files_only=True
        )
    except Exception as exc:  # not cached, or cache unreadable
        raise FileNotFoundError(
            f"FLORES-200 not in the HuggingFace cache ({exc}). Run:\n"
            f"  hf download {FLORES_REPO} --repo-type dataset"
        ) from exc


def load_parallel(languages: list[str]) -> pd.DataFrame:
    """Load an N-way parallel frame indexed by the shared English sentence.

    Asserted rather than assumed: if a future mirror breaks the alignment, the
    counterfactual
    stops being content-matched and every Stage-1 number silently changes meaning.
    """
    root = _snapshot_dir()
    frames: dict[str, pd.Series] = {}
    english: pd.Series | None = None

    for code in languages:
        paths = glob.glob(f"{root}/en-{code}/*.parquet")
        if not paths:
            raise FileNotFoundError(f"no en-{code} split in {root}")
        pair = f"en-{code}"
        raw = pd.read_parquet(paths[0])[pair]
        en_side = raw.map(lambda d: d["en"])
        frames[code] = raw.map(lambda d, c=code: d[c])
        if english is None:
            english = en_side
        elif not english.equals(en_side):
            raise ValueError(
                f"en-{code} has a different English side; FLORES is no longer "
                f"N-way parallel and the counterfactual is not content-matched"
            )

    out = pd.DataFrame(frames)
    out.insert(0, "en", english)
    return out


def usable_symbols(tokenizer: Any) -> list[str]:
    """Operator output symbols that are single-token for this tokenizer."""
    return single_token_subset(tokenizer, OPERATOR_SYMBOLS)


def usable_languages(tokenizer: Any, codes: list[str] | None = None) -> dict[str, str]:
    """Languages whose names are single-token here; ambiguous labels are excluded, not
        worked around, the lens being token-indexed.
    """
    codes = codes or list(LANGUAGE_NAMES)
    names = {c: LANGUAGE_NAMES[c] for c in codes if c in LANGUAGE_NAMES}
    keep = set(single_token_subset(tokenizer, list(names.values())))
    return {c: n for c, n in names.items() if n in keep}


@dataclass
class LanguageConfig:
    """Generation settings.

    Attributes:
        n_instances: Semantic instances; each yields 4 records.
        n_choices: Candidate language set size per instance.
        sentences_per_passage: Passage length, long enough to clear the lens's position
            floor.
        matched_legend: Show the operator table in *every* condition, so the gold label
            is
            printed in all arms or none. ``False`` is the literal reading of the design
            and a
            confound that inflates the effect in the predicted direction; ``True`` is
            conservative the other way, since a table present during the automatic arm
            may
            itself cue the model. Both are generated; matched is primary.
        matched_length: Instructions tokenising to the **same length** with an identical
            trailing span. Default ``False``, keeping what every published number used
            -- those
            are 13/12/14/12 tokens, harmless at *the* query position and fatal over a
            *span*,
            which is what made the attention-route measurement uninterpretable. With
            ``True``
            all arms are 14 tokens sharing a 9-token tail, so use ``--last-n 9``.
        seed: Language assignment, operator tables and distractors.
    """

    n_instances: int = 400
    n_choices: int = 4
    sentences_per_passage: int = 3
    matched_legend: bool = True
    matched_length: bool = False
    seed: int = 0


def generate(
    tokenizer: Any,
    config: LanguageConfig | None = None,
) -> list[Record]:
    """Matched automatic/report/flexible/control records sharing one passage verbatim;
        only the instruction changes, and the language name never appears in the
        passage.
    """
    config = config or LanguageConfig()
    rng = random.Random(config.seed)

    names = usable_languages(tokenizer)
    if len(names) < config.n_choices:
        raise ValueError(
            f"only {len(names)} single-token language names for this tokenizer, "
            f"need {config.n_choices}"
        )
    symbol_pool = usable_symbols(tokenizer)
    if len(symbol_pool) < config.n_choices:
        raise ValueError(
            f"only {len(symbol_pool)} single-token operator symbols for this "
            f"tokenizer, need {config.n_choices}. Add candidates to "
            f"OPERATOR_SYMBOLS."
        )
    codes = sorted(names)
    frame = load_parallel(codes)

    span = config.sentences_per_passage
    # One sentence is held back per instance as the true continuation, which
    # supplies the automatic condition's gold next token.
    max_start = len(frame) - span - 1
    if max_start < 1:
        raise ValueError(f"FLORES has too few rows ({len(frame)}) for span {span}")

    yes_no = ["Yes", "No"]
    for word in yes_no:
        single_token_id(tokenizer, word)

    records: list[Record] = []
    for i in range(config.n_instances):
        choices = rng.sample(codes, config.n_choices)
        gold_code = choices[0]
        rng.shuffle(choices)
        gold_name = names[gold_code]
        choice_names = [names[c] for c in choices]

        start = rng.randrange(1, max_start)
        rows = frame.iloc[start : start + span]
        passage = " ".join(rows[gold_code].tolist())
        next_row = frame.iloc[start + span]

        try:
            check_label_absent(passage, gold_name)
            for name in choice_names:
                check_label_absent(passage, name)
        except ValueError:
            continue  # a passage that names a language cannot be used

        instance_id = f"lang_{i:06d}"
        gold_token = single_token_id(tokenizer, gold_name)
        # Controls are the rival language names: same category, same frequency
        # band, same part of speech. The tightest available contrast.
        controls = [
            single_token_id(tokenizer, n) for n in choice_names if n != gold_name
        ]

        # The operator table. Under `matched_legend` it becomes shared context
        # carried by every condition, so label presence cannot differ between
        # arms; otherwise it stays inside the flexible instruction.
        symbols = rng.sample(symbol_pool, config.n_choices)
        table = dict(zip(choice_names, symbols, strict=True))
        table_text = "\n".join(f"{k} -> {v}" for k, v in table.items())
        operator_id = f"lookup_{rng.randrange(10_000):04d}"
        suffix = "matched" if config.matched_legend else "unmatched"
        if config.matched_length:
            suffix = f"{suffix}_len"
        wording = INSTRUCTIONS[config.matched_length]

        if config.matched_legend:
            context = f"Reference table:\n{table_text}\n\n{passage}"
            flexible_instruction = wording["flexible"]
        else:
            context = passage
            flexible_instruction = (
                f"Use this table:\n{table_text}\n" + wording["flexible"]
            )

        shared = dict(
            family="language",
            semantic_instance_id=instance_id,
            context=context,
            latent_name="language",
            latent_value=gold_name,
            latent_token_id=gold_token,
            control_token_ids=controls,
        )

        # --- Condition A: automatic. Continue the text; never name the language.
        # Candidates are the first token of the true continuation in each
        # candidate language, so the distractors are content-matched.
        group: dict[Condition, Record] = {}
        cont_tokens, cont_words = [], []
        for code in choices:
            first_word = str(next_row[code]).split()[0]
            ids = tokenizer.encode(f" {first_word}", add_special_tokens=False)
            cont_tokens.append(int(ids[0]))
            cont_words.append(first_word)
        if len(set(cont_tokens)) == len(cont_tokens):
            gold_idx = choices.index(gold_code)
            group[Condition.AUTOMATIC] = Record(
                id=f"{instance_id}_automatic",
                condition=Condition.AUTOMATIC,
                template_id=f"lang_continue_{suffix}",
                # The trailing "Answer:" is not cosmetic. Every other condition
                # ends that way, and a question-answering cue by itself lifts
                # workspace entry: the format-matched control read the gold
                # language at median rank 236 against the automatic arm's 2327
                # while needing the language for nothing at all. Without this the
                # contrast is partly measuring prompt format.
                instruction=wording["automatic"],
                gold_answer=cont_words[gold_idx],
                candidate_answers=cont_words,
                candidate_token_ids=cont_tokens,
                operator_family=None,
                operator_id=None,
                **shared,
            )

        # --- Condition B: explicit report.
        group[Condition.REPORT] = Record(
            id=f"{instance_id}_report",
            condition=Condition.REPORT,
            template_id=f"lang_report_{suffix}",
            instruction=wording["report"],
            gold_answer=gold_name,
            candidate_answers=choice_names,
            candidate_token_ids=[single_token_id(tokenizer, n) for n in choice_names],
            operator_family="report",
            operator_id="report_00",
            **shared,
        )

        # --- Condition C: flexible. The mapping is defined in the prompt, so the
        # answer cannot be retrieved from parametric knowledge -- z must be held
        # and then passed into an operator the model has never seen.
        group[Condition.FLEXIBLE] = Record(
            id=f"{instance_id}_flexible",
            condition=Condition.FLEXIBLE,
            template_id=f"lang_lookup_{suffix}",
            instruction=flexible_instruction,
            gold_answer=table[gold_name],
            candidate_answers=symbols,
            candidate_token_ids=[single_token_id(tokenizer, s) for s in symbols],
            operator_family="lookup",
            operator_id=operator_id,
            meta={"table": table},
            **shared,
        )

        # --- Condition D: matched control. Same instruction shape and a
        # single-token answer, but the language is irrelevant to it.
        has_question = "?" in passage
        group[Condition.CONTROL] = Record(
            id=f"{instance_id}_control",
            condition=Condition.CONTROL,
            template_id=f"lang_control_{suffix}",
            instruction=wording["control"],
            gold_answer="Yes" if has_question else "No",
            candidate_answers=yes_no,
            candidate_token_ids=[single_token_id(tokenizer, w) for w in yes_no],
            operator_family="control",
            operator_id="control_00",
            **shared,
        )

        # --- Condition E: supplied. The operator is present but its input is given,
        # so the arm needs the composition without needing the inference. The value
        # supplied is a *dummy* -- a candidate other than the passage's language --
        # because naming the gold one would put the measured concept in the prompt
        # and pin its rank at ~1 by copying (trap 2).
        dummies = [n for n in choice_names if n != gold_name]
        if dummies:
            # Drawn from an *independent* stream keyed on the instance index. Using
            # the main `rng` here would consume draws mid-loop and shift every later
            # instance's language, passage and operator table -- silently changing a
            # dataset that every published number was measured on.
            dummy = dummies[
                random.Random(config.seed * 1_000_003 + i).randrange(len(dummies))
            ]
            group[Condition.SUPPLIED] = Record(
                id=f"{instance_id}_supplied",
                condition=Condition.SUPPLIED,
                template_id=f"lang_supplied_{suffix}",
                instruction=wording["supplied"].format(dummy=dummy),
                gold_answer=table[dummy],
                candidate_answers=symbols,
                candidate_token_ids=[single_token_id(tokenizer, s) for s in symbols],
                operator_family="lookup",
                operator_id=operator_id,
                meta={"table": table, "dummy": dummy},
                **shared,
            )

        if config.matched_legend:
            check_label_symmetry(group)
        # In the unmatched variant the asymmetry is deliberate -- it is the
        # artifact being quantified -- so the invariant is not applied. Nothing
        # from this variant may be a primary result.
        records.extend(group.values())

    return records
