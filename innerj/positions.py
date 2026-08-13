"""Where to patch, resolved per record.

The latent variable is *constructed* while the passage is read, so a write mechanism may
operate there and nowhere near the query; patching only the query region would be
looking in
the wrong place. The mode is therefore a first-class axis: :func:`last_n` takes the
final
``k`` prompt tokens, :func:`context_end` the final ``k`` *context* tokens.

Donor and target have different passage lengths, so positions are resolved per record
and the
two runs use different absolute indices for the same *semantic* site. Both yield ``k``
positions, so a captured activation is shape-compatible with where it is written.
"""

from __future__ import annotations

from collections.abc import Callable

from jlens.hf import HFLensModel

from innerj.model import MIN_FIT_POSITION
from innerj.tasks.base import Record

PositionFn = Callable[[HFLensModel, Record], list[int]]


def context_length(
    model: HFLensModel, record: Record, *, max_seq_len: int = 512
) -> int:
    """Number of prompt tokens belonging to the context.

    Verified to be a token-level *prefix* rather than assumed: byte-pair merges can
    straddle the
    boundary, and a wrong one looks like a weak effect, not a bug.
    """
    prompt_ids = model.encode(record.prompt, max_length=max_seq_len)[0].tolist()
    context_ids = model.encode(record.context, max_length=max_seq_len)[0].tolist()
    if prompt_ids[: len(context_ids)] != context_ids:
        raise ValueError(
            f"{record.id}: the context is not a token prefix of the prompt, so the "
            f"passage boundary cannot be located by prefix length. Tokenizer merges "
            f"straddle the join."
        )
    return len(context_ids)


def last_n(k: int) -> PositionFn:
    """The final ``k`` prompt tokens: the query region."""

    def resolve(model: HFLensModel, record: Record) -> list[int]:
        return list(range(-k, 0))

    return resolve


def context_end(k: int) -> PositionFn:
    """The final ``k`` context tokens; raises if the passage cannot supply ``k`` above
    the
        lens's position floor.
    """

    def resolve(model: HFLensModel, record: Record) -> list[int]:
        n_context = context_length(model, record)
        start = n_context - k
        if start < MIN_FIT_POSITION:
            raise ValueError(
                f"{record.id}: context is {n_context} tokens, so the last {k} start "
                f"at {start}, below the fitted floor {MIN_FIT_POSITION}. Use a "
                f"shorter span or longer passages."
            )
        return list(range(start, n_context))

    return resolve


def context_and_query(k: int) -> PositionFn:
    """The last ``k`` context tokens *and* the last ``k`` prompt tokens.

    Turns the repair account into a prediction: if a value installed below the window
    fails to
    survive because the layers above re-derive the target's own from the unpatched
    passage, then
    patching both should raise survival -- and not merely because more tokens were
    patched, which
    a span-matched query-only arm controls for.

    Order is passage-then-query and identical for donor and target, which is what makes
    the rows
    line up; ``capture`` and ``run_patched`` never sort.
    """

    passage, query = context_end(k), last_n(k)

    def resolve(model: HFLensModel, record: Record) -> list[int]:
        n_prompt = int(model.encode(record.prompt, max_length=512).shape[1])
        n_context = context_length(model, record)
        # Patching one index twice with two different activations writes whichever
        # hook ran last and reports the span it did not apply. Silent, and it would
        # look like a weak effect.
        if n_context > n_prompt - k:
            raise ValueError(
                f"{record.id}: the context ends at {n_context} of {n_prompt} prompt "
                f"tokens, so the last {k} context positions overlap the last {k} "
                f"prompt positions. Use a shorter span or a longer instruction."
            )
        return [*passage(model, record), *query(model, record)]

    return resolve


def describe(fn: PositionFn) -> str:
    """Human-readable label for an artifact record."""
    return getattr(fn, "_label", fn.__qualname__)


def labelled(fn: PositionFn, label: str) -> PositionFn:
    fn._label = label  # type: ignore[attr-defined]
    return fn


def build(mode: str, k: int) -> PositionFn:
    """``"query"``, ``"passage"`` or ``"both"``, with a span of ``k`` tokens."""
    if mode == "query":
        return labelled(last_n(k), f"query:last{k}")
    if mode == "passage":
        return labelled(context_end(k), f"passage:last{k}")
    if mode == "both":
        return labelled(context_and_query(k), f"both:last{k}")
    raise ValueError(
        f"unknown position mode {mode!r}; use 'query', 'passage' or 'both'"
    )


#: Where every experiment *reads*, as opposed to where it patches. The patch span is a
#: first-class axis; the readout position is not, because the measurement is pinned to
#: the retrieval cue -- trap 7. One definition, because it was written four ways and
#: omitted in ``sweep``; see ``scratchpad/deadcode_audit.md`` §3.1.
READ_AT = [-1]


