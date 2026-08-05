"""Where to patch, resolved per record.

The component-level negative result came from patching the last 12 tokens -- the
instruction and query region. But the latent variable is *constructed* while the
passage is read, so a write mechanism may operate there and nowhere near the query.
Patching only the query region would then be looking in the wrong place, and the
absence of an effect would say nothing about the mechanism.

Two position modes, and the choice is a first-class experimental axis rather than a
detail:

* :func:`last_n` -- the final ``k`` tokens of the prompt. The query region: where the
  operator is applied and the answer is produced.
* :func:`context_end` -- the final ``k`` tokens of the *context*, i.e. the end of the
  passage, before any instruction. Where the latent variable has just been
  established.

Donor and target are different instances with different passage lengths, so absolute
indices do not correspond between them. Positions are therefore resolved per record
and the two runs use different absolute indices for the same *semantic* location.
Both yield ``k`` positions, so a captured activation is shape-compatible with the
site it is written into.
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

    The context is verified to be a token-level *prefix* of the prompt rather than
    assumed to be one. Byte-pair merges can straddle the boundary between the
    context and the instruction that follows it, and a silently wrong boundary would
    place every passage-position patch a few tokens off -- which looks like a weak
    effect, not like a bug.
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
    """The final ``k`` context tokens: the end of the passage.

    Raises if the passage is too short to supply ``k`` tokens above the lens's
    unfitted position floor, rather than quietly reading positions the lens never
    saw.
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


def describe(fn: PositionFn) -> str:
    """Human-readable label for an artifact record."""
    return getattr(fn, "_label", fn.__qualname__)


def labelled(fn: PositionFn, label: str) -> PositionFn:
    fn._label = label  # type: ignore[attr-defined]
    return fn


def build(mode: str, k: int) -> PositionFn:
    """``"query"`` or ``"passage"``, with a span of ``k`` tokens."""
    if mode == "query":
        return labelled(last_n(k), f"query:last{k}")
    if mode == "passage":
        return labelled(context_end(k), f"passage:last{k}")
    raise ValueError(f"unknown position mode {mode!r}; use 'query' or 'passage'")
