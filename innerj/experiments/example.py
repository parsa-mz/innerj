"""A worked example: what the J-lens actually reads at the query position.

The paper's dependent variable is $R_z$, the percentile rank of the gold concept
among the vocabulary. That is precise and completely opaque on first reading. This
dumps, for **one** semantic instance, what the readout literally contains: the
top-$k$ tokens at the query position at each band layer, and the gold language's
rank, in every one of the four conditions.

It reads through ``lens.apply`` -- the same call ``innerj.entry`` uses -- so the
ranks here are the same quantity the paper reports, on the same forward pass shape,
not a re-derivation that could drift.

One thing the figure must not be allowed to imply. Under the matched legend the
operator table names every candidate language in *all four* arms, so the gold token
is present in every prompt and its absolute rank is inflated everywhere. The
comparison across conditions is unaffected because the inflation is symmetric, but
the absolute levels are not evidence of unprompted representation. Hence the figure
reports **rank per condition side by side** rather than one arm's ranks alone.

Usage:
    innerj example --records <jsonl> --instance lang_000000
"""

from __future__ import annotations

import torch

from innerj.analysis.readout import concept_rank, percentile_rank
from innerj.model import check_positions
from innerj.tasks.base import Condition

ORDER = [Condition.CONTROL, Condition.AUTOMATIC, Condition.REPORT, Condition.FLEXIBLE]


@torch.no_grad()
def read_instance(model, lens, tokenizer, record, *, layers: list[int], top_k: int,
                  max_seq_len: int = 512) -> dict:
    """Top-$k$ tokens and the gold token's rank at the query position, per layer."""
    input_ids = model.encode(record.prompt, max_length=max_seq_len)
    seq_len = int(input_ids.shape[1])
    query = seq_len - 1
    check_positions([query], seq_len)

    lens_logits, model_logits, _ = lens.apply(
        model, record.prompt, layers=layers, positions=[query],
        max_seq_len=max_seq_len,
    )
    rows = []
    for layer in layers:
        logits = lens_logits[layer][0].float().cpu()
        top = torch.topk(logits, top_k)
        rows.append(
            {
                "layer": layer,
                "rank": concept_rank(logits, record.latent_token_id),
                "r_z": percentile_rank(logits, record.latent_token_id),
                "top_tokens": [tokenizer.decode([int(i)]) for i in top.indices],
            }
        )
    winner = max(
        record.candidate_token_ids, key=lambda t: float(model_logits[0][t])
    )
    answers = dict(zip(record.candidate_token_ids, record.candidate_answers,
                       strict=True))
    return {
        "condition": str(record.condition),
        "instruction": record.instruction,
        "seq_len": seq_len,
        "answer": answers[winner],
        "correct": answers[winner] == record.gold_answer,
        "layers": rows,
    }
