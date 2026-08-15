<div align="center">

![innerJ](assets/innerj-banner.svg)

# innerJ

**A latent variable does not reach a language model's self-report because a gate opens. It gets there because attention carries it.**

</div>

## Overview

A model computes plenty it never says. Ask it to *use* the language of a passage and it will; ask it to *name* that language and something different has to happen first. Recent work identifies the directions a model can report on, a *verbalizable workspace*, and shows a latent quantity is more present there under flexible task demand, leaving the mechanism open ([Gurnee et al., 2026](https://transformer-circuits.pub/2026/workspace/index.html)). The vocabulary invites a guess: contents are *admitted*, so admission implies a gate.

We looked for that gate on an open-weight model, at the position the account predicts, and it is not there. What we find instead is **transport**:

- **Availability is not the variable.** One shared linear probe decodes the latent in every arm, including the arm needing it for nothing, at 6.4-9.0x its corrected floor. Demand changes visibility, not presence.
- **Attention does the carrying.** At matched readout distance, transport concentrates in a mid-depth window by >=17x over anywhere shallower. No tested MLP contributes positively inside it.

This repository is the benchmark (**JGateBench**) and the causal toolkit. The paper is not in it; `innerj audit` checks every number in the paper against the artifact that produced it.

## Installation

```bash
uv sync --extra dev
git clone https://github.com/anthropics/jacobian-lens.git 
uv pip install -e ./jacobian-lens
uv run pytest -n 16
```

> Python 3.13+, torch 2.13, transformers 5.14.1, and one GPU with ~60 GB for the 27B primary checkpoint. Every path is derived in `innerj/config.py`, so there is nothing to configure. 

## Data

Each semantic instance yields five prompts over an **identical context**, so every contrast is within-instance and paired. `flexible - control` is the primary one: same `"Answer:"` format, no demand for the variable.

| Condition | Demand on `z` |
|---|---|
| `automatic` | must be used, never exposed |
| `report` | must be stated |
| `flexible` | must be passed into an operator the prompt defines |
| `control` | matched instruction needing `z` for nothing |
| `supplied` | same operator, value *given* - composition without inference |

Two families ship in `data/`: **language identity** on FLORES-200, which is *N*-way parallel so the counterfactual varies the latent variable and nothing else, and **object tracking** under swaps, where `z` is a progressively updated state. The primary four-arm language set is 1954 records over 400 semantic instances.

Language records are committed as **stubs** — every authored field, with the FLORES-200 passage stored as a language code and row offset. Rebuild byte-identically, pinned to revision `8ecaf1bb`:

```bash
hf download haoranxu/FLORES-200 --repo-type dataset
innerj flores --rehydrate data/language/*.jsonl --out-dir data/language-full
```

`tests/test_flores_ref.py` pins the rebuild to its published SHA-256 (`00623f40...`) rather than asserting it.

## Running

One `innerj` command; `python -m innerj.cli <command>` is equivalent and is the form every published number was produced with.

```bash
innerj build-dataset --family language --n 400          # CPU, seconds
innerj stage1 --records data/language-full/Qwen3.6-27B_matched_n400_s0.jsonl --limit 200
innerj --help                                           # the rest, in the order they run
```

The remaining commands are one per causal step — `probe`, `screen`, `sweep`, `attention`, `specificity`, `counterfactual`, `ablate`, `mediate` — plus `fit-lens`, `gauge`, `example`, `figures` and `audit`. Each writes a `_results.json` **and** an `_observations.jsonl`, so any result can be re-pooled without re-running the GPU pass. A full replication is ~1.3 GB.

## Results

Qwen3.6-27B, 64 layers, band L24-L59. Intervals are 95% bootstrap over 10,000 resamples clustered on the semantic instance; `excludes_zero` is the only significance claim made anywhere.

| Finding | Measure |
|---|---|
| Flexible demand raises visibility | `ΔR_z = +0.089 [+0.080, +0.098]`, against a control matched on format **and** accuracy |
| Part of that demand is the operator, not the variable | the `supplied` arm shifts **43%** as much without inferring anything |
| Transport is localised | **>=17x** the largest cell anywhere shallower, over four donor pairings |
| Half the behaviour runs through one lens direction | projecting it out costs **~50%** of the counterfactual effect, over four controls |
| A readout shift does not measure use | three components at one layer shift the readout to within **12%** of each other and differ **7.4x** in behaviour |

**What does not generalise, stated up front.** The *concentration*. On language a single L39 head matches its own block and the whole residual stream to four decimals; on tracking the two attention blocks are comparable and no head dominates. The head is a case study, never the mechanism. Under percentile rank rather than log-rank the same grid does not localise at all — on a 248,320-token vocabulary `R_z` separates rank 1 from rank 25 by 0.0001 — so every transport result is reported under both, and they place the depth peak fifteen layers apart.

## Project structure

```text
innerj/model.py patch.py positions.py   loading, the workspace band, capture and patching
innerj/analysis/                        R_z, L_z, M_z, forced choice, bootstrap, FDR
innerj/experiments/                     one module per causal experiment
innerj/tasks/                           generators, with the invariants as constructor checks
innerj/figures/                         the deck and its style, built from artifacts only
tests/                                  the invariants whose violation is silent
data/language/ data/tracking/           JGateBench records
```

## License

MIT ([LICENSE](LICENSE)).
