"""Shared scaffolding for the command-line entry points.

Every experiment CLI is the same five blocks around a different middle: parse the
standard flags, select the instances complete in the arms it contrasts, load the
model and its lens, run, then write observations and a pooled summary. Only the
middle differs, so everything else lives here once.

That is not tidiness. Each of these blocks has a way of going quietly wrong, and a
copy of it in ten files is ten places for the guard to be missing:

* an artifact written without its ``args`` is a number nobody can re-derive
  (:func:`write`);
* an artifact written without its position mode cannot be split by that mode
  afterwards, which has already forced one reconstruction from row order
  (:func:`query_span`, trap 16);
* a readout at a layer the lens never fitted returns plausible numbers
  (:func:`readout_layers`);
* an arm that ends up empty reports a clean null that means nothing
  (:func:`instances`).

Nothing here decides anything scientific. The guards that do
(:func:`innerj.model.check_positions`, the label-symmetry checks, the verdict
functions) stay where they are and are called explicitly.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import torch

from innerj import config, console
from innerj.model import QWEN27B, band, load_lens, load_model
from innerj.patch import Component
from innerj.positions import build, describe
from innerj.tasks.base import Condition, Record, complete_instances, read_jsonl

DATA_ROOT = config.DATA_ROOT


#: The flags that recur across entry points, in the order they should appear.
#: Anything specific to one experiment is declared by that CLI instead.
SHARED_FLAGS: dict[str, tuple[tuple[str, ...], dict[str, Any]]] = {
    "records": (("--records",), {"required": True, "help": "dataset JSONL"}),
    "model": (("--model",), {"default": QWEN27B.model_name}),
    "lens": (("--lens",), {"default": QWEN27B.lens_path}),
    "device": (("--device",), {"default": "cuda"}),
    "lens_device": (("--lens-device",), {
        "default": None,
        "help": "keep the Jacobians resident here (default: alongside the model)",
    }),
    "pairs": (("--pairs",), {"type": int, "default": 60,
                             "help": "donor/target pairs"}),
    "last_n": (("--last-n",), {"type": int, "default": 12,
                               "help": "patched positions"}),
    "seed": (("--seed",), {"type": int, "default": 0}),
    "max_seq_len": (("--max-seq-len",), {"type": int, "default": 512}),
    # Not required: every CLI supplies its own default via set_defaults, which
    # argparse honours for the value but would not clear a required flag.
    "tag": (("--tag",), {"help": "artifact name prefix"}),
}


def parser(description: str, *, needs: tuple[str, ...] = ()) -> argparse.ArgumentParser:
    """An argument parser carrying whichever standard flags this CLI needs.

    ``needs`` selects from :data:`SHARED_FLAGS`. Anything specific to one
    experiment is added by the caller afterwards, so a reader can tell at a glance
    which arguments are shared and which are the experiment's own.

    An unknown key raises. So does a bare string, which is the failure this guard
    exists for: ``needs=("device")`` is not a tuple, and ``"model" in "device"``
    is merely False, so a missing comma used to drop every flag silently and the
    CLI ran with defaults it never declared.
    """
    if isinstance(needs, str):
        raise TypeError(f"needs must be a tuple, got the string {needs!r}")
    if unknown := set(needs) - set(SHARED_FLAGS):
        raise KeyError(
            f"unknown shared flag(s) {sorted(unknown)}; "
            f"available: {sorted(SHARED_FLAGS)}"
        )
    p = argparse.ArgumentParser(description=description)
    for name, (flags, options) in SHARED_FLAGS.items():
        if name in needs:
            p.add_argument(*flags, **options)
    return p


def load(args: argparse.Namespace, *, lens: bool = True) -> tuple[Any, Any]:
    """Load the checkpoint and, unless told otherwise, its lens.

    Announces what was loaded, because a run against the wrong lens produces
    entirely plausible numbers and the artifact is the only other place that
    records which one it was.

    Honours ``--dtype`` when the caller declares it. The default is bf16, but an
    exactness check has to run in fp32: the same construction in bf16 drifts a few
    percent over 64 layers from accumulation alone and looks like a broken
    derivation rather than a rounding floor.
    """
    dtype = getattr(args, "dtype", None)
    console.step(f"loading {args.model}" + (f" ({dtype})" if dtype else ""))
    model = load_model(
        args.model,
        device=args.device,
        **({"dtype": getattr(torch, dtype)} if dtype else {}),
    )
    console.detail(f"{model.n_layers} layers")
    if not lens:
        return model, None
    # Resident beside the model by default: transport() copies the Jacobian to
    # the residual's device on every call, which otherwise dominates every sweep.
    fitted = load_lens(
        args.lens, device=getattr(args, "lens_device", None) or args.device
    )
    console.detail(
        f"lens {Path(args.lens).name} "
        f"source layers {fitted.source_layers[0]}..{fitted.source_layers[-1]}, "
        f"n_prompts={fitted.n_prompts}"
    )
    return model, fitted


def readout_layers(model, lens, requested: list[int] | None = None) -> list[int]:
    """Band layers that the lens actually fitted, refusing an empty intersection.

    ``apply()`` will happily read a layer the lens never fitted and return
    numbers, so the intersection is taken here and an empty one is fatal rather
    than silent.
    """
    layers = [
        layer for layer in (requested or band(model.n_layers))
        if layer in set(lens.source_layers)
    ]
    if not layers:
        raise SystemExit(
            f"no requested layer is fitted in this lens (fitted: "
            f"{lens.source_layers[0]}..{lens.source_layers[-1]})"
        )
    console.detail(f"reading {len(layers)} layers, {layers[0]}..{layers[-1]}")
    return layers


def parse_component(text: str) -> Component:
    """``attn.L39`` / ``attn.L39.H7`` / ``mlp.L46`` / ``resid.L40``."""
    parts = text.split(".")
    if len(parts) < 2 or not parts[1].startswith("L"):
        raise argparse.ArgumentTypeError(f"cannot parse component {text!r}")
    head = None
    if len(parts) > 2 and parts[2] != "all":
        head = int(parts[2].lstrip("H"))
    return Component(parts[0], int(parts[1][1:]), head)


def instances(
    args: argparse.Namespace,
    arms: tuple[Condition, ...],
    *,
    limit: int | None = None,
) -> dict[str, dict[Condition, Record]]:
    """Instances present in every one of ``arms``, restricted to those arms.

    ``arms`` means *these arms*, not *at least these*. Every dataset here carries
    all four conditions, so returning whatever an instance happens to have would
    silently add a fourth row to a three-arm design --- which it did: an ablation
    over flexible/report/control reported an automatic arm too.

    An instance missing from one arm would unbalance a comparison that is only
    meaningful paired, and an empty result reports a clean null that means
    nothing --- so the empty case raises rather than returning ``{}``.

    ``limit`` defaults to ``--pairs`` where the CLI declares it. A limit of ``0``
    or ``None`` means no cap. Selection is by sorted instance id, never by
    sampling: two CLIs run over the same cap must see the same instances or their
    results cannot be compared.
    """
    groups = complete_instances(list(read_jsonl(args.records)), arms)
    if not groups:
        raise SystemExit(
            f"no semantic instance covers all of {[str(a) for a in arms]} in "
            f"{Path(args.records).name}. An empty arm reports a clean null that "
            f"means nothing."
        )
    cap = limit if limit is not None else getattr(args, "pairs", None)
    kept = {
        i: {arm: groups[i][arm] for arm in arms} for i in sorted(groups)[: cap or None]
    }
    console.step(
        f"{len(kept)} instances complete in {'/'.join(str(a) for a in arms)}"
        + (f" (of {len(groups)})" if len(kept) < len(groups) else "")
    )
    return kept


def query_span(args: argparse.Namespace) -> tuple[list[int], str]:
    """The final ``--last-n`` prompt positions, with the label to stamp on rows.

    The label comes from :func:`innerj.positions.build` rather than being typed, so
    it cannot drift from the format the position-mode CLIs record. Every
    observation carries it: an artifact that omits the position mode cannot be
    re-analysed by it, which once forced a reconstruction from row order.
    """
    positions = list(range(-args.last_n, 0))
    label = describe(build("query", args.last_n))
    console.detail(f"patching {label} ({positions})")
    return positions, label


def _plain(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    return value


def write(
    subdir: str, stem: str, payload: dict[str, Any], *, args: argparse.Namespace
) -> Path:
    """Write a result artifact under the data root, always stamped with ``args``.

    Every artifact carries the arguments that produced it. That is the difference
    between a number one can re-derive and a number one can only trust.
    """
    out = DATA_ROOT / subdir
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{stem}.json"
    path.write_text(
        json.dumps({"args": vars(args), **payload}, indent=2, default=_plain)
    )
    console.wrote(path)
    return path


def write_lines(subdir: str, stem: str, rows: list[Any]) -> Path:
    """Write per-observation rows as JSONL beside their summary.

    ``default=_plain`` rather than ``default=str`` so a nested dataclass lands as
    a JSON object. Stringified, an ``Estimate`` becomes the prose
    ``"+0.0891 [+0.0799, +0.0983]"``, which ``innerj audit`` cannot see as a
    number and which no downstream re-analysis can parse.
    """
    out = DATA_ROOT / subdir
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{stem}.jsonl"
    path.write_text("\n".join(json.dumps(_plain(r), default=_plain) for r in rows))
    console.wrote(path)
    return path


def save(
    subdir: str,
    args: argparse.Namespace,
    *,
    observations: list[Any],
    results: list[Any],
    **extra: Any,
) -> Path:
    """The standard artifact pair: ``<stem>_observations.jsonl`` and
    ``<stem>_results.json``.

    Both are written, always together and always under the same stem. The
    observations are the evidence and the results are one pooling of it, so a
    results file without its observations cannot be re-pooled --- and re-pooling is
    the only way to correct a summary written by code that has since changed, which
    this project has had to do.

    ``**extra`` goes into the results file beside ``results``: the layers read, the
    pair count, whatever else the experiment needs to be interpretable.
    """
    name = stem(args, dataset=False)
    write_lines(subdir, f"{name}_observations", observations)
    return write(
        subdir,
        f"{name}_results",
        {**extra, "results": [_to_dict(r) for r in results]},
        args=args,
    )


def _to_dict(result: Any) -> Any:
    """Prefer a result's own ``to_dict``: several add a verdict or a derived field
    that ``asdict`` alone would drop."""
    return result.to_dict() if hasattr(result, "to_dict") else _plain(result)


def stem(args: argparse.Namespace, *extra: str, dataset: bool = True) -> str:
    """``tag_model_dataset`` --- the naming every artifact on disk already uses.

    ``dataset=False`` drops the records name. The patching experiments have always
    written ``tag_model``, and :mod:`innerj.figures.build` opens four of those by
    exact filename, so a re-run must land on the name it landed on before.
    """
    parts = [args.tag, args.model.split("/")[-1]]
    if dataset and getattr(args, "records", None):
        parts.append(Path(args.records).stem)
    parts.extend(extra)
    return "_".join(p for p in parts if p)
