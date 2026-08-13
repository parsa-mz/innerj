"""Shared scaffolding for the command-line entry points.

Every experiment CLI is the same five blocks around a different middle: parse flags,
select
instances, load model and lens, run, write observations and a summary. Each block has a
way
of going quietly wrong, and a copy in ten files is ten places for the guard to be
missing --
an artifact without its ``args`` cannot be re-derived; one without its position mode
cannot
be split by it (trap 16); a readout at an unfitted layer returns plausible numbers; an
empty
arm reports a meaningless null.

**Which exception:** ``experiments/`` raises ``ValueError`` -- a library, the caller
decides
-- and ``cli/`` raises ``SystemExit`` with a message a person can act on.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import torch

from innerj import config, console
from innerj.model import QWEN27B, band, load_lens, load_model, model_slug
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
    """A parser carrying whichever of :data:`SHARED_FLAGS` this CLI needs.

    An unknown key raises, and so does a bare string: ``needs=("device")`` is not a
    tuple, and
    ``"model" in "device"`` is merely False, so a missing comma used to drop every flag.
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

    Announces what was loaded, since a run against the wrong lens produces plausible
    numbers.
    ``--dtype`` matters because an exactness check must run in fp32.
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
    """Band layers the lens actually fitted; an empty intersection is fatal, because
        ``apply()`` will read an unfitted layer and return numbers.
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

    ``arms`` means *these arms*, not *at least these* -- returning whatever an instance
    has once
    added a fourth row to a three-arm design. Empty raises. Selection is by sorted
    instance id,
    never sampling, so two CLIs at the same cap see the same instances.
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
    """The final ``--last-n`` positions, with the label to stamp on rows.

    The label comes from :func:`innerj.positions.build` rather than being typed, so it
    cannot
    drift from what the position-mode CLIs record.
    """
    positions = list(range(-args.last_n, 0))
    label = describe(build("query", args.last_n))
    console.detail(f"patching {label} ({positions})")
    return positions, label


def _plain(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    return value


def artifact(subdir: str, stem: str, kind: str = "results") -> Path:
    """The one place an artifact path is composed, so renaming a stem breaks the build
        loudly rather than at figure time.
    """
    suffix = {
        "results": ".json",
        "observations": "_observations.jsonl",
        "raw": ".jsonl",
    }[kind]
    return DATA_ROOT / subdir / f"{stem}{suffix}"


def write(
    subdir: str, stem: str, payload: dict[str, Any], *, args: argparse.Namespace
) -> Path:
    """Write a result artifact under the data root, always stamped with ``args`` -- the
        difference between a number one can re-derive and one can only trust.
    """
    path = artifact(subdir, stem, "results")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"args": vars(args), **payload}, indent=2, default=_plain)
    )
    console.wrote(path)
    return path


def write_lines(subdir: str, stem: str, rows: list[Any]) -> Path:
    """Per-observation rows as JSONL beside their summary.

    ``default=_plain`` rather than ``default=str``: stringified, an ``Estimate`` becomes
    the
    prose ``"+0.0891 [+0.0799, +0.0983]"``, which the audit cannot see as a number.
    """
    path = artifact(subdir, stem, "raw")
    path.parent.mkdir(parents=True, exist_ok=True)
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
    """The standard pair: ``<stem>_observations.jsonl`` and ``<stem>_results.json``.

    Always together under one stem. A results file without its observations cannot be
    re-pooled, and re-pooling is the only way to correct a summary written by code that
    has
    since changed.
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
    """Prefer a result's own ``to_dict``: several add a field ``asdict`` would drop."""
    return result.to_dict() if hasattr(result, "to_dict") else _plain(result)


def stem(args: argparse.Namespace, *extra: str, dataset: bool = True) -> str:
    """``tag_model_dataset``, the naming every artifact already uses.

    ``dataset=False`` drops the records name: the patching experiments write
    ``tag_model`` and
    the figure module opens four by exact filename, so a re-run must land on the same
    name.
    """
    parts = [args.tag, model_slug(args.model)]
    if dataset and getattr(args, "records", None):
        parts.append(Path(args.records).stem)
    parts.extend(extra)
    return "_".join(p for p in parts if p)
