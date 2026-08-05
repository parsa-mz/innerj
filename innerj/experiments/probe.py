"""The availability leg of Stage 1: is ``z`` linearly decodable regardless of demand?

Stage 1's entry contrast shows the concept is *more* accessible in J-space under
flexible demand. On its own that is compatible with a dull reading: maybe the
model only computes ``z`` when the task asks for it, and workspace entry is just
"the model knows ``z``" wearing a lens.

The probe rules that out. If a linear probe on the full residual stream decodes
``z`` at the same accuracy in *every* condition -- including the control arm,
which needs ``z`` for nothing -- while ``R_z`` moves by a large margin, then the
information is equally present throughout and only its *broadcast* is gated. That
is the dissociation, and it is what makes a write mechanism the thing to look for.

Two design points, both load-bearing:

* **Held-out instances, never held-out positions.** Records from one semantic
  instance share a passage, so a random record-level split leaks the passage
  across the boundary and the probe scores its own training data.
* **A probe trained in one condition is tested in the others.** If the same linear
  direction reads ``z`` across arms, availability is genuinely shared rather than
  four separately-learned encodings.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch
from jlens.hf import HFLensModel
from jlens.hooks import ActivationRecorder

from innerj.model import check_positions
from innerj.tasks.base import Record


@torch.no_grad()
def cache_residuals(
    model: HFLensModel,
    records: list[Record],
    layers: list[int],
    *,
    max_seq_len: int = 512,
) -> np.ndarray:
    """Residuals at each record's query position: ``[n_records, n_layers, d_model]``.

    Stored fp16 to keep a full family in memory; the probe standardises before
    fitting, so the precision loss is immaterial next to the between-class
    distances it has to resolve.
    """
    out = np.empty((len(records), len(layers), model.d_model), dtype=np.float16)
    for i, record in enumerate(records, 1):
        input_ids = model.encode(record.prompt, max_length=max_seq_len)
        seq_len = int(input_ids.shape[1])
        query = seq_len - 1
        check_positions([query], seq_len)
        with ActivationRecorder(model.layers, at=layers) as recorder:
            model.forward(input_ids)
            for j, layer in enumerate(layers):
                out[i - 1, j] = (
                    recorder.activations[layer][0, query].detach().float().cpu().numpy()
                )
    return out


def _fit_logistic(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    *,
    n_classes: int,
    epochs: int = 400,
    lr: float = 0.05,
    weight_decay: float = 1e-2,
    seed: int = 0,
) -> float:
    """Multinomial logistic regression; returns held-out accuracy.

    Features are standardised on the *training* split only. Standardising on the
    pooled data would leak the test distribution into the fit.
    """
    torch.manual_seed(seed)
    mean = x_train.mean(0, keepdim=True)
    std = x_train.std(0, keepdim=True).clamp_min(1e-6)
    x_train = (x_train - mean) / std
    x_test = (x_test - mean) / std

    model = torch.nn.Linear(x_train.shape[1], n_classes).to(x_train.device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = torch.nn.CrossEntropyLoss()
    for _ in range(epochs):
        optimiser.zero_grad()
        loss_fn(model(x_train), y_train).backward()
        optimiser.step()

    with torch.no_grad():
        predicted = model(x_test).argmax(1)
        return float((predicted == y_test).float().mean())


@dataclass
class ProbeResult:
    """Held-out decoding accuracy for one (layer, train arm, test arm) cell."""

    layer: int
    train_condition: str
    test_condition: str
    accuracy: float
    chance: float
    n_train: int
    n_test: int
    n_classes: int
    #: Split index. The reported figure is a mean over seeds, not one split: a
    #: single split of 80 test instances moves by several points between seeds,
    #: which is wider than some of the differences the text draws.
    seed: int = 0
    #: ``"joint"`` for the single cross-arm probe, ``"per_arm"`` for the grid.
    kind: str = "per_arm"
    #: True when the labels were permuted across instances. These rows are the
    #: empirical floor, not a result -- see :func:`summarise`.
    shuffled: bool = False

    @property
    def above_chance(self) -> float:
        return self.accuracy - self.chance


def instance_split(
    records: list[Record], *, train_frac: float = 0.6, seed: int = 0
) -> tuple[set[str], set[str]]:
    """Split semantic instances, not records.

    Conditions of one instance share a passage verbatim, so splitting records
    would put the same passage on both sides and the probe would score memorised
    text.
    """
    instances = sorted({r.semantic_instance_id for r in records})
    rng = np.random.default_rng(seed)
    rng.shuffle(instances)
    cut = int(len(instances) * train_frac)
    if cut == 0 or cut == len(instances):
        raise ValueError(
            f"train_frac={train_frac} leaves an empty split over "
            f"{len(instances)} instances"
        )
    return set(instances[:cut]), set(instances[cut:])


def probe_grid(
    residuals: np.ndarray,
    records: list[Record],
    layers: list[int],
    *,
    train_frac: float = 0.6,
    seed: int = 0,
    device: str = "cuda",
    min_per_class: int = 3,
    center_per_condition: bool = False,
    shuffle_labels: bool = False,
) -> list[ProbeResult]:
    """Train a probe per (layer, condition) and evaluate it in every condition.

    The diagonal answers "is ``z`` decodable in this arm at all". The off-diagonal
    answers the stronger question: does *one* linear direction read ``z``
    regardless of what the task asks for.

    ``center_per_condition`` subtracts each condition's own mean residual before
    fitting. Without it, a cross-arm score conflates two different things: whether
    the ``z``-direction is shared, and whether the arms simply sit in different
    regions of activation space. The prompts differ in format, so they do sit
    apart, and an uncentred transfer number understates sharing. Report both --
    but the centring is estimated on the **training instances only**. Estimating it
    over all records lets the test arm's own mean inform the transform applied to
    the test arm, which is leakage; it is worth 0.014 of transfer accuracy here, in
    our favour, so the honest version is also the better one.

    ``shuffle_labels`` permutes ``z`` across semantic instances, keeping an
    instance's four arms consistent. Those rows are the empirical floor for
    whatever statistic is reported downstream. This matters more than it sounds:
    the headline number is a *maximum over twelve layers*, and a maximum over
    twelve noisy draws does not sit at 1/n_classes. Measured here, the nominal
    floor is 0.05 and the best-of-twelve floor is 0.093.
    """
    if residuals.shape[0] != len(records):
        raise ValueError(
            f"{residuals.shape[0]} cached residuals for {len(records)} records"
        )

    label_names = sorted({r.latent_value for r in records})
    label_of = {name: i for i, name in enumerate(label_names)}
    labels = np.array([label_of[r.latent_value] for r in records])
    if shuffle_labels:
        # Permute by instance, not by record, so an instance's four arms keep the
        # same (wrong) label. Permuting per record would let the probe recover the
        # label from the passage it shares with its other arms.
        instances = sorted({r.semantic_instance_id for r in records})
        rng = np.random.default_rng(seed + 9973)
        shuffled = dict(zip(instances, rng.permutation(instances), strict=True))
        by_instance = {
            r.semantic_instance_id: label_of[r.latent_value] for r in records
        }
        labels = np.array(
            [by_instance[shuffled[r.semantic_instance_id]] for r in records]
        )
    conditions = sorted({str(r.condition) for r in records})
    train_ids, test_ids = instance_split(records, train_frac=train_frac, seed=seed)

    is_train = np.array([r.semantic_instance_id in train_ids for r in records])
    is_test = np.array([r.semantic_instance_id in test_ids for r in records])
    cond_of = np.array([str(r.condition) for r in records])

    # A class present in the test split but absent from training is unlearnable and
    # silently drags accuracy down, so restrict to classes with support in both.
    counts = np.bincount(labels[is_train], minlength=len(label_names))
    keep_classes = {i for i, c in enumerate(counts) if c >= min_per_class}
    if not keep_classes:
        raise ValueError("no class has enough training support to fit a probe")
    keep = np.array([lab in keep_classes for lab in labels])
    remap = {old: new for new, old in enumerate(sorted(keep_classes))}
    n_classes = len(remap)
    if n_classes < 2:
        raise ValueError(f"only {n_classes} usable class; a probe is meaningless")

    results: list[ProbeResult] = []
    for layer_index, layer in enumerate(layers):
        features = torch.from_numpy(
            residuals[:, layer_index].astype(np.float32)
        ).to(device)
        if center_per_condition:
            # Remove each arm's own offset so a cross-arm score reflects the
            # z-direction rather than the arms sitting in different regions.
            # The mean comes from the TRAINING instances of that arm only: taking
            # it over all records would let the test split inform its own
            # transform. (Measured: train-only scores 0.014 higher, so this costs
            # nothing but honesty.)
            for condition in conditions:
                mask = torch.from_numpy(cond_of == condition).to(device)
                fit_mask = torch.from_numpy(
                    (cond_of == condition) & is_train
                ).to(device)
                features[mask] = features[mask] - features[fit_mask].mean(
                    0, keepdim=True
                )
        target = torch.tensor(
            [remap.get(int(lab), -1) for lab in labels], device=device
        )

        # The joint probe: ONE weight matrix fitted on every arm at once, then
        # scored per arm. This is the strongest form of the shared-direction
        # claim, because no per-arm quantity of any kind enters -- unlike the
        # centred grid below, whose per-arm means make the classifier
        # arm-dependent and which was added after seeing the raw transfer.
        joint_train = is_train & keep
        if joint_train.sum() >= n_classes * min_per_class:
            for test_condition in conditions:
                test_mask = is_test & keep & (cond_of == test_condition)
                if test_mask.sum() < n_classes:
                    continue
                results.append(
                    ProbeResult(
                        layer=layer,
                        train_condition="joint",
                        test_condition=test_condition,
                        accuracy=_fit_logistic(
                            features[joint_train], target[joint_train],
                            features[test_mask], target[test_mask],
                            n_classes=n_classes, seed=seed,
                        ),
                        chance=1.0 / n_classes,
                        n_train=int(joint_train.sum()),
                        n_test=int(test_mask.sum()),
                        n_classes=n_classes,
                        seed=seed,
                        kind="joint",
                        shuffled=shuffle_labels,
                    )
                )

        for train_condition in conditions:
            train_mask = is_train & keep & (cond_of == train_condition)
            if train_mask.sum() < n_classes * min_per_class:
                continue
            for test_condition in conditions:
                test_mask = is_test & keep & (cond_of == test_condition)
                if test_mask.sum() < n_classes:
                    continue
                accuracy = _fit_logistic(
                    features[train_mask],
                    target[train_mask],
                    features[test_mask],
                    target[test_mask],
                    n_classes=n_classes,
                    seed=seed,
                )
                results.append(
                    ProbeResult(
                        layer=layer,
                        train_condition=train_condition,
                        test_condition=test_condition,
                        accuracy=accuracy,
                        chance=1.0 / n_classes,
                        n_train=int(train_mask.sum()),
                        n_test=int(test_mask.sum()),
                        n_classes=n_classes,
                        seed=seed,
                        kind="per_arm",
                        shuffled=shuffle_labels,
                    )
                )
        print(f"  probed layer {layer}", flush=True)
    return results


def summarise(results: list[ProbeResult]) -> dict:
    """Per-arm decoding, cross-arm transfer, and the floor the headline needs.

    The claim needs the diagonal to be uniformly high. A large drop off the
    diagonal would mean each arm encodes ``z`` its own way, which is a different
    and weaker statement than shared availability.

    Three things here exist because the obvious summary is misleading.

    **``joint``** is the headline: one weight matrix over all arms, so it carries
    the shared-direction claim without any per-arm quantity. It is reported as a
    mean and standard deviation over seeds rather than a single number.

    **``shuffled_floor``** is the reference point for ``*_best``. Those are maxima
    over the layer axis, and a maximum over twelve noisy draws does not sit at
    ``chance``: on this data ``chance`` is 0.05 and the best-of-twelve floor from
    permuted labels is 0.093. Quoting a multiple of ``chance`` for a best-of-N
    statistic overstates it by about 1.9x. Report ``*_best`` against this, or
    report ``*_mean``, which needs no correction.
    """
    real = [r for r in results if not r.shuffled]
    shuffled = [r for r in results if r.shuffled]

    def _fold(rows, key) -> dict[str, list[float]]:
        out: dict[str, list[float]] = {}
        for r in rows:
            out.setdefault(key(r), []).append(r.accuracy)
        return out

    def _per_seed_max(rows, key) -> dict[str, list[float]]:
        """Max over layers within a seed, then collect across seeds.

        Taking the max over the pooled (layer, seed) cells would be a maximum over
        five times as many draws and would sit higher still.
        """
        grouped: dict[tuple[str, int], list[float]] = {}
        for r in rows:
            grouped.setdefault((key(r), r.seed), []).append(r.accuracy)
        out: dict[str, list[float]] = {}
        for (name, _seed), values in grouped.items():
            out.setdefault(name, []).append(max(values))
        return out

    def _stat(values: list[float]) -> dict[str, float]:
        return {
            "mean": float(np.mean(values)),
            "sd": float(np.std(values)),
            "n_seeds": len(values),
        }

    diag = [r for r in real if r.kind == "per_arm"
            and r.train_condition == r.test_condition]
    trans = [r for r in real if r.kind == "per_arm"
             and r.train_condition != r.test_condition]
    joint = [r for r in real if r.kind == "joint"]

    summary = {
        "chance": results[0].chance if results else float("nan"),
        "n_classes": results[0].n_classes if results else 0,
        "joint_best": {
            k: _stat(v) for k, v in
            sorted(_per_seed_max(joint, lambda r: r.test_condition).items())
        },
        "joint_by_layer": {
            f"L{layer}": _stat([r.accuracy for r in joint if r.layer == layer])
            for layer in sorted({r.layer for r in joint})
        },
        "diagonal_best": {
            k: max(v) for k, v in
            sorted(_fold(diag, lambda r: r.test_condition).items())
        },
        "diagonal_mean": {
            k: float(np.mean(v)) for k, v in
            sorted(_fold(diag, lambda r: r.test_condition).items())
        },
        "transfer_best": {
            k: max(v) for k, v in sorted(_fold(
                trans, lambda r: f"{r.train_condition}->{r.test_condition}"
            ).items())
        },
        "per_cell": [asdict(r) for r in results],
    }
    if shuffled:
        by_kind = {}
        for kind in sorted({r.kind for r in shuffled}):
            rows = [r for r in shuffled if r.kind == kind]
            by_kind[kind] = {
                "best_of_layers": _stat(
                    [v for vs in _per_seed_max(
                        rows, lambda r: f"{r.train_condition}->{r.test_condition}"
                    ).values() for v in vs]
                ),
                "mean_over_layers": float(np.mean([r.accuracy for r in rows])),
            }
        summary["shuffled_floor"] = by_kind
    return summary
