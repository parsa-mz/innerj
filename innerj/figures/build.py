"""Publication figures, built from the artifacts under the data root.

Every number plotted here is read from a JSON/JSONL artifact on disk; nothing is
transcribed by hand. If a figure and ``docs/analysis.md`` disagree, the artifact
wins and the doc is wrong.

The deck, and the argument each figure carries:

1. ``overview`` -- the two mechanisms the paper distinguishes. A schematic, the only
   figure not drawn from data. The paper ships the vector-styled render in
   ``paper/images/overview.png`` instead; this function is kept as the reproducible
   fallback and as the specification the render was made from.
2. ``entry``    -- availability is not entry: the per-layer profile, the
   effect-against-its-confound plane, and the probe.
3. ``gather``   -- attention carries the transport and MLPs never do, in two task
   families, with the single-head case study and its non-replication.
4. ``window``   -- the behavioural window with both edges measured, and what
   happens at each edge.
5. ``mechanism`` -- selective necessity, and mediation through one derived
   J-direction against a matched random one.
6. ``example``   -- one instance, all four arms, what the readout literally
   contains. Appendix: it makes ``R_z`` concrete and shows why its absolute level
   must not be read as representation strength.

Form follows the data: every panel with an ordered layer axis is a line with a
bootstrap band and a zero reference, never bars.
"""

from __future__ import annotations

import json
import textwrap
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from innerj import config
from innerj.analysis.stats import paired_bootstrap
from innerj.figures.style import (
    ATTN,
    DASH_DOT,
    DASH_LONG,
    FOURTH,
    GREY,
    GRID,
    INK,
    MLP,
    RESID,
    endlabel,
    panel,
    save,
    series,
    use_style,
    window,
    zero_line,
)
from innerj.tasks.base import Condition, read_jsonl

DATA = config.DATA_ROOT

CONDITION_STYLE = {
    "report": (MLP, "D", None),
    "flexible": (RESID, "o", None),
    "automatic": (ATTN, "s", DASH_LONG),
    "control": (FOURTH, "^", DASH_DOT),
}


# --- 1. overview -----------------------------------------------------------


def _box(ax, x, y, w, h, *, face="white", edge=GREY, lw=0.7, radius=0.02, z=2):
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h, boxstyle=f"round,pad=0,rounding_size={radius}",
            facecolor=face, edgecolor=edge, linewidth=lw, zorder=z,
        )
    )


def _arrow(ax, start, end, *, color=INK, lw=0.9, style="-|>", dashes=None, z=4):
    patch = FancyArrowPatch(
        start, end, arrowstyle=style, mutation_scale=7, color=color,
        linewidth=lw, shrinkA=0, shrinkB=0, zorder=z,
    )
    if dashes:
        patch.set_linestyle((0, dashes))
    ax.add_patch(patch)


def fig_overview() -> None:
    """The two mechanisms, so the title's contrast is visible before §1.

    Depth runs upward and the query position is the right-hand column in both
    panels, so the only difference a reader has to absorb is where the variable
    comes from. All prose sits in the right margin: nothing overlaps the diagram.
    """
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 1.72))
    for ax, title in zip(
        axes,
        ["a  admission: a gate decides", "b  transport: attention gathers"],
        strict=True,
    ):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.set_title(title, fontsize=7.6, color=INK, loc="left", pad=1)
        _arrow(ax, (0.035, 0.10), (0.035, 0.86), color=GREY, lw=0.6)
        ax.annotate("depth", xy=(0.035, 0.875), color=GREY, fontsize=6.5, ha="center")
        _box(ax, 0.11, 0.12, 0.34, 0.70, face="#fbfbfd")
        ax.annotate("passage", xy=(0.28, 0.075), color=GREY, fontsize=6.5,
                    ha="center", va="top")
        _box(ax, 0.51, 0.12, 0.16, 0.70)
        ax.annotate("query", xy=(0.59, 0.075), color=GREY, fontsize=6.5, ha="center",
                    va="top")

    # ---- a: the variable is already at the query position; a gate admits it
    ax = axes[0]
    for y in np.linspace(0.18, 0.76, 7):
        for x in (0.18, 0.28, 0.38):
            ax.plot([x], [y], marker="o", color=RESID, markersize=2.6, alpha=0.5,
                    zorder=5)
        ax.plot([0.59], [y], marker="o", color=RESID, markersize=3.4, zorder=5)
    ax.plot([0.48, 0.70], [0.44, 0.44], color=MLP, linewidth=1.8, zorder=6)
    ax.annotate("gate", xy=(0.745, 0.44), color=MLP, fontsize=7.0, va="center")
    ax.annotate("$z$ is at the query\nposition at every\ndepth; what varies\n"
                "is whether it is\nadmitted",
                xy=(0.745, 0.76), color=GREY, fontsize=6.5, va="top")

    # ---- b: the variable is in the passage and is moved into place
    ax = axes[1]
    for y in np.linspace(0.18, 0.76, 7):
        for x in (0.18, 0.28, 0.38):
            ax.plot([x], [y], marker="o", color=RESID, markersize=2.6, alpha=0.5,
                    zorder=5)
    ax.axhspan(0.40, 0.60, xmin=0.51, xmax=0.67, color=ATTN, alpha=0.13, linewidth=0,
               zorder=1)
    for y in (0.44, 0.50, 0.56):
        _arrow(ax, (0.455, y), (0.555, y), color=ATTN, lw=1.1)
        ax.plot([0.59], [y], marker="o", color=RESID, markersize=3.4, zorder=5)
    ax.annotate("gather\n(attention)", xy=(0.70, 0.50), color=ATTN, fontsize=6.7,
                va="center")
    _arrow(ax, (0.455, 0.22), (0.545, 0.22), color=GREY, lw=0.8, dashes=(2, 2))
    ax.plot([0.572, 0.608], [0.20, 0.24], color=GREY, linewidth=0.9, zorder=6)
    ax.plot([0.572, 0.608], [0.24, 0.20], color=GREY, linewidth=0.9, zorder=6)
    ax.annotate("installed below,\nthen re-derived from\nthe passage above",
                xy=(0.70, 0.24), color=GREY, fontsize=6.5, va="center")
    _arrow(ax, (0.59, 0.64), (0.59, 0.80), color=RESID, lw=1.2)
    ax.annotate("consolidated stream\n$\\rightarrow$ answer", xy=(0.70, 0.76),
                color=RESID, fontsize=6.6, va="center")

    fig.subplots_adjust(wspace=0.06)
    save(fig, "overview")


# --- 2. availability is not entry -----------------------------------------


def fig_entry() -> None:
    """Three views of one dissociation: profile, effect-vs-confound, availability."""
    summary = json.load(
        (DATA / "stage1"
         / "s1v2_Qwen3.6-27B_Qwen3.6-27B_matched_n400_s0_summary.json").open()
    )
    probe = json.load(
        (DATA / "probe"
         / "probe2_Qwen3.6-27B_Qwen3.6-27B_matched_n400_s0.json").open()
    )["raw"]

    fig, axes = plt.subplots(
        1, 3, figsize=(6.5, 1.70), gridspec_kw={"width_ratios": [1.25, 1.0, 0.85],
                                                "wspace": 0.42}
    )

    # (a) per-layer entry by condition, labelled at the left where the arms separate
    ax = axes[0]
    profile = summary["layer_profile"]
    for cond in ("report", "flexible", "automatic", "control"):
        color, marker, dashes = CONDITION_STYLE[cond]
        layers = sorted(int(k) for k in profile[cond])
        values = [profile[cond][str(k)] for k in layers]
        focal = cond in ("flexible", "control")
        series(ax, layers, values, color=color, marker="", dashes=dashes,
               width=1.5 if focal else 1.0, zorder=4 if focal else 3)
        endlabel(ax, layers[0], values[0], cond, color, dx=-1.2, ha="right",
                 weight="bold" if focal else "normal", size=6.8)
    ax.set_xlim(15, 60)
    ax.set_xticks([24, 33, 42, 51, 59])
    panel(ax, "a  entry by depth", "layer", r"mean $R_z$ at the query position")

    # (b) the effect against the confound it must not be. x=0 is the primary row.
    ax = axes[1]
    # (dx, dy, ha, va) per contrast: the five points cluster in two pairs, so the
    # label anchors are placed by hand rather than by a single uniform offset.
    contrasts = [
        ("flexible_vs_control", "flexible$-$control", True,
         (+0.014, -0.004, "left", "top")),
        ("report_vs_control", "report$-$control", False,
         (+0.018, +0.004, "left", "bottom")),
        ("flexible_vs_automatic", "flexible$-$auto.", False,
         (-0.022, -0.010, "right", "top")),
        ("report_vs_automatic", "report$-$auto.", False,
         (+0.016, +0.005, "left", "bottom")),
        ("control_vs_automatic", "control$-$auto.", False,
         (-0.022, 0.000, "right", "center")),
    ]
    ax.axvspan(0.15, 0.55, color=GREY, alpha=0.09, linewidth=0, zorder=0)
    ax.annotate("difficulty differs:\nnot interpretable", xy=(0.345, 0.006),
                color=GREY, fontsize=6.3, ha="center", va="bottom")
    for key, label, focal, (dx, dy, ha, va) in contrasts:
        c = summary["contrasts"][key]
        x, y = c["delta_accuracy"]["point"], c["delta_entry"]["point"]
        color = RESID if focal else GREY
        ax.errorbar(
            x, y,
            xerr=[[x - c["delta_accuracy"]["lo"]], [c["delta_accuracy"]["hi"] - x]],
            yerr=[[y - c["delta_entry"]["lo"]], [c["delta_entry"]["hi"] - y]],
            fmt="o", color=color, markersize=4.2 if focal else 3.0,
            elinewidth=0.9 if focal else 0.7, capsize=0,
            zorder=5 if focal else 3,
        )
        ax.annotate(label, xy=(x, y), xytext=(x + dx, y + dy), ha=ha, va=va,
                    color=INK if focal else GREY, fontsize=6.4,
                    weight="bold" if focal else "normal")
    zero_line(ax, 0.0)
    ax.axvline(0, color=GREY, linewidth=0.6, linestyle=(0, (3.5, 3)), zorder=1)
    ax.set_xlim(-0.13, 0.60)
    ax.set_ylim(-0.035, 0.128)
    panel(ax, "b  effect against its confound", r"$\Delta$ accuracy",
          r"$\Delta R_z$ (band mean)")

    # (c) availability: z is decodable in every arm, control included.
    # The joint probe -- one weight matrix over all four arms -- is what the text
    # claims, so it is what the panel draws. The reference line is the permuted-
    # label floor rather than 1/n_classes, because these are maxima over twelve
    # layers and a max over twelve draws does not sit at chance.
    ax = axes[2]
    joint = probe["joint_best"]
    floor = probe["shuffled_floor"]["joint"]["best_of_layers"]["mean"]
    arms = ["control", "automatic", "flexible", "report"]
    for i, arm in enumerate(arms):
        color = CONDITION_STYLE[arm][0]
        value, sd = joint[arm]["mean"], joint[arm]["sd"]
        ax.plot([floor, value], [i, i], color=color, linewidth=1.5, zorder=3)
        # The marker is drawn under the whisker, not over it: at this scale the
        # sd is about a marker-width, so a dot on top hides the interval the
        # caption promises.
        ax.plot([value], [i], marker="o", color=color, markersize=3.4, zorder=4)
        ax.errorbar(value, i, xerr=sd, color=INK, elinewidth=0.9,
                    capsize=2.0, capthick=0.9, zorder=5)
        ax.annotate(f"{value:.2f}", xy=(value, i), xytext=(value + sd + 0.035, i),
                    color=INK, fontsize=6.5, va="center")
    ax.axvline(floor, color=GREY, linewidth=0.6, linestyle=(0, (3.5, 3)))
    ax.annotate("permuted\nlabels", xy=(floor + 0.02, 3.25), color=GREY,
                fontsize=6.3, linespacing=0.95)
    ax.set_yticks(range(len(arms)))
    ax.set_yticklabels(arms)
    ax.set_ylim(-0.6, 3.6)
    ax.set_xlim(0, 1.06)
    ax.grid(False)
    ax.grid(True, axis="x")
    panel(ax, "c  availability (one probe, all arms)", "20-way accuracy")

    save(fig, "entry")


# --- 3. the gather --------------------------------------------------------


def _matched_distance(
    *paths: Path, lo: int = 3, hi: int = 5, field: str = "delta_donor"
) -> dict:
    """Mean over readout distances ``lo``-``hi``, per (patch layer, component).

    Averaging over a fixed distance window is what makes patch depths comparable:
    a fixed *late* readout grades survival instead of installation.

    Several paths merge, because a sweep too long for one run is sharded by patch
    layer. ``field`` selects the metric: ``delta_donor`` is the percentile rank,
    ``delta_donor_logrank`` the non-saturating companion. Artifacts written before
    the second metric existed carry only the first, so asking for a field they lack
    raises rather than silently plotting the wrong one.
    """
    grouped = defaultdict(list)
    for path in paths:
        for c in json.load(path.open())["cells"]:
            if lo <= c["distance"] <= hi:
                if field not in c:
                    raise KeyError(
                        f"{path.name} has no {field!r}; it predates that metric. "
                        f"Re-run the sweep or plot delta_donor."
                    )
                grouped[(c["patch_layer"], c["component"], c["kind"])].append(c)
    return {
        key: {
            stat: float(np.mean([c[field][stat] for c in cs]))
            for stat in ("point", "lo", "hi")
        }
        for key, cs in grouped.items()
    }


def _branch_panel(ax, table, *, label_at, annotate=None):
    """resid / attn / mlp against patch layer, at matched distance.

    ``label_at`` maps a component kind to ``(layer, dy)``: where along its own
    line to name the series, and how far to lift the text off it.
    """
    for kind, color, marker, dashes, name in [
        ("resid", RESID, "o", None, "residual stream"),
        ("attn", ATTN, "s", DASH_LONG, "attention"),
        ("mlp", MLP, "^", DASH_DOT, "MLP"),
    ]:
        rows = sorted(
            (layer, v) for (layer, component, k), v in table.items()
            if k == kind and (kind != "attn" or component.endswith(".all"))
        )
        if not rows:
            continue
        layers = [layer for layer, _ in rows]
        series(ax, layers, [v["point"] for _, v in rows],
               [v["lo"] for _, v in rows], [v["hi"] for _, v in rows],
               color=color, marker=marker, dashes=dashes)
        if kind in label_at:
            layer, dy = label_at[kind]
            endlabel(ax, layer, dict(rows)[layer]["point"] + dy, name, color,
                     dx=0.6, size=6.6)
    zero_line(ax)
    if annotate:
        text, xy, xytext = annotate
        ax.annotate(text, xy=xy, xytext=xytext, color=INK, fontsize=6.5,
                    ha="center", va="bottom",
                    arrowprops=dict(arrowstyle="-", color=GREY, linewidth=0.6,
                                    shrinkA=1, shrinkB=2))


def fig_gather() -> None:
    """Attention transports and MLPs never do -- in both families.

    The third panel is the honest part: the single head that accounts for the
    whole stream effect on one family does nothing on the other, so the general
    claim is about the branch and the window, not the site.
    """
    # Panel (a) is the n=120 sweep under the non-saturating readout, sharded by
    # patch layer. R_z saturates above 0.999 through most of the window, so a
    # profile drawn from it compresses the peak the panel exists to show.
    language = _matched_distance(
        DATA / "sweep" / "n120_early_Qwen3.6-27B.json",
        DATA / "sweep" / "n120_window_Qwen3.6-27B.json",
        field="delta_donor_logrank",
    )
    tracking = _matched_distance(DATA / "sweep" / "F_track_gather_Qwen3.6-27B.json")
    heads = _matched_distance(DATA / "sweep" / "A_heads_n60_Qwen3.6-27B.json")

    fig, axes = plt.subplots(
        1, 3, figsize=(6.5, 1.72),
        gridspec_kw={"width_ratios": [1.32, 0.62, 1.0], "wspace": 0.30},
    )

    _branch_panel(
        axes[0], language,
        label_at={"resid": (45, 0.14), "attn": (42, 0.20), "mlp": (45, -0.34)},
        annotate=("attention carries most\nof the stream here",
                  (39, 1.92), (26.5, 1.32)),
    )
    axes[0].set_xticks([12, 18, 24, 30, 36, 42, 48])
    axes[0].set_xlim(11, 57)
    # The metric goes in the title, not only the axis: panel (b) is drawn in R_z
    # from an older artifact, and two panels side by side in different units with
    # nothing saying so is how a reader is misled.
    panel(axes[0], r"a  language family, $-\log_{10}(\mathrm{rank})$",
          "layer of the patched component", r"$\Delta$ donor, at matched distance")

    # Panel (b) keeps R_z: its artifact predates the second metric, and the axes
    # are therefore *not* shared. Different scales, so both are labelled -- a
    # borrowed y-axis here would silently misplace every tracking point.
    _branch_panel(axes[1], tracking, label_at={})
    axes[1].set_xticks([33, 39, 45])
    axes[1].set_xlim(32, 49.5)
    axes[1].annotate("attention peak\nmoves to L48",
                     xy=(48, 0.026), xytext=(38.5, 0.046),
                     color=INK, fontsize=6.5, ha="center", va="bottom",
                     arrowprops=dict(arrowstyle="-", color=GREY, linewidth=0.6,
                                     shrinkA=1, shrinkB=2))
    panel(axes[1], r"b  tracking family, $R_z$", "layer", None)

    # (c) the case study and its non-replication, head by head at L39
    ax = axes[2]
    for table, color, marker, name, dashes, dy in [
        (heads, RESID, "o", "language", None, 0.004),
        (tracking, ATTN, "s", "tracking", DASH_LONG, -0.005),
    ]:
        rows = sorted(
            (int(component.split(".H")[1]), table[(layer, component, kind)])
            for (layer, component, kind) in table
            if layer == 39 and ".H" in component
        )
        if not rows:
            continue
        xs = [h for h, _ in rows]
        ax.plot(xs, [v["point"] for _, v in rows], color=color, marker=marker,
                markersize=3.0, linewidth=0.9,
                linestyle=(0, dashes) if dashes else "-", zorder=3)
        ax.fill_between(xs, [v["lo"] for _, v in rows], [v["hi"] for _, v in rows],
                        color=color, alpha=0.13, linewidth=0)
        endlabel(ax, xs[-1], rows[-1][1]["point"] + dy, name, color, dx=0.6, size=6.6)
    # the three-way coincidence: one head, the whole block, the whole stream
    reference = heads.get((39, "resid.L39", "resid"))
    if reference:
        ax.axhline(reference["point"], color=GREY, linewidth=0.7,
                   linestyle=(0, (4, 2.5)))
        ax.annotate("attn.L39 block $=$ resid.L39", xy=(0.2, reference["point"]),
                    color=GREY, fontsize=6.2, va="top")
    ax.annotate("H15", xy=(15, heads[(39, "attn.L39.H15", "attn")]["point"] + 0.003),
                color=INK, fontsize=6.5, ha="center", va="bottom")
    zero_line(ax)
    ax.set_xticks([0, 5, 10, 15, 20, 23])
    ax.set_xlim(-0.7, 29)
    panel(ax, "c  heads of L39", "attention head", r"$\Delta$ donor $R_z$")

    save(fig, "gather")


# --- 4. the behavioural window -------------------------------------------


def _depth_rates() -> tuple[list[int], dict, list[float], list[float]]:
    """Answer rates by patch layer, with the per-distractor control.

    A patch that merely destroys the computation drives the answer toward uniform
    over the candidate set, which lifts the donor's symbol from ~0 to ~1/n for
    free. So the effect is donor rate minus the *matched* per-distractor rate, and
    the distractor series is plotted beside it rather than assumed away.
    """
    observations = [
        json.loads(line)
        for line in (
            DATA / "counterfactual" / "cf_resid_depth_Qwen3.6-27B_observations.jsonl"
        ).read_text().splitlines()
        if line.strip()
    ]
    candidates = {
        r.semantic_instance_id: set(r.candidate_answers)
        for r in read_jsonl(DATA / "language" / "Qwen3.6-27B_matched_n400_s0.jsonl")
        if r.condition is Condition.FLEXIBLE
    }
    by_layer = defaultdict(list)
    for o in observations:
        by_layer[int(o["component"].split(".L")[1])].append(o)

    layers, rates, lo, hi = [], defaultdict(list), [], []
    for layer in sorted(by_layer):
        group = by_layer[layer]
        donor, other, gold = [], [], []
        for o in group:
            others = candidates[o["target_instance"]] - {o["gold_symbol"],
                                                         o["donor_symbol"]}
            donor.append(float(o["patched_is_donor_symbol"]))
            other.append(float(o["patched_answer"] in others) / max(len(others), 1))
            gold.append(float(o["patched_is_gold"]))
        estimate = paired_bootstrap(np.array(donor), np.array(other), seed=0)
        layers.append(layer)
        rates["donor"].append(float(np.mean(donor)))
        rates["distractor"].append(float(np.mean(other)))
        rates["accuracy"].append(float(np.mean(gold)))
        rates["effect"].append(estimate.point)
        lo.append(estimate.lo)
        hi.append(estimate.hi)
    return layers, rates, lo, hi


def fig_window() -> None:
    """Both edges of the transport window, and what each edge is."""
    layers, rates, lo, hi = _depth_rates()
    repair = json.load(
        (DATA / "specificity" / "repair_Qwen3.6-27B_results.json").open()
    )["results"]

    fig, axes = plt.subplots(
        1, 3, figsize=(6.5, 1.70), gridspec_kw={"width_ratios": [1.1, 1.0, 0.95],
                                                "wspace": 0.40}
    )

    # (a) the three rates, so destruction is visible rather than argued
    ax = axes[0]
    window(ax, 38.5, 48.5, "L39-L48")
    for key, color, marker, dashes, name, focal, dy in [
        ("donor", RESID, "o", None, "donor's symbol", True, 0.0),
        ("accuracy", ATTN, "s", DASH_LONG, "correct answer", False, -0.055),
        ("distractor", GREY, "^", DASH_DOT, "distractor", False, 0.055),
    ]:
        series(ax, layers, rates[key], color=color, marker=marker, dashes=dashes,
               width=1.5 if focal else 1.0, zorder=4 if focal else 3)
        endlabel(ax, layers[-1], rates[key][-1] + dy, name, color, dx=0.8, size=6.5,
                 weight="bold" if focal else "normal")
    ax.axhline(0.25, color=GREY, linewidth=0.6, linestyle=(0, (3.5, 3)))
    ax.annotate("chance", xy=(24.3, 0.265), color=GREY, fontsize=6.3)
    ax.set_xlim(23, 71)
    ax.set_xticks([24, 33, 42, 51, 57])
    ax.set_ylim(-0.03, 1.03)
    panel(ax, "a  answer rates", "layer of the residual patch", "rate")

    # (b) the controlled effect: where transport beats destruction
    ax = axes[1]
    window(ax, 38.5, 48.5, "L39-L48")
    series(ax, layers, rates["effect"], lo, hi, color=RESID, marker="o", width=1.5)
    zero_line(ax)
    ax.annotate("repair\nbelow", xy=(33, 0.03), xytext=(28.0, 0.30), color=GREY,
                fontsize=6.4, ha="center",
                arrowprops=dict(arrowstyle="-", color=GREY, linewidth=0.6,
                                shrinkA=1, shrinkB=2))
    ax.annotate("destruction\nabove", xy=(54, 0.15), xytext=(53.0, 0.44), color=GREY,
                fontsize=6.4, ha="center",
                arrowprops=dict(arrowstyle="-", color=GREY, linewidth=0.6,
                                shrinkA=1, shrinkB=2))
    ax.set_xticks([24, 33, 42, 51, 57])
    panel(ax, "b  transport, distractor-controlled", "layer of the residual patch",
          "donor $-$ distractor")

    # (c) why the lower edge is repair, and what it costs the target's own value
    ax = axes[2]
    rows = sorted((int(r["component"].split(".L")[1]), r) for r in repair)
    xs = [layer for layer, _ in rows]
    for key, color, marker, dashes, name in [
        ("delta_donor", RESID, "o", None, "donor's value,\ninstalled"),
        ("delta_target", MLP, "^", DASH_DOT, "target's own value"),
    ]:
        series(ax, xs, [r[key]["point"] for _, r in rows],
               [r[key]["lo"] for _, r in rows], [r[key]["hi"] for _, r in rows],
               color=color, marker=marker, dashes=dashes)
        endlabel(ax, xs[-1], rows[-1][1][key]["point"], name, color, dx=0.5, size=6.4)
    zero_line(ax)
    ax.set_xticks(xs)
    ax.set_xlim(32, 58)
    panel(ax, "c  survival to L46-51", "layer of the residual patch",
          r"$\Delta R_z$ read above the patch")

    save(fig, "window")


# --- 5. necessity and mediation ------------------------------------------


def fig_mechanism() -> None:
    """Selective necessity, and mediation through one derived direction."""
    ablation = json.load(
        (DATA / "ablate" / "loo_necessity_Qwen3.6-27B_results.json").open()
    )["results"]
    observations = [
        json.loads(line)
        for line in (
            DATA / "mediate" / "wide_n80_Qwen3.6-27B_observations.jsonl"
        ).read_text().splitlines()
        if line.strip()
    ]

    fig, axes = plt.subplots(
        1, 2, figsize=(6.5, 1.70), gridspec_kw={"width_ratios": [1.3, 1.0],
                                               "wspace": 0.30}
    )

    # (a) mean ablation: cost where z is needed, against cost in the control arm
    ax = axes[0]
    components = ["resid.L39", "attn.L39.all", "attn.L39.H15", "attn.L43.all",
                  "mlp.L39"]
    rows = {(r["component"], r["mode"], r["condition"]): r for r in ablation}
    for i, component in enumerate(components):
        for offset, cond in ((0.24, "report"), (0.0, "flexible"), (-0.24, "control")):
            r = rows.get((component, "mean", cond))
            if r is None:
                continue
            color = CONDITION_STYLE[cond][0]
            e = r["delta_accuracy"]
            y = len(components) - 1 - i + offset
            ax.plot([e["lo"], e["hi"]], [y, y], color=color, linewidth=0.9, zorder=3)
            ax.plot([e["point"]], [y], marker=CONDITION_STYLE[cond][1], color=color,
                    markersize=3.8, zorder=4)
            if i == 0:
                endlabel(ax, e["hi"], y, cond, color, dx=0.02, size=6.4)
    ax.axvline(0, color=GREY, linewidth=0.6, linestyle=(0, (3.5, 3)))
    ax.set_yticks(range(len(components)))
    ax.set_yticklabels(list(reversed(components)), fontsize=6.6)
    ax.set_ylim(-0.6, len(components) - 0.4)
    ax.set_xlim(-0.78, 0.42)
    ax.grid(False)
    ax.grid(True, axis="x")
    ax.annotate("costs the arms that need $z$", xy=(-0.40, -0.52), color=GREY,
                fontsize=6.3, ha="center")
    ax.annotate("costs the control arm", xy=(0.19, -0.52), color=GREY, fontsize=6.3,
                ha="center")
    panel(ax, "a  necessity (mean ablation)", r"$\Delta$ accuracy")

    # (b) mediation: the loss from removing the concept direction, against the null
    # distribution over every wrong concept. Paired per instance, so each row is the
    # same 80 trials. A single random control could be a lucky draw; twenty rivals
    # plotted individually cannot be, which is why the null is drawn as a spread
    # rather than one interval.
    ax = axes[1]
    grouped = defaultdict(lambda: defaultdict(dict))
    for o in observations:
        if o["dose"] not in (0.0, 1.0) or o["projection"] != "absolute":
            continue
        key = (
            f"wrong:{o['wrong_token_id']}" if o["control"] == "wrong_all"
            else o["mode"]
        )
        grouped[o["component"]][o["target_instance"]][key] = o
    components = ["resid.L39", "resid.L42"]

    def loss(instances, key):
        """Paired full-minus-`key` loss in donor-symbol rate, or None if absent."""
        rows = [v for v in instances if "full" in v and key in v]
        if len(rows) < 10:
            return None
        full = np.array([float(v["full"]["is_donor_symbol"]) for v in rows])
        other = np.array([float(v[key]["is_donor_symbol"]) for v in rows])
        return paired_bootstrap(full, other, seed=0)

    for i, component in enumerate(components):
        instances = list(grouped[component].values())
        y = len(components) - 1 - i
        rivals = sorted({
            k for v in instances for k in v if k.startswith("wrong:")
        })
        # The null, one faint marker per wrong concept.
        points = [e.point for k in rivals if (e := loss(instances, k)) is not None]
        if points:
            ax.plot(points, [y - 0.20] * len(points), marker="o", linestyle="none",
                    color=GREY, markersize=2.6, alpha=0.55, zorder=3)
            if i == 0:
                endlabel(ax, max(points), y - 0.20,
                         f"one per concept identity ({len(points)}),\n"
                         f"evaluated when it is a rival", GREY,
                         dx=0.012, size=6.4)
        for offset, key, color, marker, name in (
            (0.20, "gold:static:absolute@1.00", RESID, "o", "concept direction"),
            (0.02, "orthogonal:static:absolute@1.00", ATTN, "D",
             "orthogonalised to the other 19"),
        ):
            e = loss(instances, key)
            if e is None:
                continue
            ax.plot([e.lo, e.hi], [y + offset] * 2, color=color, linewidth=1.1,
                    zorder=4)
            ax.plot([e.point], [y + offset], marker=marker, color=color,
                    markersize=4.0, zorder=5)
            if i == 0:
                endlabel(ax, e.hi, y + offset, name, color, dx=0.012, size=6.4)
    ax.axvline(0, color=GREY, linewidth=0.6, linestyle=(0, (3.5, 3)))
    ax.set_yticks(range(len(components)))
    ax.set_yticklabels(list(reversed(components)), fontsize=6.6)
    ax.set_ylim(-0.45, len(components) - 0.55)
    ax.set_xlim(-0.05, 0.46)
    ax.grid(False)
    ax.grid(True, axis="x")
    panel(ax, "b  mediation, against every wrong concept",
          "loss in counterfactual answer rate")

    save(fig, "mechanism")



# --- the two body figures: the mechanism spine, and the calibration failure ----


def fig_spine() -> None:
    """The causal chain in one row: transport, its behaviour, and its specificity.

    Built for the main text, which otherwise carries these three results as walls of
    layer numbers. Each panel is one of the paper's claims, and they are placed
    together because the argument is that they are the same story measured three ways.
    """
    language = _matched_distance(
        DATA / "sweep" / "n120_early_Qwen3.6-27B.json",
        DATA / "sweep" / "n120_window_Qwen3.6-27B.json",
        field="delta_donor_logrank",
    )
    layers, rates, lo, hi = _depth_rates()
    observations = [
        json.loads(line)
        for line in (
            DATA / "mediate" / "wide_n80_Qwen3.6-27B_observations.jsonl"
        ).read_text().splitlines()
        if line.strip()
    ]

    fig, axes = plt.subplots(
        1, 3, figsize=(6.5, 1.85),
        gridspec_kw={"width_ratios": [1.25, 1.0, 0.95], "wspace": 0.34},
    )

    # (a) where the variable can be installed, in the non-saturating readout
    _branch_panel(
        axes[0], language,
        label_at={"resid": (45, 0.10), "attn": (42, 0.18), "mlp": (45, -0.30)},
    )
    axes[0].set_xticks([12, 24, 36, 48])
    axes[0].set_xlim(11, 55)
    axes[0].annotate("no positive transport\nL24--L33",
                     xy=(28, 0.02), xytext=(19, 0.75),
                     color=GREY, fontsize=6.2, ha="center", va="bottom",
                     arrowprops=dict(arrowstyle="-", color=GREY, linewidth=0.6,
                                     shrinkA=1, shrinkB=2))
    panel(axes[0], "a  transport, matched distance", "patch layer",
          r"$\Delta$ donor $-\log_{10}(\mathrm{rank})$")

    # (b) the behavioural counterpart, with both edges visible
    ax = axes[1]
    series(ax, layers, rates["effect"], lo, hi, color=RESID, marker="o")
    zero_line(ax)
    ax.set_xticks([33, 39, 45, 51, 57])
    ax.annotate("survival", xy=(33, 0.01), xytext=(34, 0.44), color=GREY,
                fontsize=6.2, ha="center")
    ax.annotate("destruction", xy=(55, 0.13), xytext=(53, 0.44), color=GREY,
                fontsize=6.2, ha="center")
    panel(ax, "b  counterfactual answer", "patch layer",
          "donor $-$ distractor")

    # (c) the specificity control: gold against every wrong concept
    ax = axes[2]
    grouped = defaultdict(lambda: defaultdict(dict))
    for o in observations:
        if o["projection"] != "absolute" or o["dose"] not in (0.0, 1.0):
            continue
        key = (f"wrong:{o['wrong_token_id']}" if o["control"] == "wrong_all"
               else o["mode"])
        grouped[o["component"]][o["target_instance"]][key] = o

    def loss(instances, key):
        rows = [v for v in instances if "full" in v and key in v]
        if len(rows) < 10:
            return None
        return paired_bootstrap(
            np.array([float(v["full"]["is_donor_symbol"]) for v in rows]),
            np.array([float(v[key]["is_donor_symbol"]) for v in rows]), seed=0,
        )

    components = ["resid.L39", "resid.L42"]
    for i, component in enumerate(components):
        instances = list(grouped[component].values())
        y = len(components) - 1 - i
        rivals = sorted({k for v in instances for k in v if k.startswith("wrong:")})
        points = [e.point for k in rivals if (e := loss(instances, k)) is not None]
        if points:
            ax.plot(points, [y - 0.17] * len(points), marker="o", linestyle="none",
                    color=GREY, markersize=2.4, alpha=0.55, zorder=3)
            if i == 0:
                endlabel(ax, max(points), y - 0.17, "each wrong concept", GREY,
                         dx=0.012, size=6.2)
        e = loss(instances, "gold:static:absolute@1.00")
        if e is not None:
            ax.plot([e.lo, e.hi], [y + 0.17] * 2, color=RESID, linewidth=1.1, zorder=4)
            ax.plot([e.point], [y + 0.17], marker="o", color=RESID, markersize=3.8,
                    zorder=5)
            if i == 0:
                endlabel(ax, e.hi, y + 0.17, "concept direction", RESID, dx=0.012,
                         size=6.2)
    ax.axvline(0, color=GREY, linewidth=0.6, linestyle=(0, (3.5, 3)))
    ax.set_yticks(range(len(components)))
    ax.set_yticklabels(list(reversed(components)), fontsize=6.4)
    ax.set_ylim(-0.45, len(components) - 0.55)
    ax.set_xlim(-0.05, 0.44)
    ax.grid(False)
    ax.grid(True, axis="x")
    panel(ax, "c  mediation vs. every rival", "loss in answer rate")

    save(fig, "spine")


def fig_calibration() -> None:
    """A readout shift is not a calibrated measure of use.

    The paper's most portable claim had no figure. Panel (a) is the whole argument:
    three components land at the same readout level and do three different things to
    behaviour. Panel (b) forestalls the obvious misreading --- the trial-level
    relationship is real but weak, so the readout is uncalibrated, not uninformative.
    """
    rows = [
        json.loads(line)
        for line in (
            DATA / "counterfactual" / "trial_level_Qwen3.6-27B_observations.jsonl"
        ).read_text().splitlines()
        if line.strip()
    ]
    order = ["resid.L39", "attn.L39.H15", "attn.L39"]
    shift, margin = {}, {}
    for component in order:
        group = [o for o in rows if o["component"] == component]
        shift[component] = np.array(
            [o["donor_logrank_patched"] - o["donor_logrank_clean"] for o in group]
        )
        margin[component] = np.array(
            [o["donor_vs_other_margin_patched"] - o["donor_vs_other_margin_clean"]
             for o in group]
        )

    fig, axes = plt.subplots(
        1, 2, figsize=(6.5, 1.85), gridspec_kw={"width_ratios": [1.0, 1.0],
                                                "wspace": 0.30}
    )

    # (a) equal readout, unequal behaviour
    ax = axes[0]
    for i, component in enumerate(order):
        y = len(order) - 1 - i
        for offset, values, color, name in (
            (0.16, shift[component], RESID, r"readout ($-\log_{10}$ rank)"),
            (-0.16, margin[component], ATTN, r"behaviour (margin)"),
        ):
            e = paired_bootstrap(values, np.zeros_like(values), seed=0)
            ax.plot([e.lo, e.hi], [y + offset] * 2, color=color, linewidth=1.1,
                    zorder=3)
            ax.plot([e.point], [y + offset], marker="o", color=color,
                    markersize=3.8, zorder=4)
            if i == 0:
                endlabel(ax, e.hi, y + offset, name, color, dx=0.06, size=6.2)
    ax.axvline(0, color=GREY, linewidth=0.6, linestyle=(0, (3.5, 3)))
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(list(reversed(order)), fontsize=6.4)
    ax.set_ylim(-0.45, len(order) - 0.5)
    ax.set_xlim(-0.3, 4.6)
    ax.grid(False)
    ax.grid(True, axis="x")
    panel(ax, "a  same readout, different behaviour", "shift")

    # (b) the trial-level relationship: real but weak
    ax = axes[1]
    for component, color, marker in (("resid.L39", RESID, "o"),
                                     ("attn.L39.H15", ATTN, "s")):
        x, y = shift[component], margin[component]
        ax.plot(x, y, marker=marker, linestyle="none", color=color, markersize=2.2,
                alpha=0.5, zorder=3)
        slope, intercept = np.polyfit(x, y, 1)
        grid = np.linspace(x.min(), x.max(), 2)
        ax.plot(grid, slope * grid + intercept, color=color, linewidth=1.0,
                zorder=4)
        r = float(np.corrcoef(x, y)[0, 1])
        endlabel(ax, grid[-1], slope * grid[-1] + intercept,
                 f"{component}, $r={r:+.2f}$", color, dx=0.04, size=6.2)
    zero_line(ax)
    ax.set_xlim(-0.2, 6.9)
    panel(ax, "b  per trial, weak but real", r"$\Delta$ readout",
          r"$\Delta$ behaviour")

    save(fig, "calibration")


# --- 6. the worked example ------------------------------------------------


def _token(text: str) -> str:
    """Display a token unambiguously: leading spaces and newlines must be visible.

    ``repr`` already escapes the newline and shows the leading space, which is the
    whole point: `' Korean'` and `'Korean'` are different tokens, and a figure that
    renders them identically hides the distinction the readout turns on.
    """
    return repr(text)[1:-1]


def fig_example() -> None:
    """What the readout literally contains, for one instance in all four arms.

    Two things this figure exists to show. The concept becomes top-1 at L42-L45 in
    exactly the arms that need it verbalizably, and in the other two arms the readout
    at those layers holds *that arm's* task content instead. And the absolute level of
    ``R_z`` is not representation strength: the control arm reads $R_z = 0.9953$ at L45
    while the gold token ranks 1155th and the top of the readout is about punctuation.
    """
    path = next(
        (DATA / "example").glob("example_*_Qwen3.6-27B.json")
    )
    d = json.load(path.open())
    order = ["control", "automatic", "report", "flexible"]
    gold = d["latent_value"]

    fig, axes = plt.subplots(
        1, 2, figsize=(6.5, 2.2),
        gridspec_kw={"width_ratios": [0.85, 1.5], "wspace": 0.30},
    )

    # (a) the rank of the gold concept, which is what R_z is a percentile of
    ax = axes[0]
    window(ax, 38.5, 48.5)
    for cond in order:
        color, marker, dashes = CONDITION_STYLE[cond]
        rows = d["conditions"][cond]["layers"]
        xs = [r["layer"] for r in rows]
        ys = [r["rank"] + 1 for r in rows]
        focal = cond in ("flexible", "control")
        series(ax, xs, ys, color=color, marker=marker, dashes=dashes,
               width=1.5 if focal else 1.0, zorder=4 if focal else 3)
        endlabel(ax, xs[-1], ys[-1], cond, color, dx=0.8, size=6.4,
                 weight="bold" if focal else "normal")
    ax.set_yscale("log")
    ax.invert_yaxis()
    ax.set_yticks([1, 10, 100, 1000, 10000, 100000])
    ax.set_yticklabels(["top-1", "10", "100", "1k", "10k", "100k"])
    ax.set_xticks([24, 33, 42, 51, 57])
    ax.set_xlim(23, 68)
    panel(ax, "a  where the concept ranks", "layer",
          f"rank of \u201c{gold}\u201d of {248320:,} tokens")

    # (b) the readout itself, at the three layers where the arms diverge
    ax = axes[1]
    ax.axis("off")
    ax.set_xlim(0, 4)
    ax.set_ylim(0, 1)
    layers = [39, 42, 45]
    panel(ax, "b  top-3 of the readout at the query position")
    for col, cond in enumerate(order):
        color = CONDITION_STYLE[cond][0]
        row = d["conditions"][cond]
        ax.text(col + 0.5, 0.99, cond, color=color, fontsize=6.8, ha="center",
                va="top", weight="bold")
        question = row["instruction"].split("\n")[0].strip()
        ax.text(col + 0.5, 0.935, textwrap.fill(question, 21), color=GREY,
                fontsize=5.4, ha="center", va="top", linespacing=1.3)
        by_layer = {r["layer"]: r for r in row["layers"]}
        for i, layer in enumerate(layers):
            r = by_layer[layer]
            top = 0.755 - i * 0.250
            ax.text(col + 0.03, top, f"L{layer}", color=GREY, fontsize=5.8,
                    va="top")
            ax.text(col + 0.97, top, f"rank {r['rank']:,}", color=GREY,
                    fontsize=5.8, va="top", ha="right")
            for j, tok in enumerate(r["top_tokens"][:3]):
                is_gold = tok.strip() == gold
                ax.text(col + 0.06, top - 0.058 - j * 0.056, _token(tok),
                        color=color if is_gold else INK,
                        fontsize=5.8, va="top", family="monospace",
                        weight="bold" if is_gold else "normal")
    for col in range(1, 4):
        ax.axvline(col, color=GRID, linewidth=0.5)
    for i in range(1, len(layers)):
        ax.axhline(0.790 - i * 0.250, color=GRID, linewidth=0.5)

    save(fig, "example")


def main() -> None:
    use_style()
    # fig_overview is not in the default build: the paper ships the vector-styled
    # render in paper/images/. Call it explicitly to regenerate the fallback.
    for fn in (fig_entry, fig_gather, fig_window, fig_mechanism, fig_spine,
               fig_calibration, fig_example):
        fn()


if __name__ == "__main__":
    main()
