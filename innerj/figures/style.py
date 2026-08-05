"""Shared figure style: one place for the palette, the typography and the
primitives every panel uses.

The rules the deck follows, and why each one:

* **White ground, no chrome.** No top or right spine, no box, no frame on the
  legend, y-gridlines only. Anything that is not data or a label is removed.
* **Direct labels beat legends.** A series named at the end of its own line costs
  the reader no lookup, so :func:`endlabel` is the default and a legend is the
  exception (used only where lines cross too much to label in place).
* **Emphasis is a design variable.** The focal series is saturated and full
  weight; controls are :data:`GREY` and thinner. A reader should see the argument
  before reading the axis labels.
* **Identity is never colour alone.** Every series also carries a marker and a
  dash pattern, so the panels survive greyscale and colour-vision deficiency.
* **The triple is validated, not eyeballed.** Blue / orange / magenta were
  scored for OKLab separation under deuteranopia, protanopia and tritanopia:
  worst-case separation 17.5 against a target of 8. Blue/purple (1.8) and
  blue/teal (3.3) were rejected.
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt

from innerj import config

OUT = config.FIGURE_DIR

# Semantic colour assignment, fixed for the whole paper.
RESID = "#277da1"   # residual stream, and the focal series generally
ATTN = "#f8961e"    # attention branch
MLP = "#9e0059"     # MLP branch
FOURTH = "#43aa8b"  # a fourth ordered level (the control condition)
GREY = "#8e8e93"    # controls, distractors, reference lines: visually quiet
INK = "#1d1d1f"
GRID = "#e8e8ed"

DASH_LONG = (5, 2)
DASH_DOT = (1.5, 1.6)

STYLE = {
    "figure.dpi": 150,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "font.family": "sans-serif",
    "font.sans-serif": ["Lato", "Noto Sans CJK SC", "Open Sans", "Nimbus Sans",
                        "DejaVu Sans"],
    "font.monospace": ["Noto Sans Mono CJK SC", "DejaVu Sans Mono"],
    "font.size": 7.4,
    "axes.labelsize": 7.4,
    "axes.titlesize": 7.8,
    "legend.fontsize": 7.0,
    "xtick.labelsize": 6.9,
    "ytick.labelsize": 6.9,
    "axes.edgecolor": GREY,
    "axes.linewidth": 0.5,
    "axes.labelcolor": INK,
    "axes.labelpad": 3.0,
    "text.color": INK,
    "xtick.color": GREY,
    "ytick.color": GREY,
    "xtick.labelcolor": INK,
    "ytick.labelcolor": INK,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "grid.color": GRID,
    "grid.linewidth": 0.5,
    "legend.frameon": False,
    "legend.handlelength": 1.9,
    "legend.borderpad": 0.0,
    "legend.labelspacing": 0.35,
    "lines.linewidth": 1.4,
    "lines.markersize": 3.6,
    "lines.markeredgewidth": 0.0,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
}


def use_style() -> None:
    mpl.rcParams.update(STYLE)


def series(ax, x, point, lo=None, hi=None, *, color, marker="o", dashes=None,
           label=None, alpha=0.15, width=None, zorder=3):
    """One series with its interval. Identity is colour *and* marker *and* dashes."""
    if lo is not None:
        ax.fill_between(x, lo, hi, color=color, alpha=alpha, linewidth=0,
                        zorder=zorder - 1)
    line, = ax.plot(x, point, color=color, marker=marker, label=label,
                    zorder=zorder, linewidth=width or STYLE["lines.linewidth"])
    if dashes:
        line.set_dashes(dashes)
    return line


def endlabel(ax, x, y, text, color, *, dx=0.9, dy=0.0, va="center", ha="left",
             size=7.0, weight="normal"):
    """Name a series at the end of its own line, so no legend lookup is needed."""
    ax.annotate(text, xy=(x, y), xytext=(x + dx, y + dy), color=color,
                fontsize=size, va=va, ha=ha, weight=weight,
                annotation_clip=False)


def zero_line(ax, y=0.0, label=None):
    ax.axhline(y, color=GREY, linewidth=0.6, linestyle=(0, (3.5, 3)), zorder=1)
    if label:
        ax.annotate(label, xy=(0.005, y), xycoords=("axes fraction", "data"),
                    color=GREY, fontsize=6.4, va="bottom")


def window(ax, lo, hi, label=None, color=RESID):
    """Shade a layer window. Very light: it is context, not a series."""
    ax.axvspan(lo, hi, color=color, alpha=0.055, linewidth=0, zorder=0)
    if label:
        ax.annotate(label, xy=((lo + hi) / 2, 0.985), va="top",
                    xycoords=("data", "axes fraction"),
                    color=INK, fontsize=6.6, ha="center")


def panel(ax, title=None, xlabel=None, ylabel=None):
    """A small lowercase left-aligned title reads as a caption, not a heading."""
    if title:
        ax.set_title(title, fontsize=7.4, color=INK, loc="left", pad=5)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)


def save(fig, name: str) -> None:
    """PNG only, at 600 dpi.

    These are vector plots and a PDF export would be strictly better for print, but the
    deck ships one image format so that nothing in ``paper/`` is a raster wrapped in a
    PDF container. 600 dpi at the 6.5in text width is ~3900px, well above the 300 dpi
    print threshold. Add ``fig.savefig(OUT / f"{name}.pdf")`` here to get vector back.
    """
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.png")
    plt.close(fig)
    print(f"  wrote {name}.png", flush=True)
