"""Terminal output. One place, so that runs look the same and stay greppable.

Two rules this module exists to enforce.

**Progress must be visible.** A long GPU run that prints nothing is
indistinguishable from a hang, and we have sat through a 27-minute silence
against a timeout to learn it. :func:`track` wraps any iterable in a live
progress bar with an elapsed and remaining estimate.

**Progress must not become the output.** When stdout is not a terminal --- a
``nohup`` log, a CI capture, a pipe into ``tail`` --- Rich's live redraw turns
into thousands of near-identical lines and the log becomes unreadable. Every
helper here degrades to a plain flushed line in that case, so the same command
is pleasant interactively and quiet in a log file.

The rest of the module is a thin vocabulary: :func:`step` for a stage boundary,
:func:`detail` for a subordinate fact, :func:`warn` for something not to scroll
past, and :func:`wrote` for an artifact path.
"""

from __future__ import annotations

import sys
from collections.abc import Iterable, Iterator
from pathlib import Path

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from rich.theme import Theme

#: Muted where the paper's figures are muted; the accent marks a real result.
THEME = Theme(
    {
        "step": "bold cyan",
        "detail": "dim",
        "path": "green",
        "warn": "yellow",
    }
)

# markup=False: message text is data, not a template. A path or verdict string
# containing a bracket would otherwise be parsed as a tag -- and since Rich only
# parses markup on a terminal, the resulting MarkupError could never show up in a
# logged run. Styling is applied via style=, which does not touch the text.
console = Console(theme=THEME, highlight=False, markup=False)


def interactive() -> bool:
    """True when a live redraw will help rather than flood a log file."""
    return sys.stdout.isatty()


def _say(style: str, prefix: str, message: str) -> None:
    """Write one line, styled on a terminal and plain in a log.

    The branch lives here rather than in each verb. The message is passed as
    ``style=`` rather than interpolated into markup, because interpolating it
    makes any square bracket in the text a tag: ``wrote("/tmp/a [/tmp/b]")``
    raised ``MarkupError`` on a terminal and printed fine in a log, so the crash
    could not appear in a logged run. Paths and verdict strings both contain
    brackets.

    Going through ``print`` for the plain path keeps the log format ours rather
    than a side effect of Rich's width detection and soft-wrapping.
    """
    if interactive():
        console.print(f"{prefix}{message}", style=style, highlight=False)
    else:
        print(f"{prefix}{message}", flush=True)


def step(message: str) -> None:
    """A stage boundary: loading a model, starting a sweep, pooling results."""
    _say("step", "", f":: {message}")


def detail(message: str) -> None:
    """A subordinate fact about the stage just announced."""
    _say("detail", "   ", message)


def warn(message: str) -> None:
    """Something the reader must not scroll past --- a refused contrast, a null."""
    _say("warn", "", f"!! {message}")


def wrote(path: Path | str) -> None:
    """An artifact landed. Always the full path: it is the provenance."""
    _say("path", "   wrote ", str(path))


def track[T](
    items: Iterable[T], description: str, total: int | None = None
) -> Iterator[T]:
    """Iterate with a progress bar, or with periodic plain lines in a log.

    The non-interactive branch prints at most twenty lines regardless of how many
    items there are, which keeps a ``nohup`` log readable while still proving the
    job is alive.
    """
    sequence = list(items) if total is None else items
    total = total if total is not None else len(sequence)
    if total == 0:
        return

    if not interactive():
        every = max(total // 20, 1)
        print(f":: {description} ({total})", flush=True)
        for i, item in enumerate(sequence, 1):
            yield item
            if i % every == 0 or i == total:
                print(f"   {description} {i}/{total}", flush=True)
        return

    columns = (
        SpinnerColumn(style="cyan"),
        TextColumn("{task.description}", style="step"),
        BarColumn(bar_width=28, complete_style="cyan", finished_style="cyan"),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TextColumn("eta", style="detail"),
        TimeRemainingColumn(),
    )
    with Progress(*columns, console=console, transient=True) as progress:
        task = progress.add_task(description, total=total)
        for item in sequence:
            yield item
            progress.advance(task)
    console.print(f":: {description} ({total} done)", style="step")


def table(title: str, columns: list[str], rows: list[list[str]]) -> None:
    """A results table. Falls back to aligned plain text outside a terminal."""
    if not interactive():
        widths = [
            max(len(str(r[i])) for r in [columns, *rows]) for i in range(len(columns))
        ]
        print(f"\n{title}", flush=True)
        for row in [columns, *rows]:
            print("  " + "  ".join(str(c).ljust(w) for c, w in zip(row, widths,
                                                                  strict=True)),
                  flush=True)
        return
    grid = Table(title=title, title_style="step", title_justify="left",
                 header_style="detail", box=None, pad_edge=False)
    for i, column in enumerate(columns):
        grid.add_column(column, justify="left" if i == 0 else "right")
    for row in rows:
        grid.add_row(*(str(c) for c in row))
    console.print(grid)
