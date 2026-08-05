"""``innerj`` --- one entry point for every experiment.

Each subcommand is a module in this package exposing ``main()``. They are imported
lazily, one at a time, because importing any of them pulls in torch and
transformers: ``innerj --help`` should not cost eight seconds and a CUDA context.

The commands are listed in dependency order, which is also the order the paper
runs them in. Anything that reads a dataset takes ``--records``; anything that
patches takes ``--pairs`` and ``--last-n``; the shared flags live in
:mod:`innerj.cli.common`.
"""

from __future__ import annotations

import importlib
import sys

#: Subcommand -> (module, one-line description). Order is the order to run them.
COMMANDS: dict[str, tuple[str, str]] = {
    "build-dataset": ("build_dataset",
                      "generate a task family, enforcing the invariants"),
    "stage1": ("stage1", "is workspace entry different from availability?"),
    "probe": ("probe", "is the latent variable decodable in every arm?"),
    "screen": ("screen", "which components move the readout?"),
    "sweep": ("sweep", "where can a value be installed, at matched distance?"),
    "attention": ("attention", "is the gather's attention route demand-dependent?"),
    "specificity": ("specificity", "does a patch favour the donor's concept?"),
    "counterfactual": ("counterfactual",
                       "does it change the answer, against the control?"),
    "ablate": ("ablate", "is the component selectively necessary?"),
    "mediate": ("mediate", "does the behaviour run through J-space?"),
    "fit-lens": ("fit_lens", "fit a Jacobian lens for a checkpoint"),
    "gauge": ("gauge", "check which diagnostics survive a reparameterisation"),
    "example": ("example", "dump what the lens reads for one instance"),
    "figures": ("figures", "rebuild the paper deck from artifacts"),
    "audit": ("audit", "check every number in the paper against the artifacts"),
}


def usage() -> str:
    width = max(len(name) for name in COMMANDS)
    lines = [
        "usage: innerj <command> [options]",
        "",
        "commands, in the order they are run:",
        "",
    ]
    lines += [f"  {name:<{width}}  {help}" for name, (_, help) in COMMANDS.items()]
    lines += ["", "`innerj <command> --help` for a command's own options."]
    return "\n".join(lines)


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] in {"-h", "--help"}:
        print(usage())
        raise SystemExit(0 if len(sys.argv) > 1 else 1)

    command = sys.argv[1]
    if command not in COMMANDS:
        near = [c for c in COMMANDS if c.startswith(command[:3])]
        hint = f"  did you mean: {', '.join(near)}\n" if near else ""
        raise SystemExit(f"unknown command {command!r}\n{hint}\n{usage()}")

    module, _ = COMMANDS[command]
    # argparse reads sys.argv, and the subcommand name is not one of its arguments.
    sys.argv = [f"innerj {command}", *sys.argv[2:]]
    importlib.import_module(f"innerj.cli.{module}").main()


if __name__ == "__main__":
    main()
