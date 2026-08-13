"""What makes a latent variable reach a language model's self-report?

Given that a model already represents some latent ``z``, what makes it enter the
verbalizable
workspace when the task requires flexible reuse? The source paper leaves the mechanism
open;
this package is the benchmark and causal toolkit, and the answer it arrives at is
transport,
not admission.

Lens algebra comes from the authors' ``jlens``, vendored at ``jacobian-lens/``.
**Nothing
heavy is imported here**, so ``innerj --help`` and :mod:`innerj.config` do not pull
torch.
"""

from __future__ import annotations

__version__ = "1.0.0"
__all__ = ["__version__"]
