"""What makes a latent variable reach a language model's self-report?

Given that a model already represents some latent quantity ``z``, what makes that
quantity enter its verbalizable workspace when the task requires flexible reuse?
The source paper establishes that the automatic/flexible distinction exists and
leaves the mechanism open; this package is the benchmark and causal toolkit for
answering it. The answer it arrives at is transport, not admission.

Lens algebra comes from the authors' own ``jlens`` package, vendored at
``jacobian-lens/``, so the estimator is parity-correct by construction. This
package adds the guards it lacks, the workspace-entry dependent variables, the
task generators, component-level interventions, and the staged pipeline.

**Nothing heavy is imported here.** Re-exporting from :mod:`innerj.model` would
pull torch and transformers into every consumer, including ``innerj --help`` and
:mod:`innerj.config`, which need neither. Import from the module that owns the
name: ``from innerj.model import load_model``, ``from innerj.analysis.readout
import percentile_rank``.
"""

from __future__ import annotations

__version__ = "1.0.0"
__all__ = ["__version__"]
