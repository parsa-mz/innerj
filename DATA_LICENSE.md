# Licensing of the released data

The **code** in this repository is MIT licensed (see `LICENSE`).

The **JGateBench records** are licensed
**[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)**, not MIT.

## Why

Every JGateBench prompt quotes a passage from **FLORES-200** verbatim — the generator
never synthesises text, which is a design invariant of the benchmark. FLORES-200 is
released under CC BY-SA 4.0, and share-alike propagates to a derived dataset, so the
records inherit those terms. Attribution and share-alike apply to anyone redistributing
them or a work derived from them.

This covers the generated record files (`*.jsonl` under the language and tracking
families) and any redistribution of the passages inside them. It does not cover the
generator, the analysis code or the figure code, which stay MIT.

## Attribution

FLORES-200 is from *No Language Left Behind: Scaling Human-Centered Machine
Translation* (NLLB Team, 2022), which also introduces the benchmark, building on
FLORES-101 (Goyal et al., 2022) and the Nepali–English / Sinhala–English datasets
(Guzmán et al., 2019). The copy consumed here is `haoranxu/FLORES-200` at revision
`8ecaf1bb`.

```bibtex
@article{nllb2022,
  title  = {No Language Left Behind: Scaling Human-Centered Machine Translation},
  author = {{NLLB Team}},
  journal = {arXiv preprint arXiv:2207.04672},
  year   = {2022}
}
```

## Other upstream terms

- **wikitext** (`Salesforce/wikitext`, `wikitext-103-raw-v1`) is used only as the fit
  corpus for the lenses fitted here, and carries its own terms.
- **Model weights** are never redistributed. Only measurements taken from them appear
  here, and each checkpoint keeps its own licence.
- **`jacobian-lens/`** is cloned, not authored here, and carries its own terms.
