# MindLoop OCR Lab

A standalone, reproducible OCR research component and benchmark, not a general-purpose compute proxy. It develops shared document preprocessing and recognition strategies on synthetic German document images and an expressly authorized, de-identified paired document corpus. It has its own experimental memory: versioned ledgers of measured configuration outcomes, inspired by bounded MindLoop learning. It does not depend on the private MindLoop repositories or any external LLM.

## First vertical slice — reference engine only

Tesseract German OCR, deterministic generated statement/form text, rotation, blur, downsampling and low contrast; candidate preprocessing and segmentation strategies. Truth comes from the generator, not OCR itself. Character error rate (CER) preserves umlauts, case and punctuation; exact critical tokens include amounts, years, umlaut words, ẞ, m², m³, § and €. The selection policy prioritizes related untested variants of the best measured candidate. Research uses strict paired field gains with no lost baseline hits on any page/modality, and no synthetic dev or validation cohort regression. Synthetic-only smoke keeps its CER improvement gate. Every evaluated candidate is recorded, including rejected variants. No result claims to improve any Android app. See [cross-project experience and next experiments](docs/prior-ocr-experience.md).

Tesseract supplies a cheap, deterministic reference, not the chosen application engine. **Priority P0** is a like-for-like comparison with compact German-capable PaddleOCR (including ONNX deployment) and bundled Android ML Kit, followed by other neural/document models only if they fit the measured resource budget. The comparison adapters are not yet implemented; the current CI ledger runs the reference engine only. See the P0 experiment design in the cross-project evidence document.

## Run locally

On Ubuntu 24.04 install `tesseract-ocr tesseract-ocr-deu python3-pil fonts-dejavu-core poppler-utils` from the distribution. Run:

```sh
python3 -m unittest -v
python3 lab.py --lang deu --max-candidates 2 --corpus fixtures
```

Output: `results/<context-sha256>/ledger.json`. A context binds exact lab and benchmark source, corpus digest, Poppler, OCR engine/language model, Pillow and font. Do not overwrite or delete historical evidence. An incompatible context starts a distinct ledger. A recorded candidate is not repeated within its context; exhausted catalogs end with `EXHAUSTED`. The validation split is fixed and reused for lab screening; independent consumer-specific acceptance is still needed to prevent overfitting. The eleven-variant catalog is deliberately finite. Add new hypotheses through a reviewed change, not by launching an endless job or generating load. No arbitrary uploaded input or code execution from issues.

## CI

Standard Ubuntu CPU runners only. Push/PR smoke runs read-only. A daily trusted-main research run (or an owner manual dispatch) evaluates at most two novel candidates, with one concurrent research job and a 20-minute job timeout. Each OCR invocation has its own 25-second timeout and one OCR thread. Results are committed only beneath `results/`; no workflow or application source changes from the research job. If a catalog is exhausted, scheduled runs skip dependency installation and OCR until the research code changes; manual runs can requalify a changed environment. No secrets, private source, private documents, paid compute, external model call or cross-repository credentials. GitHub schedules are best-effort; manual dispatch is the recovery route.

## Consumer handoff

The ledger is the versioned interoperability contract. A `LAB_CANDIDATE` is a suggestion, not an approved release. Pin the exact public commit, context, candidate config and full ledger; review dependencies/license and reproduce on the actual consumer OCR engine and held-out local fixtures. Preserve offline execution. Run app-specific regression tests and Android target acceptance before merging any integration PR. The apps can reject a lab candidate without modifying its recorded evidence. Do not place any private app code, original private scans or production output in this public repository.

Next experiments: multipass voting with calibrated confidence, layout/checkbox detection with positional ground truth, multi-page grouping, amount normalization and larger licensed/synthetic held-out datasets. Each must have a bounded falsifiable experiment and tests. This initial slice measures text OCR only; it does not yet measure handwriting, field localization, or multi-page capture.
# Paired camera-photo / scan-PDF regression corpus

`document_benchmark.py` evaluates five or more paired pages as a single document group, with a manually transcribed list of critical fields per page. The matching PDF and photos share expected values; they are not independent train/test samples. Reports contain only counts and a corpus digest, not OCR text. This is a field-presence metric, not full-document CER or a test of correct table-field association.

Only privacy-reviewed, explicitly cleared fixtures belong in `fixtures/`. The manifest schema is `{"schema":1,"documents":[{"id":"example-01","scan":"scan.pdf","pages":[{"photo":"photo-01.png","fields":["fictitious field"]}]}]}`. The PDF must be image-only with one raster image per page. Run `/usr/bin/python3 document_benchmark.py fixtures --lang deu --candidate baseline` after installing `poppler-utils`. The smoke and research workflows verify the published file hashes. Research includes the paired corpus explicitly; every candidate must improve at least one exact field and preserve every previously matched field. The five pages are one underlying document (development/regression), not an independent holdout. Reports retain per-field hit indices and counts, not raw recognized text. No candidate is accepted for either Android app from this gate alone.

