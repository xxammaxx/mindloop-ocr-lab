# MindLoop OCR Lab

A standalone, reproducible OCR research component and benchmark, not a general-purpose compute proxy. It develops shared document preprocessing and recognition strategies on synthetic German document images. It has its own experimental memory: versioned ledgers of measured configuration outcomes, inspired by bounded MindLoop learning. It does not depend on the private MindLoop repositories or any external LLM.

## First vertical slice

Tesseract German OCR, deterministic generated statement/form text, rotation, blur, downsampling and low contrast; candidate preprocessing and segmentation strategies. Truth comes from the generator, not OCR itself. Character error rate (CER) preserves umlauts, case and punctuation; exact critical tokens include amounts, years and umlaut words. The selection policy prioritizes related untested variants of the best measured development candidate. A strict development and independent validation-split comparison requires improved overall CER and no deterioration in cohort CER or critical-token accuracy. Every evaluated candidate is recorded, including rejected variants. No result claims to improve any Android app.

## Run locally

On Ubuntu 24.04 install `tesseract-ocr tesseract-ocr-deu python3-pil fonts-dejavu-core` from the distribution. Run:

```sh
python3 -m unittest -v
python3 lab.py --lang deu --max-candidates 2
```

Output: `results/<context-sha256>/ledger.json`. A context binds exact lab source, OCR engine/language model, Pillow and font. Do not overwrite or delete historical evidence. An incompatible context starts a distinct ledger. A complete candidate is never repeated within its context; exhausted catalogs end with `EXHAUSTED`. The validation split is fixed and reused for lab screening; independent consumer-specific acceptance is still needed to prevent overfitting. The current nine-variant catalog is deliberately finite. Add new hypotheses through a reviewed change, not by launching an endless job or generating load. No arbitrary uploaded input or code execution from issues.

## CI

Standard Ubuntu CPU runners only. Push/PR smoke runs read-only. A daily trusted-main research run (or an owner manual dispatch) evaluates at most two novel candidates, with one concurrent research job and a 20-minute job timeout. Each OCR invocation has its own 25-second timeout and one OCR thread. Results are committed only beneath `results/`; no workflow or application source changes from the research job. If a catalog is exhausted, scheduled runs skip dependency installation and OCR until the research code changes; manual runs can requalify a changed environment. No secrets, private source, private documents, paid compute, external model call or cross-repository credentials. GitHub schedules are best-effort; manual dispatch is the recovery route.

## Consumer handoff

The ledger is the versioned interoperability contract. A `LAB_CANDIDATE` is a suggestion, not an approved release. Pin the exact public commit, context, candidate config and full ledger; review dependencies/license and reproduce on the actual consumer OCR engine and held-out local fixtures. Preserve offline execution. Run app-specific regression tests and Android target acceptance before merging any integration PR. The apps can reject a lab candidate without modifying its recorded evidence. Do not place any private app code, real scan or production output in this public repository.

Next experiments: multipass voting with calibrated confidence, layout/checkbox detection with positional ground truth, multi-page grouping, amount normalization and larger licensed/synthetic held-out datasets. Each must have a bounded falsifiable experiment and tests. This initial slice measures text OCR only; it does not yet measure handwriting, field localization, or multi-page capture.
