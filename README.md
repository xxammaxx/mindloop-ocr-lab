# MindLoop OCR Lab

A standalone, reproducible OCR research component and benchmark, not a general-purpose compute proxy. It develops shared document preprocessing and recognition strategies on synthetic German document images and an expressly authorized, de-identified paired document corpus. It has its own experimental memory: versioned ledgers of measured configuration outcomes, inspired by bounded MindLoop learning. It does not depend on the private MindLoop repositories or any external LLM.

## First vertical slice — reference engine only

Tesseract German OCR, deterministic generated statement/form text, rotation, blur, downsampling and low contrast; candidate preprocessing and segmentation strategies. Truth comes from the generator, not OCR itself. Character error rate (CER) preserves umlauts, case and punctuation; exact critical tokens include amounts, years, umlaut words, ẞ, m², m³, § and €. The selection policy prioritizes related untested variants of the best measured candidate. Research uses strict paired field gains with no lost baseline hits on any page/modality, and no synthetic dev or validation cohort regression. Synthetic-only smoke keeps its CER improvement gate. Every evaluated candidate is recorded, including rejected variants. No result claims to improve any Android app. See [cross-project experience and next experiments](docs/prior-ocr-experience.md).

Tesseract is a historical CPU reference and manual-only research path. **Priority P0 is the existing bundled Google ML Kit Latin SDK 16.0.1**, already used by both consumers. The ML Kit Android research workflow compares baseline and one new preprocessing variant each scheduled run on API 35 x86_64. It separately exercises Umliva's NV21/even-padding route and Feldexa's gray8/ARGB bitmap route in an isolated benchmark APK. No new OCR engine, dependency or model is introduced into either consumer app. Results are emulator evidence, not proof of either production adapter or physical-device acceptance.

## Run locally

On Ubuntu 24.04 install `tesseract-ocr tesseract-ocr-deu python3-pil fonts-dejavu-core poppler-utils` from the distribution. Run:

```sh
python3 -m unittest -v
python3 lab.py --lang deu --max-candidates 2 --corpus fixtures
```

Output: `results/<context-sha256>/ledger.json`. A context binds exact lab and benchmark source, corpus digest, Poppler, OCR engine/language model, Pillow and font. Do not overwrite or delete historical evidence. An incompatible context starts a distinct ledger. A recorded candidate is not repeated within its context; exhausted catalogs end with `EXHAUSTED`. The validation split is fixed and reused for lab screening; independent consumer-specific acceptance is still needed to prevent overfitting. The eleven-variant catalog is deliberately finite. Add new hypotheses through a reviewed change, not by launching an endless job or generating load. No arbitrary uploaded input or code execution from issues.

## CI

Standard Ubuntu runners only. Push/PR reference smoke runs read-only. The daily trusted-main ML Kit run uses one API 35 emulator, one new candidate vs its current-run control, a 25-minute job budget and 30 seconds per OCR call. The old Tesseract research workflow is manual only. Both share a single concurrency group. Results are committed only beneath `results/`; no workflow or app source changes from research jobs. An exhausted or infrastructure-blocked ML Kit catalog skips expensive work until the experiment context changes. No paid compute, external image inference, private source, private documents or cross-repository credentials. Each ML Kit pipeline must gain a paired field and retain every old hit and synthetic per-case CER; aggregate-only gains are not promoted. Runtime SDK AAR SHA-256s and emulator fingerprint are recorded. Raw OCR output remains ephemeral; only counts, hit indices and timing are persisted. GitHub schedules are best-effort; manual dispatch is the recovery route.

After a successful research job has committed and pushed the score ledger, a dependent `continue` job in the same `ML Kit OCR research` workflow explicitly dispatches the next eligible experiment. It excludes its own finishing run from the active-run check, but any other active or queued run blocks another dispatch. Exhausted or blocked catalogs stop; failures or missing persisted evidence do not dispatch. The separate `Continue ML Kit OCR research` workflow is manual recovery only; the daily schedule is retained as a fallback. The orchestration-only transition preserves the two prior results in the new context with an explicit source-ledger provenance record; the original ledger remains unchanged.

## Consumer handoff

The ledger is the versioned interoperability contract. A `LAB_CANDIDATE` is a suggestion, not an approved release. Pin the exact public commit, context, candidate config and full ledger; review dependencies/license and reproduce on the actual consumer OCR engine and held-out local fixtures. Preserve offline execution. Run app-specific regression tests and Android target acceptance before merging any integration PR. The apps can reject a lab candidate without modifying its recorded evidence. Do not place any private app code, original private scans or production output in this public repository.

Next experiments: multipass voting with calibrated confidence, layout/checkbox detection with positional ground truth, multi-page grouping, amount normalization and larger licensed/synthetic held-out datasets. Each must have a bounded falsifiable experiment and tests. This initial slice measures text OCR only; it does not yet measure handwriting, field localization, or multi-page capture.
# Paired camera-photo / scan-PDF regression corpus

`document_benchmark.py` evaluates five or more paired pages as a single document group, with a manually transcribed list of critical fields per page. The matching PDF and photos share expected values; they are not independent train/test samples. Reports contain only counts and a corpus digest, not OCR text. This is a field-presence metric, not full-document CER or a test of correct table-field association.

Only privacy-reviewed, explicitly cleared fixtures belong in `fixtures/`. The manifest schema is `{"schema":1,"documents":[{"id":"example-01","scan":"scan.pdf","pages":[{"photo":"photo-01.png","fields":["fictitious field"]}]}]}`. The PDF must be image-only with one raster image per page. Run `/usr/bin/python3 document_benchmark.py fixtures --lang deu --candidate baseline` after installing `poppler-utils`. The smoke and research workflows verify the published file hashes. Research includes the paired corpus explicitly; every candidate must improve at least one exact field and preserve every previously matched field. The five pages are one underlying document (development/regression), not an independent holdout. Reports retain per-field hit indices and counts, not raw recognized text. No candidate is accepted for either Android app from this gate alone.

