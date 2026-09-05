# BodyBuilder

Local PyQt6 reconstruction of a person or an object from complementary photographic fragments. Code and interface are in English.

## Update in your existing environment

From the repository directory, use the same Python interpreter that launches the application:

```bash
git pull --ff-only origin main
python -m bodybuilder
```

For a first installation or a dependency update, run `python -m pip install -r requirements.txt` before launching. Python 3.11-3.13 is supported. No new virtual environment is required. The batch/quality update adds no dependencies; the earlier version 0.3 dependency update is still required for native Florence-2 support.

## Restore the sources AND generate complete views

The desktop default is **Restore source photos + create complete views**, **5 synthetic views**, **Balanced**, and **2x output**. Select a folder of original fragments of one physical subject, select the result folder, and click **Reconstruct**. For three usable sources, this creates three restored source photos plus five complete synthetic views, not five outputs in total and not six synthetic views.

The **Synthetic views** count is visible on the main screen and can be set to 10 (up to 16). **Restore source photos only** and **Create complete views only** remain separate choices. Sources are restored first, with the original perspective and working-resolution observed pixels protected. New views use a fresh complete-subject composition, not the cropped source framing. Both stages share the original evidence; generated restorations are not reused as reference evidence.

**Fast / Balanced / High quality** change the actual working resolution, generation steps, and number and depth of local detail passes. Fast uses 768 pixels, 24 steps and up to two regional passes; Balanced uses 1024 pixels, 40 steps and up to four regional passes; High quality uses 1024 pixels, 60 steps and up to eight regional passes. All retain original-reference conditioning, observed-pixel protection and validation. More computation is not a guarantee of a more accurate identity. See [Quality presets and batch outputs](docs/QUALITY_AND_BATCHES.md) for exact budgets, tests and compatibility.

2x output is enabled by default. Its resource override remains under **Advanced options**. Doubling dimensions is not the same as recovering missing real detail: the photographed areas are deterministically resampled, not generatively repainted. The UI shows planned and saved counts for each output type.

**Advanced options** also contains an optional short **Subject hint**, such as `person`, `bicycle`, or `vase`, for ambiguous detection. Use original fragments of the same instance: common-category detection cannot prove that two similar objects or people are the same subject.

## Complementary details and provenance

A local Florence-2 detector proposes subjects and parts. Generic parts and targeted person details use the same evidence selection and regional refinement mechanism. Transparent or explicitly masked regions do not count as visible evidence. Obvious empty/erased part crops are excluded from detailed references.

Genuinely overlapping fragments are tested with masked SIFT/RANSAC and photometric agreement before a non-generative assembly is accepted. Different poses are not forced into one planar collage. During generation, localized parts can be refined from different original photographs using spatial reference masks; for example, eyes from one source and mouth details from another.

Whole-view framing is checked, with bounded correction for suspected clipping. Uncertain framing is retained as an explicitly review-required image rather than blocking subsequent views. See [Framing recovery](docs/FRAMING_RECOVERY.md). A conditional eyewear check discourages introducing glasses when visible eye references do not support them. These are imperfect model-based checks, not guarantees of exact facial identity, unseen anatomy, or geometry.

All files are inspected. Diffusion uses at most 16 selected references per output; regional pass budgets depend on the quality preset. Selected original paths, part coordinates, registration transforms, active quality budgets and uncertainty notices are recorded. See [Joint reconstruction](docs/JOINT_RECONSTRUCTION.md) for the underlying algorithm and limitations; the newer quality document supersedes its fixed four-pass budget.

## Missing regions inside a photograph

Select an original and click **Mark missing area in selected photo**. Paint the opaque erasure or obstruction to reconstruct. A sidecar `photo.jpg.mask.png` is saved; the original is not overwritten. White in the sidecar means missing, black means keep. Transparent regions are recognized automatically. Ordinary black or white object surfaces are not automatically deleted.

## Output

```text
BodyBuilder_YYYYMMDD_HHMMSS/
  images/
    source__...png            # restored original photos
    subject__whole_...png     # new complete synthetic views
  diagnostics/               # source parts, drafts, masks, metadata
  subject_evidence.json      # common-subject candidate and selected part sources
  run_manifest.json          # planned/saved counts, quality budget, status, hashes
  bodybuilder.log
```

The result tab labels restorations and synthetic views separately. Diagnostic masks are not displayed as final photographs. Do not feed generated results back as original evidence. Earlier completed images and error diagnostics are retained when a later stage fails or is cancelled.

## Local AI and resource requirements

SDXL inpainting plus IP-Adapter handles generation; native Florence-2 handles subject/part localization. Models download on first use. Photographs are not uploaded by BodyBuilder. Remote model Python code is not enabled. Detection runs on CPU; generation uses the selected compute device. The analysis, regional passes and default 2x enhancement make a combined batch slower than a single outpainting pass. CPU generation can be very slow.

Initialization, non-finite outputs, blank images and cancellations are handled explicitly. Numerical, framing and semantic retries remain bounded. An active model download cannot stop immediately. No fixed duration or speedup is promised for the presets.

## Fidelity limits

New compositions are fully generated. They are not recovered originals or a persistent 3D model. Visible original regions in source-repair mode are preserved at the AI working resolution; inputs are resized for that canvas. 2x output restores those observed areas with deterministic Lanczos resampling after enhancement.

Unseen sides, body parts and fine details remain hypotheses. A partial photograph cannot uniquely determine all of them. Similar-looking sources can be mismatched, and semantic part localization can fail. Review the evidence report and actual outputs before relying on them. Only process photographs you are authorized to use.

## Development

```bash
python -m pip install -e ".[dev]"
ruff check .
python -m compileall -q src tests
QT_QPA_PLATFORM=offscreen pytest
```

CI includes real feature-registration tests, controlled pipeline/UI regressions, actual Diffusers API tests, and a real-weight detector smoke test on a synthetic fixture. Batch/quality tests exercise orchestration, masks, exports and parameter propagation with controlled generation doubles. These tests do not establish reconstruction fidelity or a monotonic visual-quality improvement on private photographs. No user photographs are included in the repository or CI.
