# BodyBuilder

Local PyQt6 reconstruction of a person or an object from complementary photographic fragments. Code and interface are in English.

## Update in your existing environment

From the repository directory, use the same Python interpreter that launches the application:

```bash
git pull --ff-only origin main
python -m pip install -r requirements.txt
python -m bodybuilder
```

Python 3.11-3.13 is supported. No new virtual environment is required. Version 0.3 requires the dependency update for native Florence-2 support.

## Choose the result, not technical diffusion parameters

The desktop default is **Complete subject - new full view**. Select a source folder containing original fragments of one physical subject, select the result folder, and click **Reconstruct**. The application proposes the common subject, collects complementary parts, and generates a new whole-subject composition rather than inheriting the cut-off framing of the photographs. Optional extra synthetic views use the same original evidence.

Choose **Restore each original photo using all fragments** to preserve the existing photo's perspective and working-resolution observed pixels while repairing its missing areas. This is a separate operation from generating a complete subject in a new pose.

**Advanced options** contains an optional short **Subject hint**, such as `person`, `bicycle`, or `vase`, for ambiguous detection. Use original fragments of the same instance: common-category detection cannot prove that two similar objects or people are the same subject.

## Complementary details and provenance

A local Florence-2 detector proposes subjects and parts. Generic parts and targeted person details use the same evidence selection and regional refinement mechanism. Transparent or explicitly masked regions do not count as visible evidence. Obvious empty/erased part crops are excluded from detailed references.

Genuinely overlapping fragments are tested with masked SIFT/RANSAC and photometric agreement before a non-generative assembly is accepted. Different poses are not forced into one planar collage. During generation, localized parts can be refined from different original photographs using spatial reference masks; for example, eyes from one source and mouth details from another.

Whole-view framing is checked, and a bounded retry is used for detected clipping. A conditional eyewear check discourages introducing glasses when visible eye references do not support them. These are imperfect model-based checks, not guarantees of exact facial identity, unseen anatomy, or geometry.

All files are inspected. Diffusion uses at most 16 selected references and four regional refinements per output. The selected original paths, part coordinates, registration transforms and uncertainty notices are recorded. See [Joint reconstruction](docs/JOINT_RECONSTRUCTION.md) for the algorithm, budgets and limitations.

## Missing regions inside a photograph

Select an original and click **Mark missing area in selected photo**. Paint the opaque erasure or obstruction to reconstruct. A sidecar `photo.jpg.mask.png` is saved; the original is not overwritten. White in the sidecar means missing, black means keep. Transparent regions are recognized automatically. Ordinary black or white object surfaces are not automatically deleted.

## Output

```text
BodyBuilder_YYYYMMDD_HHMMSS/
  images/                    # final photographs only
  diagnostics/               # source parts, assemblies, drafts, masks, metadata
  subject_evidence.json      # common-subject candidate and selected part sources
  run_manifest.json          # status, source hashes, versions, outputs, warnings
  bodybuilder.log
```

The result tab and **Open reconstructed images** show photographs, not diagnostic masks. Do not feed generated results back as original evidence. The application retains completed images and error diagnostics when a later stage fails.

## Local AI and resource requirements

SDXL inpainting plus IP-Adapter handles generation; native Florence-2 handles subject/part localization. Models download on first use. Photographs are not uploaded by BodyBuilder. Remote model Python code is not enabled. Detection runs on CPU; generation uses the selected compute device. The extra analysis and regional passes make this workflow slower and more memory-intensive than a single outpainting pass. CPU generation can be very slow.

Initialization, non-finite outputs, blank images and cancellations are handled explicitly. Numerical retry and framing retry are bounded. An active model download cannot stop immediately.

## Fidelity limits

New compositions are fully generated. They are not recovered originals or a persistent 3D model. Visible original regions in source-repair mode are preserved at the AI working resolution; inputs are resized for that canvas. Optional 2x output restores those observed areas with deterministic Lanczos resampling.

Unseen sides, body parts and fine details remain hypotheses. A partial photograph cannot uniquely determine all of them. Similar-looking sources can be mismatched, and semantic part localization can fail. Review the evidence report and actual outputs before relying on them. Only process photographs you are authorized to use.

## Development

```bash
python -m pip install -e ".[dev]"
ruff check .
python -m compileall -q src tests
QT_QPA_PLATFORM=offscreen pytest
```

CI includes real feature-registration tests, controlled pipeline/UI regressions, actual Diffusers API tests, and a real-weight detector smoke test on a synthetic fixture. These tests do not establish reconstruction fidelity on private photographs. No user photographs are included in the repository or CI.
