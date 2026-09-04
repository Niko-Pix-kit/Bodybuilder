# BodyBuilder

BodyBuilder is a local PyQt6 desktop application for completing badly cropped or fragmented photographs. It can work with people, faces, hands, limbs, animals, objects, and scenes.

The application deliberately separates two different operations:

1. **Source completion** extends each real source photograph without letting the generative model repaint its visible region.
2. **Synthetic variants** create new views or poses from the reference set. These are generated hypotheses and are never presented as recovered evidence.

No model can recover pixels that were never captured. BodyBuilder uses the available photographs as evidence, aligns overlapping fragments when possible, and generates plausible missing content. The result can be useful, but it is not proof of what the missing area originally contained.

## Main features

- English PyQt6 interface.
- Input-folder and output-folder workflow.
- Recursive JPEG, PNG, WebP, TIFF, BMP, HEIC, and HEIF discovery with EXIF orientation handling.
- Blur, exposure, detail, and resolution quality analysis.
- Optional geometric stitching of overlapping fragments before AI generation.
- Local SDXL inpainting/outpainting through Hugging Face Diffusers.
- IP-Adapter image conditioning from a contact sheet of all fragments.
- Person-aware reference conditioning when faces can be detected.
- Configurable completions per source and optional pose/view variants.
- Optional Swin2SR 2x enhancement with a conservative Lanczos fallback.
- Exact restoration of observed pixels at the AI working resolution after diffusion generation. Inputs are resized for the AI canvas; with 2x output enabled, the observed region is deterministically resampled and is never generatively enhanced.
- Per-image observed/generated masks, comparison sheets, seeds, and JSON manifests.
- Background processing, cancellation, progress reporting, and crash-safe error dialogs.
- No telemetry and no photo upload performed by BodyBuilder itself.

## Installation

Python 3.11 to 3.13 is supported.

### Full local AI installation

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[ai]"
bodybuilder
```

On Windows, activate with:

```powershell
.venv\Scripts\activate
```

Install a PyTorch build appropriate for your GPU before installing the project when the default PyTorch package is not suitable for your system.

### Lightweight installation

```bash
python -m pip install -e .
bodybuilder
```

The lightweight installation can analyze images, create provenance assets, and test geometric overlap stitching. Choose **Classical preview (no generative AI)** in the interface. It cannot plausibly invent large missing body or object regions.

## First run

The local AI backend downloads model files from Hugging Face on first use. The default stack is:

- `diffusers/stable-diffusion-xl-1.0-inpainting-0.1`
- `h94/IP-Adapter`
- `caidas/swin2SR-classical-sr-x2-64` when 2x enhancement is enabled
- `facebook/dinov2-small` only when automatic multi-subject grouping is requested

Set `HF_HOME` to move the Hugging Face model cache:

```bash
export HF_HOME=/path/to/model-cache
```

## Recommended workflow

1. Put all fragments of the same person or object in one folder.
2. Open BodyBuilder and select that folder.
3. Select an output folder.
4. Leave **Treat all images as one subject** enabled unless the folder contains unrelated subjects.
5. Click **Analyze source** and inspect the quality scores and detected face count.
6. Run one completion per source first.
7. Increase completion count or add synthetic variants only after checking identity consistency.
8. Inspect the generated mask and manifest beside every output.

A close-up face cannot be converted into a geometrically faithful full-body photograph while preserving the original close-up pixels. For that case, use **Synthetic variants**; BodyBuilder will mark the entire result as generated.

## Output structure

Each run creates a time-stamped directory:

```text
BodyBuilder_YYYYMMDD_HHMMSS/
├── run_manifest.json
├── bodybuilder.log
└── subject_001/
    ├── analysis.json
    ├── reference_board.jpg
    ├── face_reference_board.jpg          # when faces were detected
    ├── diagnostics/
    │   ├── source_001_canvas.png
    │   ├── source_001_observed_mask.png
    │   └── source_001_generated_mask.png
    ├── completions/
    │   ├── source_001__completion_01.png
    │   ├── source_001__completion_01__comparison.jpg
    │   └── source_001__completion_01.json
    └── variants/
        ├── variant_01__standing_front.png
        └── variant_01__standing_front.json
```

Mask convention:

- `observed_mask`: white pixels came from input photographs.
- `generated_mask`: white pixels were synthesized or classically filled.

## Fidelity controls

- **Reference fidelity** controls IP-Adapter influence. Too low can cause identity drift; too high can copy framing or create visual artifacts.
- **Denoising strength** controls how aggressively the missing region is generated.
- **Steps** trades speed for refinement.
- **Completion margin** determines how far each source is extended.
- **Stitch overlapping fragments** tries feature-based alignment before generation.

The application does not perform face recognition, identify a person, or compare the subject against an external database.

## Development

```bash
python -m pip install -e ".[dev]"
ruff check .
pytest
```

Install all AI and development dependencies with:

```bash
python -m pip install -e ".[ai,dev]"
```

See [Architecture](docs/ARCHITECTURE.md) and [Fidelity and limitations](docs/FIDELITY_AND_LIMITATIONS.md).

## Responsible use

Only process photographs you are authorized to use. Do not present generated regions or synthetic variants as documentary originals. Keep the accompanying masks and manifests when results are shared or used for decisions.
