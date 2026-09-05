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

Python 3.11 to 3.13 is required. Use the **same Python interpreter for installation and launch**. An existing virtual environment can be reused: do not recreate it. A Git pull updates source files, not the packages installed in that environment.

### Full local AI installation in an existing environment

Run these commands from the repository root. Here, `python` must be the interpreter selected in your IDE or existing environment:

```bash
python -m pip install -r requirements.txt
python -m bodybuilder
```

An environment does not need to be activated when its interpreter is called explicitly:

```bash
/path/to/existing/environment/bin/python -m pip install -r requirements.txt
/path/to/existing/environment/bin/python -m bodybuilder
```

Replace the example interpreter path with your actual environment path. On Windows, use that environment's `Scripts\python.exe` instead.

`requirements.txt` installs the project in editable mode with its `ai` extra. This includes PyQt6, NumPy, OpenCV, Pillow, pillow-heif, PyTorch, Diffusers, Transformers, Accelerate, and safetensors, plus their dependencies. The authoritative dependency declarations and version ranges remain in `pyproject.toml`, avoiding duplicate lists that can drift apart. These requirements files are installation entry points, not fully pinned lockfiles.

Install a PyTorch build appropriate for your GPU before installing the project when the default PyTorch package is not suitable for your system. Installing Python packages does not install a GPU driver. Model weights download separately on first use.

### Updating an existing checkout

From the repository root, using your existing environment's interpreter:

```bash
git fetch origin &&
git switch main &&
git pull --ff-only origin main &&
python -m pip install -r requirements.txt &&
python -m bodybuilder
```

These commands do not discard local changes. Resolve any Git error before continuing with installation.

### Lightweight installation

```bash
python -m pip install -r requirements-minimal.txt
python -m bodybuilder
```

The lightweight installation includes PyQt6 and the image-processing dependencies, but not the local AI stack. It can analyze images, create provenance assets, and test geometric overlap stitching. Choose **Classical preview (no generative AI)** in the interface and disable **Enhance final images 2x** to avoid requesting the optional AI upscaler. It cannot plausibly invent large missing body or object regions.

### Missing PyQt6 or another dependency

`ModuleNotFoundError: No module named 'PyQt6'` means that the interpreter launching the application cannot import PyQt6. Install the requirements with that exact interpreter, rather than an unrelated `pip` executable:

```bash
/path/to/existing/environment/bin/python -m pip install -r requirements.txt
/path/to/existing/environment/bin/python -m pip check
/path/to/existing/environment/bin/python -c "from PyQt6.QtWidgets import QApplication; print('PyQt6 import OK')"
```

In an IDE, select the same interpreter and run the module `bodybuilder` rather than executing `src/bodybuilder/__main__.py` as a standalone script. Editable installation registers the `src` package without requiring a custom `PYTHONPATH`.

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
python -m pip install -r requirements-dev.txt
ruff check .
pytest
```

Install all AI and development dependencies with:

```bash
python -m pip install -r requirements.txt -r requirements-dev.txt
```

See [Architecture](docs/ARCHITECTURE.md) and [Fidelity and limitations](docs/FIDELITY_AND_LIMITATIONS.md).

## Responsible use

Only process photographs you are authorized to use. Do not present generated regions or synthetic variants as documentary originals. Keep the accompanying masks and manifests when results are shared or used for decisions.
