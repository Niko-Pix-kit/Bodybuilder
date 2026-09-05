# BodyBuilder

A local PyQt6 application for completing cropped photographs using other fragments of the same person or object as references. The interface and code are in English.

## Update and launch in an existing Python environment

From your repository folder, using the Python interpreter already selected in your IDE:

```bash
git pull --ff-only origin main
python -m pip install -r requirements.txt
python -m bodybuilder
```

No new virtual environment is needed. Python 3.11–3.13 is supported. `requirements.txt` installs the editable application and its local AI dependencies from `pyproject.toml`.

## Simple workflow

Select **Source folder**, select **Save results to**, and click **Reconstruct**. Put only photographs of the same person or object in a source folder. Person mode, automatic device selection and one completion per source are the defaults. Analysis runs automatically. Technical options are hidden under **Advanced options**; old saved engine settings cannot silently select the non-generative preview.

**Cropped edges:** the application extends the canvas around the photographed portion. It does not require a detectable complete face. The default frame is portrait; frame and extension can be changed under Advanced options. A thin fragment will not create a degenerate, thin AI canvas.

**Missing areas inside an image:** select a source photo and click **Mark missing area in selected photo**. Paint only the part to reconstruct. Marks are saved alongside the original as `photo.jpg.mask.png` (white = missing, black = keep). The original is not overwritten. Transparent PNG regions are also treated as missing. Ordinary black/white photographic content is not automatically erased: use the marking tool for opaque obstruction or blank patches.

**Additional poses/views:** optional under Advanced options. These are entirely synthetic, not restored photographs. Completing a close-up and generating a full-body view are different tasks; the application cannot infer an unseen body faithfully from one facial fragment.

## Results, not masks

Each run has this structure:

```text
BodyBuilder_YYYYMMDD_HHMMSS/
  images/                  # completed photographs only
  diagnostics/             # evidence masks, working canvases, per-image metadata
  bodybuilder.log          # processing details and errors
  run_manifest.json        # status, sources, SHA-256 hashes, versions and outputs
```

The **Reconstructed images** tab and **Open reconstructed images** button show final images only. Black/white diagnostic masks are never shown as reconstructed photographs. Source scanning excludes explicit masks and previous run folders.

The developer-only classical backend writes to `diagnostic_previews/`, not `images/`. It is not offered in the desktop reconstruction workflow and is never used as an automatic fallback.

## AI and failure handling

The local backend uses SDXL inpainting and IP-Adapter Plus. Its ViT-H encoder is explicitly loaded from `h94/IP-Adapter/models/image_encoder`; the different encoder under `sdxl_models/image_encoder` is not compatible with these Plus weights. Each reference is encoded separately and padded without center-cropping away fragment edges. Up to 16 references are used per source; the source itself is always included and the exact list is recorded.

The reference adapter must load successfully. The application stops with an error rather than quietly generating without the reference photographs. Floating-point pixels are checked before image conversion. Empty, almost uniform, mask-like or unchanged generated regions are rejected, with one full-precision retry. Repeated failure does not export a blank image as success. This is a technical sanity check, not an assessment of identity accuracy. A genuinely featureless region can also trigger the conservative check; inspect the log in that case.

Models download on first use. Photos are processed locally and are not uploaded by BodyBuilder. CPU inference is supported but can be very slow. Cancellation is checked between stages and diffusion steps; an active model download/loading call cannot be interrupted immediately.

## Fidelity limits

Observed pixels are restored **at the AI working resolution** after generation. Inputs are resampled to fit that canvas. Optional 2x enhancement does not generatively repaint observed details: those areas are restored with deterministic Lanczos resizing. It cannot recover facial details that were never recorded. Different poses are used as references rather than automatically stitched together; geometric stitching remains an explicit programmatic option for genuine overlaps.

Missing facial/body/object parts remain estimates and may be incorrect. Keep the metadata when sharing outputs. Only process photographs you are authorized to use.

## Testing

```bash
python -m pip install -e ".[dev]"
ruff check .
python -m compileall -q src tests
QT_QPA_PLATFORM=offscreen pytest
```

Tests cover mask handling, source preservation, invalid output rejection, reference loader configuration, retry limits, file separation, manifests and the simplified UI. Model calls in regression tests are mocked: these tests do **not** demonstrate visual fidelity on real photographs. Assess that separately on representative, authorized source fragments with the actual model and hardware.

Technical references: [IP-Adapter model card](https://huggingface.co/h94/IP-Adapter), [Diffusers IP-Adapter guide](https://huggingface.co/docs/diffusers/en/using-diffusers/ip_adapter).
