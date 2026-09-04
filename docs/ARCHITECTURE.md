# Architecture

BodyBuilder is split into a thin PyQt6 presentation layer and testable reconstruction services.

## Layers

- `bodybuilder.ui`: folder selection, settings, analysis table, previews, progress, cancellation, and detailed errors. Work runs on a `QThread`; the UI thread never performs model inference.
- `bodybuilder.core.image_io`: recursive discovery, EXIF-safe decoding, HEIF registration, atomic PNG/JSON writes, and collision-safe run directories.
- `bodybuilder.core.analysis`: deterministic quality metrics and local face-region detection. It does not perform identity matching.
- `bodybuilder.core.stitching`: SIFT/AKAZE feature matching, RANSAC homography validation, canvas expansion, and non-destructive insertion of previously unobserved source pixels.
- `bodybuilder.core.canvas`: AI canvas sizing, observed/generated masks, and exact restoration of the observed region after generation.
- `bodybuilder.core.clustering`: optional DINOv2 grouping for folders containing unrelated subjects, with a non-neural fallback.
- `bodybuilder.ai.sdxl`: lazy local SDXL inpainting and IP-Adapter conditioning.
- `bodybuilder.ai.upscale`: optional Swin2SR enhancement and deterministic Lanczos fallback.
- `bodybuilder.core.pipeline`: orchestration, manifests, provenance outputs, per-item fault isolation, and model lifetime management.

## Data invariants

1. White in `observed_mask` means that a pixel is derived from an input photograph at the current working resolution.
2. White in `generated_mask` means that a pixel may be synthetic. The masks are complements for normal source completions.
3. Diffusion output is never trusted inside `observed_mask`; the source canvas is pasted back after every generation.
4. Neural upscaling may modify the entire frame, so the observed region is pasted back again from a deterministic Lanczos resize.
5. Fully synthetic variants contain no observed pixels and are explicitly marked as such.
6. Every final image has sidecar masks, a comparison sheet, metadata, and a run-level manifest.

## Failure model

A corrupt source file, a rejected overlap candidate, or one failed output is recorded and the run continues by default. Model initialization failures stop the run because subsequent outputs would fail identically. Writes use a temporary file followed by `os.replace`, so interruption cannot leave a half-written final artifact.
