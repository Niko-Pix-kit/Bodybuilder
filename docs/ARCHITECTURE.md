# Architecture

`ui/main_window.py` implements the folder-to-folder desktop workflow. It exposes no classical reconstruction engine. Advanced options are closed initially. `ui/mask_editor.py` writes explicit missing-area sidecars without overwriting originals. Qt work runs on a QThread; cancellation uses a thread-safe event and thread shutdown is allowed to finish cooperatively.

`core/image_io.py` separates display decoding from evidence-aware fragment decoding. Alpha and optional missing-area masks define evidence; arbitrary black or white RGB regions are not automatically removed. Run outputs and masks are excluded from source discovery.

`core/canvas.py` creates working canvases with usable SDXL dimensions. `core/pipeline.py` sends individual references to the model, restores observed working pixels, separates final images from diagnostics, and persists failure/cancellation states. It records exact source hashes, reference paths, effective generation seeds and environment versions.

`ai/sdxl.py` explicitly loads the ViT-H image encoder, requires a functioning IP-Adapter, installs CPU-offload hooks only after loading adapter components, and uses VAE upcasting. Invalid numeric/empty/mask-like output is detected before export. A single full-precision retry is bounded; exhausted retries are fatal. Technical checks do not measure identity fidelity.

`ai/base.py` retains classical filling for developer diagnostics only. It is not used as a fallback. `ai/upscale.py` is optional; observed areas are restored after enhancement. All broad exception handlers are application boundaries that record and propagate/display faults, not silent success paths.

The regression suite mocks model calls and does not download weights. UI tests require PyQt6 and use the offscreen Qt platform. Real-model visual validation requires a separate representative dataset and appropriate hardware.
