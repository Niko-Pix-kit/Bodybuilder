# Framing recovery and review-required images

A subject detector returns approximate bounding boxes. A box touching an image edge is a useful warning, but does not prove that the person or object is cropped. It must not discard an otherwise valid reconstruction or stop all subsequent views.

## Bounded correction

For each whole-view composition the application makes one initial generation. If its box touches an edge, the generated draft is resampled to 70% of its dimensions and centered on a canvas with the same output dimensions. A real SDXL inpainting pass completes the new margins using the original reference photographs. The protected center is restored after that pass. This is not a padding-only workaround. The correction does not enlarge the diffusion canvas or request more memory by raising the output resolution.

When localization fails or several substantial subjects are detected, the single corrective attempt is a fresh wider composition instead. Near-identical bounding-box proposals are deduplicated. If the second candidate has a worse detector assessment, the first candidate is retained. Both drafts, the recovery mask where applicable, generation metadata and check history are saved under `diagnostics/`.

These are at most two layout generations per semantic attempt. Existing numerical retries and the separate unsupported-eyewear retry remain bounded. Regional detail passes still use original source evidence. A further framing check after regional refinement is advisory and does not start another framing loop.

## Explicit uncertainty, not false success

If the final check remains uncertain, the image is saved as `*__needs_review.png` in `images/` and the next view is processed. The result list labels the image **framing needs review**. The completion status says **Completed with warnings** and opens the processing log.

The output metadata contains `review_required: true`, `framing_check: needs_review`, the warning and `framing_history`. The run manifest records `completed_with_warnings` (or `completed_with_errors` when separate errors occurred), `review_count` and the affected output paths in its records. These files are provisional images for inspection, not confirmed complete reconstructions. A passed bounding-box check also does not verify anatomy, identity or unseen geometry.

The generated draft used for the correction is working material only. It is not added to the original references, and the final observed-pixel mask stays empty for a fully synthetic output. Source-repair mode is not recentered: its photographed pixels remain protected at the working resolution.

Cancellation, memory failures, model-loading errors, invalid numerical outputs and unsupported-eyewear errors are not swallowed by this change. Completed files and diagnostic drafts are retained when an actual later error occurs.

## Tests

`test_framing.py` checks real Pillow masks and geometry. `test_framing_pipeline.py` exercises the run, exports and UI with controlled detectors/backends: persistent edge warnings, a successful correction, lost localization, multiple subjects, post-refinement warnings, selecting the better draft, preserving original evidence, cancellation and propagation of operational errors. These tests do not establish visual fidelity on real photographs.

Reference mask convention: [Diffusers inpainting API](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/inpaint). Detector output schema: [Florence-2 documentation](https://huggingface.co/docs/transformers/model_doc/florence2).
