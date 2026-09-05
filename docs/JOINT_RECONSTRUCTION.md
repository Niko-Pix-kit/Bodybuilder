# Joint reconstruction (0.3)

## Two different outputs

**Complete subject - new full view** is the desktop default. Original fragments supply appearance evidence, not output framing. A blank canvas is used, with full-length person framing or a complete-object composition. Optional extra views use the same original evidence, not crops of the first result.

**Restore each original photo using all fragments** preserves the observed working pixels of each source and repairs missing areas. Genuine overlapping source pixels are considered before generation. This is distinct from synthesizing a new complete view.

## Evidence processing

A local native Florence-2 model (`florence-community/Florence-2-base`) detects candidate common object categories and regions. Shared category support and box area propose the subject; ambiguous choices stop and ask for the optional **Subject hint**. The model does not recognize a person's identity or prove that two similar objects are the same instance. Keep one physical subject per folder.

Generic region captions supply object parts. Person mode additionally queries eyes, mouth, hair, nose, hands and feet through the same region interface. Visible, nonempty, sufficiently detailed crops are ranked per part across all original sources. A white erasure or a masked crop is not selected as reliable part evidence. The exact selected source and coordinates appear in `subject_evidence.json` and per-output diagnostics.

SIFT/RANSAC registration requires distributed inliers and photometric overlap agreement. Verified overlapping crops can form non-generative union-canvas references; missing pixels in an original can be filled from a matching donor. Different poses are not forcibly pasted together. Registration is limited to the first eight inspected sources for bounded cost; at most two assemblies enter a generation pass.

Whole-view generation suppresses the IP-Adapter layout-copying blocks and uses appearance conditioning. Up to four selected details are then localized in the draft and refined with spatial IP-Adapter masks, from broad regions to small regions. Each refinement uses the original detail crop, never generated content as source evidence. This is regional conditioning, not an exact pixel copy across different poses.

A box-margin/multiple-instance check rejects obviously clipped whole-view drafts, with one framing retry. When uncovered eyes are selected and the source descriptions do not indicate eyewear, a negative constraint and output-description check discourage added glasses. A detected contradiction retries the view once. Detector mistakes remain possible; neither this check nor the framing check proves semantic correctness, exact anatomy, or full 3D consistency.

## Budgets and fidelity

All source files are inspected; at most 16 references enter each diffusion pass, reserving slots for complementary parts. The exact subset is recorded. Up to four regional refinements are performed per output. Detection runs on CPU, alongside the existing SDXL generation backend, so processing is slower and requires additional RAM. The first run downloads the detector's weights. Photos remain local; remote Python code is not enabled.

Use **Mark missing area** for opaque erasures where appropriate. Transparency and explicit masks are authoritative; arbitrary white clothing or black objects must not be erased automatically. Do not place generated reconstructions back in the source folder.

This pipeline estimates a coherent complete subject from partial evidence. It does not reconstruct a persistent 3D mesh, solve arbitrary disconnected fragments uniquely, recover unobserved facts, or guarantee identical geometry across generated views. Unlocalized or unsupported details are reported, not certified as recovered.

## Update

Use the existing environment's Python interpreter from the repository root:

```bash
git pull --ff-only origin main
python -m pip install -r requirements.txt
python -m bodybuilder
```

The dependency update is required for native Florence-2 support. No new virtual environment is required.

## Validation

Tests include actual feature registration, generic object selection, complementary-region provenance, fresh whole-subject canvases, pixel preservation, UI routing and real Diffusers mask processing. CI also downloads the small detector and runs its detection, localization and region-description tasks on a synthetic fixture. Diffusion image quality and exact fidelity on a person's private photos are not validated by these tests.

Primary technical references: [Florence-2 in Transformers](https://huggingface.co/docs/transformers/en/model_doc/florence2), [spatial masks and style/layout control in Diffusers](https://huggingface.co/docs/diffusers/en/using-diffusers/ip_adapter).
