# Restored sources, synthetic views and quality presets

## Desktop defaults

The desktop application now selects **Restore source photos + create complete views**,
**5 synthetic views**, **Balanced**, and **2x output**. For three usable source photos,
this schedules three restored photos followed by five new complete views: eight
outputs, not nine. The source originals are never overwritten.

**Synthetic views** is the total number of new views in all new workflow modes.
It can be changed to 10 (up to 16). **Restore source photos only** generates no new
views; **Create complete views only** restores no source photos and requires at
least one new view. In combined mode, zero new views is allowed. Source restoration
uses the existing photographed perspective. Synthetic views use an independent
composition of the whole subject.

Sources are analyzed once. Both output types share the same original evidence,
part selection and loaded generation backend. Restored images and intermediate
generated drafts never become original reference evidence for later outputs.
Original opaque erasures still require a missing-area mask; transparency is
recognized automatically.

## Real compute budgets

| Profile | Working long edge | Main generation steps | Maximum regional passes | Steps per regional pass |
| --- | ---: | ---: | ---: | ---: |
| Fast | 768 px | 24 | 2 | 16 |
| Balanced | 1024 px | 40 | 4 | 28 |
| High quality | 1024 px | 60 | 8 | 44 |

These values reach the actual generation requests. Regional passes only run for
usable localized details inside the allowed reconstruction mask. A regional
strength of 0.75 is unchanged, so the effective denoising steps can be fewer than
the configured steps shown above. All presets inspect the source set and keep the
same 16-reference budget, observed-pixel protection, numerical checks, bounded
framing recovery, and cancellation handling. Fast does not bypass validation or
replace generation with a non-generative preview.

High quality spends more computation on generation and on separately conditioned
parts. It does not simply request more final images, raise text guidance, or sleep
longer. It retains a 1024-pixel working edge to avoid additional GPU memory pressure.
More work provides additional opportunities for refinement, not proof of a better
likeness. Semantic localization and generated detail can still be wrong; a
monotonic visual-quality improvement has not been established by these tests.

The active values are visible in the quality selector's tooltip, processing log,
run manifest and each output's metadata. No inference-time estimates are claimed;
actual duration depends on hardware, number of sources, available parts, models
already cached, enhancement, and any bounded retries.

## 2x output and fidelity are different

2x output is enabled by default in every preset. The resource override remains
under **Advanced options**, not in the main workflow. The existing Swin2SR enhancer
is used, with an explicitly logged deterministic Lanczos fallback if enhancement
is unavailable. A missing enhancer does not silently become a face-restoration model.

After enhancement, observed regions in source-restoration outputs are restored
using deterministic Lanczos resampling of the working canvas. They are not
repainted by AI. All other observed-pixel protections also remain active with 2x
disabled. Fast's usual output long edge is 1536 pixels; Balanced/High quality use
2048 pixels. The working canvas is already a resampling of the input. Doubling
output dimensions does not recover lost pixels or establish identity accuracy.
Synthetic outputs remain entirely generated, including any plausible unseen parts.

## Progress and outputs

The source photos are restored first, then the requested synthetic views are
created. The UI shows both planned counts before running and saved counts at
completion. The manifest separately records planned and saved counts, so a later
cancellation or model error cannot be presented as a complete batch.

Results remain in `images/`, named `source__...png` for restorations and
`subject__whole_...png` for synthetic views. The result list labels the two kinds.
Diagnostics stay separate. Uncertain framing continues to use `__needs_review`
files and warnings; this change does not turn the framing warning back into a
fatal error. Earlier completed files survive a later operational failure.

## Programmatic compatibility

`desktop_config(input_dir, output_dir)` supplies the new defaults. Pass
`workflow=WorkflowMode.BOTH` and `quality=QualityMode.BALANCED` to
`JointReconstructionPipeline`. Existing low-level `PipelineConfig` defaults and
calls using `complete_subject` keep their old behavior for compatibility; the
application always passes the new explicit workflow and quality preset.

## Validation scope

Tests verify exact batch counts, original-only reference provenance, propagation
of all three budgets into generation/detail calls, 2x observed-pixel preservation,
UI defaults, mode switching, worker configuration, and retaining partial results.
Model calls in these tests are controlled doubles, not real visual-quality
benchmarks. Existing real-library compatibility tests remain enabled.

Reference: [Diffusers SDXL parameters](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/stable_diffusion_xl)
explains the inference-step and speed tradeoff. It does not guarantee identity
fidelity on a particular set of fragments.
