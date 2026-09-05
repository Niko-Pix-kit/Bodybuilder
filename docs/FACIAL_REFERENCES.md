# Complementary facial references

BodyBuilder can route eyes, mouth and hair references from different original photographs to corresponding missing regions. This is regional generative guidance, not face recognition, identity verification or the literal recovery of hidden pixels.

## Working with complementary crops

Keep only originals of the same person in the source folder. Select a source, then **Facial references in selected photo**. Draw around **Visible eyes and eyebrows (no eyewear)**, **Visible mouth and surrounding skin**, or **Visible hair**, click **Use selection**, then **Save**. Repeat for another source when a different part is visible there. You do not have to label every photograph.

Eye-pair detection supplies editable proposals when a plausible full face and two eyes are detected. It can miss cropped or tilted faces and can make false detections. Mouth and hair boxes are user-labelled rather than guessed from an incomplete face. Saving the editor confirms the displayed boxes; deleting a proposal and saving prevents it from being automatically reintroduced for that source.

For an opaque white/blurred patch covering the mouth, first use **Mark missing area in selected photo**. Without this step those opaque source pixels remain locked. Facial-reference boxes identify information to use; they do not declare information missing. A white wall, black hair or white clothing is never automatically erased.

For a badly cropped target, choose **Face location, including missing parts** in the facial-reference editor. Draw the intended whole face on the extended canvas. This is a geometry hint, not an additional source photograph. It may extend outside the original crop. Visible mouth/eye boxes can provide approximate layout automatically, but manually correcting the whole-face location is more reliable for strong perspective, partial eyes or extreme crops.

Guides are saved as `photo.jpg.evidence.json` beside the original. The original image is not changed. A SHA-256 and oriented dimensions tie a guide to the source it describes. A source changed after annotation invalidates its saved guide. Update the guide before reconstructing that source.

## What the engine does

1. Inspect all source paths in the subject group for usable regional evidence, independently of the 16-photo cap used for general context. For each anatomical part, select the sharpest usable region, prioritising an explicit user selection over a detector proposal. Marked/transparent pixels and featureless white/black regions cannot become evidence patches.
2. Generate a general completion, including the selected detail crops in its reference set. When uncovered-eye evidence exists, add unobstructed-eye guidance and negative eyewear prompts. This does **not** guarantee that a generative model will obey the instruction.
3. Locate the target face from a source guide, source features, or a detected eye pair in the draft. A draft may supply layout only: its pixels never become new reference evidence. If the face cannot be located, retain the general completion for review and explicitly report that regional guidance was not applied.
4. Run a second inpainting pass. Each selected detail has its own output-space IP-Adapter attention mask. Eye information is routed to the eyes, mouth information to the mouth, and hair information around the face. Only the intersection with the original missing-pixel mask can change. Known source pixels and non-targeted draft areas are restored after this pass.

The second pass adds processing time. Device selection remains automatic; there are no new generation sliders. Numerical retries remain bounded and no longer reduce reference strength silently.

## Inspecting results

`generation.facial_guidance` in each output's metadata records the selected source paths, crop rectangles, source hashes, automatic/manual origin, target face frame, refinement status and seeds. `identity_verified` is always false: technical checks are not identity validation. A completion without enough regional evidence is explicitly marked, including in the desktop completion status.

Previous run folders and exported PNGs carrying BodyBuilder metadata are excluded from source discovery, even when such a PNG has been copied elsewhere. Screenshots and files re-encoded without metadata cannot be reliably identified as generated. Never add an earlier synthetic result to the original reference folder.

## Limits and validation

The masks constrain where reference features are injected and the final compositing hard-locks observed pixels. They do not force exact eye colour, anatomy, likeness, expression or accessory absence. Neural attention affects the surrounding computation, and a 2D crop from a different perspective cannot be pasted faithfully without pose/geometry estimation. The current placement templates are approximate, especially for profile views. Review results rather than treating them as documentary originals.

Tests cover multi-source regional selection, missing-region exclusion, guide invalidation, source pixel preservation, copied-generated-image exclusion, mask tensor shape, the visual guide editor, and real Diffusers IP-Adapter spatial routing with tiny random tensors. No private photographs are committed. These tests do not constitute a pretrained-model visual benchmark or a guarantee of likeness.

Technical basis: [Diffusers regional IP-Adapter masks](https://huggingface.co/docs/diffusers/en/using-diffusers/ip_adapter#masking) and [IP-Adapter model card](https://huggingface.co/h94/IP-Adapter). This implementation uses the existing IP-Adapter Plus weights and does not add a recognition service or biometric database.
