# Fidelity and limitations

## What can be recovered

Pixels visible in an input photograph can be retained, geometrically aligned, and resampled. When two fragments genuinely overlap, feature matching may recover their relative placement. Those operations are evidence-based.

## What cannot be recovered

Content never captured by any source image has no unique correct answer. A model can only generate a plausible hypothesis. A hidden ear, hand, logo, garment section, rear view, or full body inferred from a face crop is not a recovered fact.

BodyBuilder therefore distinguishes:

- **source completion**, where visible pixels are preserved and only the masked region is generated;
- **synthetic variants**, where the complete frame is generated from references and prompts.

## Identity and object consistency

IP-Adapter conditioning, all-fragment reference boards, and face-focused reference boards reduce drift but cannot guarantee identity. Consistency normally improves when the input set contains sharp, well-lit views from several angles. Conflicting ages, clothing, lighting, or subjects produce ambiguous evidence.

Automatic face detection is only a crop-selection aid. It is not face recognition and does not establish that two photographs depict the same person.

## Enhancement

Swin2SR may synthesize plausible high-frequency detail. BodyBuilder applies it to the generated frame but then restores observed areas from deterministic Lanczos resampling. Consequently, real source areas are not neural-enhanced. This is intentionally conservative.

## Geometric stitching

Stitching requires repeated visual features. Separate photographs of a forehead and a shoe have no geometric overlap and cannot be stitched by homography; they are used only as conditioning references. Perspective change, articulation, motion, smooth surfaces, or heavy blur can cause a candidate to be rejected.

## Practical validation

Use multiple seeds, compare results, and reject unstable details. Keep masks and sidecars. Never use a generated completion as forensic, medical, legal, biometric, or documentary evidence.
