# Fidelity and limitations

Observed pixels are preserved at the working canvas resolution, not necessarily at native source resolution. Missing regions and new poses are generated hypotheses. The program has no identity database and does not verify the person's identity.

Do not expect a complete face detector to recognize every fragment. Person mode is the default even if no complete face is detected. Source conditioning is mandatory and each selected image is encoded individually without a destructive center crop. A maximum of 16 references is selected per source, with the source itself always included; exact reference paths are recorded.

Geometric overlap stitching is off by default because different poses are not one planar photograph. Explicit stitching remains available through the Python configuration. No multi-view 3D reconstruction or physical anatomy constraints are implemented. Text-guided synthetic poses are approximate.

Transparent pixels and explicit `photo.ext.mask.png` sidecars define missing content. Opaque occlusion cannot be distinguished reliably from real content by color alone; mark the region manually. Do not mark genuine evidence merely to obtain a more attractive result.

A successful technical check only establishes that the model returned a non-degenerate image. It does not establish that generated facial features, anatomy or object details are correct. Uniform real backgrounds may be rejected by the conservative invalid-output check. Runtime errors, missing models and memory shortages are displayed and logged, not converted to successful blank output.

Optional 2x enhancement keeps observed details deterministic through Lanczos restoration and may alter/generated details elsewhere. It is not a way to recover unknown high-frequency facial information. Source files are never overwritten.
