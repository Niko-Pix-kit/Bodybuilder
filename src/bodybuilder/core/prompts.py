"""Conservative prompts for completion and synthetic variants."""

from __future__ import annotations

from bodybuilder.config import SubjectKind, VariantFrame

_COMMON_NEGATIVE = (
    "different identity, duplicate subject, multiple heads, extra face, extra limbs, missing limbs, "
    "deformed hands, fused fingers, asymmetrical eyes, changed clothing, changed object design, "
    "text, watermark, logo, low resolution, oversharpening, plastic skin, illustration, CGI"
)


def completion_prompt(subject_kind: SubjectKind, evidence: str = "") -> tuple[str, str]:
    if subject_kind == SubjectKind.PERSON:
        prompt = (
            "complete the same photographed person outside the existing crop, preserve facial identity, "
            "age, body proportions, skin tone, hair, clothing, accessories, camera perspective, lighting, "
            "background, and photographic grain; anatomically plausible natural continuation; realistic photo"
        )
    else:
        prompt = (
            "complete the same photographed object or scene outside the existing crop, preserve geometry, "
            "materials, colors, markings, camera perspective, lighting, background, and photographic grain; "
            "physically plausible continuation; realistic photo"
        )
    if evidence.strip():
        prompt += f"; verified source evidence: {evidence.strip()}"
    return prompt, _COMMON_NEGATIVE


def variant_prompt(
    subject_kind: SubjectKind,
    index: int,
    frame: VariantFrame,
    evidence: str = "",
) -> tuple[str, str, str]:
    person_views = (
        ("standing_front", "standing naturally, front view, full subject visible"),
        ("standing_three_quarter", "standing naturally, three-quarter view, full subject visible"),
        ("standing_side", "standing naturally, side view, full subject visible"),
        ("seated_natural", "seated in a natural relaxed pose, full subject visible"),
        ("walking_natural", "walking naturally, full subject visible"),
        ("portrait_front", "front-facing neutral portrait"),
        ("portrait_three_quarter", "three-quarter neutral portrait"),
        ("arms_relaxed", "standing with arms relaxed, full subject visible"),
    )
    object_views = (
        ("front_view", "front view, entire object visible"),
        ("three_quarter_view", "three-quarter view, entire object visible"),
        ("side_view", "side view, entire object visible"),
        ("rear_view", "rear view, entire object visible"),
        ("top_view", "elevated top view, entire object visible"),
        ("context_view", "natural contextual view, entire object visible"),
    )
    views = person_views if subject_kind == SubjectKind.PERSON else object_views
    slug, view = views[index % len(views)]
    frame_phrase = {
        VariantFrame.FULL_BODY: "vertical 2:3 composition",
        VariantFrame.PORTRAIT: "portrait 4:5 composition",
        VariantFrame.SQUARE: "square composition",
        VariantFrame.LANDSCAPE: "landscape 3:2 composition",
    }[frame]
    subject_phrase = (
        "the same person, preserve identity, age, proportions, hair, skin, clothing, and accessories"
        if subject_kind == SubjectKind.PERSON
        else "the same object or scene, preserve geometry, materials, colors, markings, and design"
    )
    prompt = f"realistic photograph of {subject_phrase}; {view}; {frame_phrase}; coherent lighting and detail"
    if evidence.strip():
        prompt += f"; verified source evidence: {evidence.strip()}"
    return slug, prompt, _COMMON_NEGATIVE
