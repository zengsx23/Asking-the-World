"""Sparse observed-contact mask evidence for qualitative material response."""
from __future__ import annotations

import cv2
import numpy as np


def _mask(backend, object_id, frame):
    try:
        value = backend._mask(object_id, frame)
    except (KeyError, ValueError, IndexError):
        return None
    value = cv2.resize(value.astype(np.uint8), (backend.width, backend.height),
                       interpolation=cv2.INTER_NEAREST).astype(bool)
    return value if value.any() else None


def _touches(left, right):
    if left is None or right is None:
        return False
    kernel = np.ones((5, 5), np.uint8)
    return bool((cv2.dilate(left.astype(np.uint8), kernel) & right).any())


def _episodes(frames):
    groups = []
    for frame in frames:
        if groups and frame <= groups[-1][-1] + 2:
            groups[-1].append(frame)
        else:
            groups.append([frame])
    return groups


def _first_contact(backend, first, candidates):
    contacts = []
    for frame in range(backend.frame_count):
        left = _mask(backend, first, frame)
        for second in candidates:
            if _touches(left, _mask(backend, second, frame)):
                contacts.append((frame, second))
                break
    if not contacts:
        return None
    partner = contacts[0][1]
    span = next(group for group in _episodes([f for f, p in contacts if p == partner])
                if contacts[0][0] in group)
    return first, partner, span


def _sample_frames(span, frame_count, fps):
    gap = max(2, round(.18 * fps))
    start, end = span[0], span[-1]
    before = [max(0, start - 2 * gap), max(0, start - gap)]
    during = np.linspace(start, end, min(3, max(1, end - start + 1))).round().astype(int).tolist()
    after = [min(frame_count - 1, end + gap), min(frame_count - 1, end + 2 * gap)]
    labeled = [(frame, 'before') for frame in before]
    labeled += [(frame, 'contact') for frame in during]
    labeled += [(frame, 'after') for frame in after]
    return list(dict(labeled).items())


def _sheet(backend, first, second, samples, index):
    capture = cv2.VideoCapture(str(backend.video))
    panels = []
    try:
        masks = [(_mask(backend, first, frame), _mask(backend, second, frame)) for frame, _ in samples]
        union = np.zeros((backend.height, backend.width), bool)
        for left, right in masks:
            if left is not None:
                union |= left
            if right is not None:
                union |= right
        if not union.any():
            raise ValueError("Contact masks are unavailable")
        yy, xx = np.where(union)
        pad = max(12, round(max(xx.max()-xx.min()+1, yy.max()-yy.min()+1) * .25))
        x0, x1 = max(0, xx.min()-pad), min(backend.width, xx.max()+pad+1)
        y0, y1 = max(0, yy.min()-pad), min(backend.height, yy.max()+pad+1)
        for (frame, stage), (left, right) in zip(samples, masks):
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame)
            ok, image = capture.read()
            if not ok:
                continue
            for mask, color in ((left, (0, 80, 255)), (right, (255, 180, 0))):
                if mask is None:
                    continue
                contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(image, contours, -1, color, 2)
            crop = image[y0:y1, x0:x1]
            scale = 260 / max(crop.shape[:2])
            crop = cv2.resize(crop, (max(1, round(crop.shape[1]*scale)),
                                     max(1, round(crop.shape[0]*scale))))
            panel = np.full((310, 280, 3), 24, np.uint8)
            header = {'before': (100, 70, 25), 'contact': (35, 35, 150), 'after': (35, 110, 45)}[stage]
            panel[:38] = header
            oy, ox = 40 + (260-crop.shape[0])//2, 10 + (260-crop.shape[1])//2
            panel[oy:oy+crop.shape[0], ox:ox+crop.shape[1]] = crop
            cv2.putText(panel, f"{stage.upper()}  f{frame}  {frame/backend.fps:.2f}s", (8, 26),
                        cv2.FONT_HERSHEY_SIMPLEX, .58, (255, 255, 255), 2)
            panels.append(panel)
    finally:
        capture.release()
    if not panels:
        raise ValueError("Contact frames could not be decoded")
    sheet = np.hstack(panels)
    path = backend.output / "material_response" / f"episode_{index}_{first}_{second}.jpg"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 90]):
        raise OSError(f"Cannot write {path}")
    return str(path)


def inspect_material_response(backend, object_ids, property_name):
    ids = backend._ids(object_ids)
    if len(ids) != 2 or len(set(ids)) != 2:
        raise ValueError("object_ids must contain exactly two distinct IDs")
    allowed = {"bendability"} if backend.engine == "cloth" else {"elasticity", "plasticity"}
    if property_name not in allowed:
        raise ValueError(f"{backend.engine} material response supports: {sorted(allowed)}")
    physical = {name: row.get("physical_type") for name, row in backend.objects.items()}
    expected = "soft_body" if backend.engine == "soft_body" else "cloth"
    if any(physical.get(name) != expected for name in ids):
        raise ValueError(f"Both targets must be modeled {expected} objects")
    episodes = []
    if backend.engine == "soft_body":
        direct = _first_contact(backend, ids[0], [ids[1]])
        if direct is not None:
            episodes.append(direct)
    mode = "direct_target_contact" if episodes else "first_rigid_contact_per_target"
    if not episodes:
        rigid = [name for name, kind in physical.items()
                 if name not in ids and kind not in {"cloth", "soft_body", "fluid", "rope"}]
        for target in ids:
            contact = _first_contact(backend, target, rigid)
            if contact is not None:
                episodes.append(contact)
            if len(episodes) == 2:
                break
    records, images = [], []
    for index, (first, second, span) in enumerate(episodes[:2], 1):
        samples = _sample_frames(span, backend.frame_count, backend.fps)
        images.append(_sheet(backend, first, second, samples, index))
        records.append(dict(object_ids=[first, second], contact_span_frames=[span[0], span[-1]],
                            sampled_frames=[dict(frame=frame, stage=stage) for frame, stage in samples],
                            stages="spaced before/contact/after panels; contact duration may limit distinct contact samples"))
    convention = backend._contphy.evidence_catalog()["material_response_convention"]
    if not records:
        return dict(source_kind="observed_interaction_masks", property_name=property_name,
                    object_ids=ids, status="insufficient_contact_evidence", images=[],
                    reason="No target-target or target-rigid contact was found in available masks",
                    material_response_convention=convention)
    return dict(source_kind="observed_interaction_masks", property_name=property_name,
                object_ids=ids, status="observed", interaction_mode=mode, episodes=records,
                images=images, material_response_convention=convention,
                interpretation=(
                    "For cloth, compare visible-area change across interaction frames: smaller change supports "
                    "harder to bend; a larger post-interaction area decrease supports easier wrinkling."
                    if backend.engine == "cloth" else
                    "Compare deformation magnitude during contact and recovery after unloading; missing recovery coverage is inconclusive."))
