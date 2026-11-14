"""Read-only MovingTarget adapter for bounded S1 development clips.

Original frame IDs and point coordinates are retained. Explicit source origin
0 (default) becomes MATLAB 1-based via +1. The local source document
MovingTarget/视频集说明.doc, annotation-format item (7), specifies upper-left
(0,0), right/down positive. No origin is selected using tracking performance.
The conservative manifest metadata is retained for compatibility with already
frozen clips; source-document verification is recorded separately.
"""

from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import math
from pathlib import Path
import re

from PIL import Image, ImageChops


def _sha256(path):
    with Path(path).open("rb") as stream:
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _issue(kind, line=None, frame_id=None, raw=None, **details):
    return dict(type=kind, line=line, frame_id=frame_id, raw=raw, **details)


def _invalidate(row, reason):
    row["label_valid"] = False
    row["exclusion_reason"] = reason


def _coordinate(token):
    value = float(token)
    if not math.isfinite(value):
        raise ValueError("coordinate must be finite")
    return int(value) if value.is_integer() else value


def parse_labels(path):
    """Return header, int-keyed rows, source SHA256 and explicit parse issues.

    Each row contains frame_id, target_count (None when unavailable), objects
    with original object_id/x/y, label_valid, exclusion_reason, line/raw/issues.
    Duplicate rows preserve every occurrence in source_rows and are invalid.
    Positive incomplete labels never become target_count=0. Count-zero residual
    coordinates produce a warning and no objects, as in the source scorer.
    """
    path = Path(path)
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    header = dict(raw=lines[0] if lines else "", sequence=None,
                  frame_count=None, track_count=None)
    issues = []
    fields = header["raw"].split()
    try:
        header.update(sequence=fields[1], frame_count=int(fields[2]),
                      track_count=int(fields[3]))
        if header["frame_count"] < 0 or header["track_count"] < 0:
            raise ValueError("negative header counts")
    except (IndexError, ValueError):
        issues.append(_issue("invalid_header", 1, raw=header["raw"]))

    rows = {}
    for line_no, raw in enumerate(lines[1:], start=2):
        if not raw.strip():
            continue
        tokens = raw.split()
        frame_match = re.fullmatch(r"frame:(\d+)", tokens[0])
        if not frame_match:
            issues.append(_issue("invalid_frame_id", line_no, raw=raw))
            continue
        frame_id = int(frame_match.group(1))
        row = dict(frame_id=frame_id, target_count=None, objects=[],
                   label_valid=True, exclusion_reason=None, line=line_no,
                   raw=raw, issues=[])

        def add_issue(kind, invalid=True):
            issue = _issue(kind, line_no, frame_id, raw)
            row["issues"].append(issue)
            issues.append(issue)
            if invalid:
                _invalidate(row, kind)

        try:
            count = int(tokens[1])
            row["target_count"] = count
            if count < 0:
                raise ValueError("negative count")
        except (IndexError, ValueError):
            add_issue("invalid_target_count")
        else:
            if count == 0:
                row["exclusion_reason"] = "target_absent"
                if len(tokens) > 2:
                    add_issue("zero_count_residual", invalid=False)
            else:
                if len(tokens) != 2 + 3 * count:
                    add_issue("object_count_mismatch")
                for offset in range(2, len(tokens), 3):
                    try:
                        object_match = re.fullmatch(r"object:([1-9]\d*)", tokens[offset])
                        if not object_match:
                            raise ValueError("invalid object ID")
                        point = dict(object_id=int(object_match.group(1)),
                                     x=_coordinate(tokens[offset + 1]),
                                     y=_coordinate(tokens[offset + 2]))
                    except (IndexError, ValueError):
                        add_issue("invalid_object")
                        continue
                    row["objects"].append(point)
                ids = [point["object_id"] for point in row["objects"]]
                if len(set(ids)) != len(ids):
                    add_issue("duplicate_object_id")
                if count > 1 and row["label_valid"]:
                    row["exclusion_reason"] = "multiple_targets"
        if frame_id in rows:
            old = rows[frame_id]
            if "source_rows" not in old:
                old["source_rows"] = [deepcopy(old)]
            old["source_rows"].append(row)
            issue = _issue("duplicate_frame", line_no, frame_id, raw)
            issues.append(issue)
            old["issues"].append(issue)
            _invalidate(old, "duplicate_frame")
        else:
            rows[frame_id] = row
    return dict(path=path.resolve().as_posix(), sha256=_sha256(path),
                header=header, rows=rows, issues=issues)


def _images(folder):
    images = {}
    issues = []
    for path in folder.iterdir():
        if path.suffix.lower() != ".bmp" or not path.is_file():
            continue
        if not path.stem.isdigit():
            issues.append(_issue("nonnumeric_image_name", path=path.as_posix()))
            continue
        frame_id = int(path.stem)
        if frame_id in images:
            issues.append(_issue("duplicate_image_id", frame_id=frame_id,
                                 paths=[images[frame_id].as_posix(), path.as_posix()]))
        else:
            images[frame_id] = path
    return dict(sorted(images.items())), issues


def _bounded_row(row, frame_id, width, height, source_origin):
    if row is None:
        return dict(frame_id=frame_id, target_count=None, objects=[], label_valid=False,
                    exclusion_reason="missing_label", issues=[])
    row = deepcopy(row)
    if row["label_valid"] and any(
        not (source_origin <= p["x"] < width + source_origin and
             source_origin <= p["y"] < height + source_origin)
        for p in row["objects"]
    ):
        _invalidate(row, "out_of_bounds")
        row["issues"].append(_issue("out_of_bounds", row["line"], frame_id,
                                     row["raw"], width=width, height=height))
    return row


def inventory(root):
    """Summarize all dataN sequences using BMP headers, without pixel decoding.

    Bounds and single-target ranges use the disclosed source-origin assumption 0.
    Ranges split at gaps, absent/invalid/multiple targets, or target-ID changes.
    Header naming/count disagreements are reported, never used to rename frames.
    """
    root = Path(root).resolve()
    sequences = []
    folders = sorted((p for p in root.iterdir() if p.is_dir() and
                      re.fullmatch(r"data\d+", p.name)),
                     key=lambda p: int(p.name[4:]))
    for folder in folders:
        images, issues = _images(folder)
        label_path = root / "data_label" / (folder.name + ".txt")
        if label_path.is_file():
            parsed = parse_labels(label_path)
        else:
            parsed = dict(header={}, rows={}, issues=[_issue("missing_label_file")],
                          sha256=None)
        issues += parsed["issues"]
        rows = parsed["rows"]
        header = parsed["header"]
        if header.get("sequence") and header["sequence"] != folder.name:
            issues.append(_issue("header_sequence_mismatch", expected=folder.name,
                                 actual=header["sequence"]))
        if header.get("frame_count") != len(images):
            issues.append(_issue("header_frame_count_mismatch", expected=len(images),
                                 actual=header.get("frame_count")))
        dimensions, modes, count_histogram = Counter(), Counter(), Counter()
        invalid = {i for i, row in rows.items() if not row["label_valid"]}
        tracks = defaultdict(list)
        ranges = []
        active = None
        image_errors = []
        zero_count = multi_count = valid_count = 0
        for frame_id, image_path in images.items():
            try:
                with Image.open(image_path) as img:
                    width, height = img.size
                    dimensions[f"{width}x{height}"] += 1
                    modes[img.mode] += 1
            except (OSError, ValueError) as exc:
                image_errors.append(dict(frame_id=frame_id, error=str(exc)))
                invalid.add(frame_id)
                active = None
                continue
            row = _bounded_row(rows.get(frame_id), frame_id, width, height, 0)
            count_histogram[str(row["target_count"])] += 1
            if not row["label_valid"]:
                invalid.add(frame_id)
                if row["exclusion_reason"] == "out_of_bounds":
                    issues += row["issues"][-1:]
                active = None
                continue
            valid_count += 1
            zero_count += int(row["target_count"] == 0)
            multi_count += int(row["target_count"] > 1)
            for point in row["objects"]:
                tracks[point["object_id"]].append(frame_id)
            if row["target_count"] != 1:
                active = None
                continue
            object_id = row["objects"][0]["object_id"]
            if active and active["last_frame"] + 1 == frame_id and active["object_id"] == object_id:
                active["last_frame"] = frame_id
                active["frame_count"] += 1
            else:
                active = dict(first_frame=frame_id, last_frame=frame_id,
                              frame_count=1, object_id=object_id)
                ranges.append(active)
        image_ids = set(images)
        label_ids = set(rows)
        sequences.append(dict(
            sequence=folder.name, image_count=len(images),
            first_frame=min(images) if images else None,
            last_frame=max(images) if images else None,
            image_gaps=sorted(set(range(min(images), max(images) + 1)) - image_ids) if images else [],
            dimensions=dict(dimensions), modes=dict(modes), label_count=len(rows),
            label_path=label_path.as_posix(), label_sha256=parsed["sha256"], header=header,
            target_count_histogram=dict(count_histogram), valid_label_frames=valid_count,
            invalid_label_frames=sorted(invalid), zero_target_frames=zero_count,
            multi_target_frames=multi_count, images_without_label=sorted(image_ids - label_ids),
            labels_without_image=sorted(label_ids - image_ids),
            single_target_ranges=ranges,
            tracks={str(k): dict(first_frame=min(v), last_frame=max(v), frame_count=len(v),
                                 internal_missing=max(v) - min(v) + 1 - len(v))
                    for k, v in sorted(tracks.items())},
            issues=issues, image_errors=image_errors))
    return dict(dataset_root=root.as_posix(), sequence_count=len(sequences),
                image_count=sum(s["image_count"] for s in sequences),
                source_origin_assumption=0, sequences=sequences)


def build_clip(root, sequence, first_frame, last_frame, source_origin=0):
    """Build a hashed manifest; reject any discontinuity or invalid selected label.

    frames contain original objects and MATLAB 1-based x/y, plus image dimensions,
    mode and grayscale channel verification. Later-frame coordinates are for
    offline evaluation only; the runner must receive only the initial coordinate.
    """
    if source_origin not in (0, 1):
        raise ValueError("source_origin must be explicitly 0 or 1")
    if not re.fullmatch(r"data\d+", sequence):
        raise ValueError("sequence must be a dataN directory name")
    if not isinstance(first_frame, int) or not isinstance(last_frame, int) or not 0 <= first_frame <= last_frame:
        raise ValueError("clip requires ordered nonnegative integer frame IDs")
    root = Path(root).resolve()
    images, image_issues = _images(root / sequence)
    for issue in image_issues:
        if issue["type"] == "duplicate_image_id" and first_frame <= issue["frame_id"] <= last_frame:
            raise ValueError(f"duplicate image ID {issue['frame_id']}")
    parsed = parse_labels(root / "data_label" / (sequence + ".txt"))
    frames = []
    target_id = None
    for frame_id in range(first_frame, last_frame + 1):
        if frame_id not in images:
            raise ValueError(f"missing image for original frame {frame_id}")
        path = images[frame_id]
        with Image.open(path) as img:
            img.load()  # Selected clips, unlike inventory, verify full image decode.
            width, height = img.size
            mode = img.mode
            if mode == "RGB":
                red, green, blue = img.split()
                grayscale = (ImageChops.difference(red, green).getbbox() is None and
                             ImageChops.difference(red, blue).getbbox() is None)
            else:
                grayscale = mode in ("L", "I", "I;16", "F", "1")
        row = _bounded_row(parsed["rows"].get(frame_id), frame_id,
                           width, height, source_origin)
        if not row["label_valid"] or row["target_count"] != 1:
            raise ValueError(f"frame {frame_id}: {row['exclusion_reason']}")
        point = row["objects"][0]
        if target_id is not None and point["object_id"] != target_id:
            raise ValueError(f"target ID switch at frame {frame_id}")
        target_id = point["object_id"]
        frames.append(dict(frame_id=frame_id, image_path=path.as_posix(),
                           image_sha256=_sha256(path), target_count=row["target_count"],
                           label_valid=True, exclusion_reason=None, objects=row["objects"],
                           x=point["x"] + 1 - source_origin, y=point["y"] + 1 - source_origin,
                           width=width, height=height, mode=mode,
                           grayscale_equal_channels=grayscale))
    return dict(dataset_root=root.as_posix(), sequence=sequence,
                first_frame=first_frame, last_frame=last_frame, frame_count=len(frames),
                source_origin=source_origin,
                coordinate_rule="x=column, y=row; MATLAB pixel centers: source + 1 - source_origin",
                coordinate_origin_status="explicit assumption; not performance-selected",
                label_path=parsed["path"], label_sha256=parsed["sha256"],
                label_header=parsed["header"], label_issues=parsed["issues"], frames=frames)
