from __future__ import annotations

from collections.abc import Sequence
import numbers

QWEN_IM_START = 151644
QWEN_IM_END = 151645
QWEN_USER = 872
QWEN_NL = 198
QWEN_IMAGE_PAD = 151655


def normalize_token(token):
    if isinstance(token, dict) and token.get("type") == "image":
        return QWEN_IMAGE_PAD
    if isinstance(token, numbers.Integral):
        return int(token)
    return ("embed", type(token).__name__)


def token_ids(tokens) -> list:
    key = next(k for k in tokens if isinstance(tokens[k], list))
    return [normalize_token(token[0]) for token in tokens[key][0]]


def user_span(ids: Sequence) -> tuple[int, int]:
    for i in range(len(ids) - 2):
        if ids[i:i + 3] == [QWEN_IM_START, QWEN_USER, QWEN_NL]:
            end = i + 3
            while end < len(ids) and ids[end] != QWEN_IM_END:
                end += 1
            return i + 3, end
    return 0, len(ids)


def visible_start(ids: Sequence) -> int:
    starts = [i for i, token in enumerate(ids) if token == QWEN_IM_START]
    return starts[1] if len(starts) > 1 else 0


def aux_span(combined: Sequence, main: Sequence) -> tuple[int, int]:
    combined_start, combined_end = user_span(combined)
    main_start, main_end = user_span(main)
    combined_user = combined[combined_start:combined_end]
    main_user = main[main_start:main_end]
    prefix = 0
    while prefix < min(len(combined_user), len(main_user)) and combined_user[prefix] == main_user[prefix]:
        prefix += 1
    return combined_start + prefix, combined_end


def cond_positions(ids: Sequence, start: int, end: int, cond_len: int) -> list[int]:
    """Map a trailing text span through Qwen's system removal and vision handling."""
    first_visible = visible_start(ids)
    length_without_vision_expansion = len(ids) - first_visible
    shift = cond_len - length_without_vision_expansion
    return [i - first_visible + shift for i in range(start, end) if 0 <= i - first_visible + shift < cond_len]


def sequence_positions(positions: Sequence[int], image_slots: Sequence[int], ref_shapes: Sequence[tuple[int, int]]) -> list[int]:
    out = []
    for position in positions:
        offset = sum(h * w for slot, (h, w) in zip(image_slots, ref_shapes) if slot <= position)
        out.append(position + offset)
    return out


def parse_block_range(spec: str, count: int) -> list[int]:
    spec = (spec or "all").strip().lower()
    if spec in ("", "all", "*"):
        return list(range(count))
    selected = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            left, right = (int(value) for value in part.split("-", 1))
            if left > right:
                left, right = right, left
            selected.extend(range(left, right + 1))
        else:
            selected.append(int(part))
    selected = list(dict.fromkeys(i for i in selected if 0 <= i < count))
    if not selected:
        raise ValueError(f"block_range {spec!r} selected no blocks in 0..{count - 1}")
    return selected
