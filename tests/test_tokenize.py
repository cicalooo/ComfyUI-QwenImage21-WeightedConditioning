from qwen21_weighted.tokenize import (
    QWEN_IM_END,
    QWEN_IM_START,
    QWEN_NL,
    QWEN_USER,
    aux_span,
    cond_positions,
    parse_block_range,
    sequence_positions,
)


def wrapped(user):
    return [QWEN_IM_START, 8948, QWEN_NL, 10, QWEN_IM_END, QWEN_IM_START, QWEN_USER, QWEN_NL, *user, QWEN_IM_END]


def test_aux_mapping_after_removed_vision():
    main = wrapped([151652, 151655, 151653, 20, 21])
    combined = wrapped([151652, 151655, 151653, 20, 21, 30, 31])
    start, end = aux_span(combined, main)
    # The one image placeholder is removed, making context one row shorter than raw visible ids.
    assert cond_positions(combined, start, end, len(combined) - 6) == [7, 8]


def test_reference_latents_shift_sequence_positions():
    assert sequence_positions([5, 8], [2, 7], [(3, 4), (2, 2)]) == [17, 24]


def test_block_range():
    assert parse_block_range("1-3,5", 8) == [1, 2, 3, 5]
