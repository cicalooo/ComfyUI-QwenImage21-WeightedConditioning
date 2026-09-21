import torch

from qwen21_weighted.attn_patch import Qwen21ValueScalePatch


def test_scales_only_positive_rows_and_selected_block():
    q = torch.ones(4, 2, 6, 3)
    v = torch.ones_like(q)
    patch = Qwen21ValueScalePatch([2, 4], 0.5, [1])
    out = patch(q, q, v, None, None, {"block_index": 1, "cond_or_uncond": [1, 0]})
    assert torch.all(out["v"][:2] == 1)
    assert torch.all(out["v"][2:, :, [2, 4]] == 0.5)
    assert torch.all(out["v"][2:, :, [0, 1, 3, 5]] == 1)


def test_unselected_block_is_noop():
    q = torch.ones(1, 2, 4, 3)
    v = torch.ones_like(q)
    out = Qwen21ValueScalePatch([1], 0.2, [3])(q, q, v, None, None, {"block_index": 2})
    assert out["v"] is v
