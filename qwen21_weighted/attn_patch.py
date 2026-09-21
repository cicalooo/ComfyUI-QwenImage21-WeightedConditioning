from __future__ import annotations

import torch


class Qwen21ValueScalePatch:
    def __init__(self, positions, strength, blocks):
        self.positions = tuple(positions)
        self.strength = float(strength)
        self.blocks = frozenset(blocks)

    def __call__(self, q, k, v, pe, attn_mask, extra_options):
        if extra_options.get("block_index") not in self.blocks:
            return {"q": q, "k": k, "v": v, "pe": pe}

        branches = extra_options.get("cond_or_uncond") or [0]
        if v.shape[0] % len(branches):
            return {"q": q, "k": k, "v": v, "pe": pe}
        rows_per_branch = v.shape[0] // len(branches)
        rows = []
        for branch, kind in enumerate(branches):
            if kind == 0:
                rows.extend(range(branch * rows_per_branch, (branch + 1) * rows_per_branch))
        positions = [position for position in self.positions if 0 <= position < v.shape[2]]
        if rows and positions:
            v = v.clone()
            row_index = torch.tensor(rows, device=v.device)
            position_index = torch.tensor(positions, device=v.device)
            v[row_index[:, None], :, position_index[None, :], :] *= self.strength
        return {"q": q, "k": k, "v": v, "pe": pe}
