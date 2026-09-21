from __future__ import annotations

import math

import torch
from comfy_api.latest import ComfyExtension, io
from typing_extensions import override

import comfy.model_management
import comfy.utils
import node_helpers

from .attn_patch import Qwen21ValueScalePatch
from .tokenize import aux_span, cond_positions, parse_block_range, sequence_positions, token_ids, user_span


def prepare_references(images, resolution):
    prepared = []
    latent_width = latent_height = resolution or 1024
    for name in sorted(images or {}, key=lambda value: int(value.rsplit("_", 1)[-1])):
        image = images[name]
        if image is None:
            continue
        samples = image[:1].movedim(-1, 1)
        if resolution > 0:
            ratio = samples.shape[3] / samples.shape[2]
            width = round(math.sqrt(resolution * resolution * ratio) / 32) * 32
            height = round(math.sqrt(resolution * resolution / ratio) / 32) * 32
        else:
            width = round(samples.shape[3] / 32) * 32
            height = round(samples.shape[2] / 32) * 32
        width, height = max(32, width), max(32, height)
        if (width, height) == (samples.shape[3], samples.shape[2]):
            resized = image[:1]
        else:
            resized = comfy.utils.common_upscale(samples, width, height, "lanczos", "disabled").movedim(1, -1)
        if not prepared:
            latent_width, latent_height = width, height
        rgb = resized[:, :, :, :3]
        if resized.shape[-1] > 3:
            rgb = rgb * resized[:, :, :, 3:] + (1.0 - resized[:, :, :, 3:])
        prepared.append((resized, rgb))
    return prepared, latent_width, latent_height


class QwenImage21PromptMix(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="QwenImage21PromptMix",
            display_name="Qwen Image 2.1 Prompt Mix",
            category="model/conditioning/qwen image",
            description="Keeps the main prompt at full strength and value-scales only auxiliary prompt tokens. Active weighting disables Qwen 2.1 prefix KV caching.",
            inputs=[
                io.Model.Input("model"),
                io.Clip.Input("clip"),
                io.String.Input("text_main", multiline=True, dynamic_prompts=True),
                io.String.Input("text_aux", multiline=True, dynamic_prompts=True),
                io.String.Input("negative_prompt", multiline=True, dynamic_prompts=True),
                io.Float.Input("aux_strength", default=0.45, min=0.0, max=2.0, step=0.05),
                io.Combo.Input("separator", options=["newline", "space", "comma"], default="newline"),
                io.String.Input("block_range", default="all"),
                io.Vae.Input("vae", optional=True),
                io.Int.Input("resolution", default=1024, min=0, max=4096, step=32),
                io.Autogrow.Input(
                    "images",
                    template=io.Autogrow.TemplateNames(io.Image.Input("image"), names=[f"image_{i}" for i in range(1, 17)], min=0),
                ),
            ],
            outputs=[
                io.Model.Output(),
                io.Conditioning.Output(display_name="positive"),
                io.Conditioning.Output(display_name="negative"),
                io.Latent.Output(display_name="latent"),
                io.String.Output(display_name="debug"),
            ],
        )

    @classmethod
    def execute(cls, model, clip, text_main, text_aux, negative_prompt, aux_strength, separator="newline", block_range="all", vae=None, resolution=1024, images=None):
        prepared, latent_width, latent_height = prepare_references(images, resolution)
        images_vl = [rgb for _, rgb in prepared]
        ref_latents = [vae.encode(image) for image, _ in prepared] if vae is not None else []
        keep_vision = not ref_latents

        main = (text_main or "").strip()
        aux = (text_aux or "").strip()
        use_aux = bool(aux) and float(aux_strength) != 0.0
        separator_text = {"newline": "\n", "space": " ", "comma": ", "}[separator]
        positive_text = main + separator_text + aux if main and use_aux else aux if use_aux else main

        tokenize_args = {"images": images_vl, "keep_vision": keep_vision, "prevent_empty_text": True}
        positive_tokens = clip.tokenize(positive_text, **tokenize_args)
        negative_tokens = clip.tokenize(negative_prompt, **tokenize_args)
        positive = clip.encode_from_tokens_scheduled(positive_tokens)
        negative = clip.encode_from_tokens_scheduled(negative_tokens)
        if ref_latents:
            values = {"reference_latents": ref_latents}
            positive = node_helpers.conditioning_set_values(positive, values, append=True)
            negative = node_helpers.conditioning_set_values(negative, values, append=True)

        latent = torch.zeros([1, 64, latent_height // 16, latent_width // 16], device=comfy.model_management.intermediate_device())
        reason = "empty aux" if not aux else "aux_strength=0.0" if float(aux_strength) == 0.0 else None
        if reason is not None:
            return io.NodeOutput(model, positive, negative, {"samples": latent}, f"{reason}; main only, no patch")
        if abs(float(aux_strength) - 1.0) < 1e-6:
            return io.NodeOutput(model, positive, negative, {"samples": latent}, "aux_strength=1.0; no patch")

        ids = token_ids(positive_tokens)
        if main:
            main_ids = token_ids(clip.tokenize(main, **tokenize_args))
            start, end = aux_span(ids, main_ids)
        else:
            start, end = user_span(ids)
        context_positions = cond_positions(ids, start, end, positive[0][0].shape[1])
        metadata = positive[0][1]
        slots = metadata.get("image_slots", [])
        ref_shapes = [(latent.shape[-2], latent.shape[-1]) for latent in ref_latents]
        positions = sequence_positions(context_positions, slots, ref_shapes)
        if not positions:
            return io.NodeOutput(model, positive, negative, {"samples": latent}, "no auxiliary positions found; no patch")

        diffusion_model = model.get_model_object("diffusion_model")
        blocks = getattr(diffusion_model, "transformer_blocks", None)
        if blocks is None:
            raise RuntimeError("Qwen Image 2.1 Prompt Mix requires a Qwen Image 2.1 model")
        selected = parse_block_range(block_range, len(blocks))
        patched = model.clone()
        patched.set_model_attn1_patch(Qwen21ValueScalePatch(positions, aux_strength, selected))
        debug = f"scaled {len(positions)} auxiliary tokens to {float(aux_strength):.4f}; patched blocks: {selected}"
        if ref_latents:
            debug += f"; {len(ref_latents)} reference latent(s)"
        elif images_vl:
            debug += f"; {len(images_vl)} vision-only reference(s)"
        return io.NodeOutput(patched, positive, negative, {"samples": latent}, debug)


class Qwen21WeightedExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [QwenImage21PromptMix]
