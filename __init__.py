try:
    from .qwen21_weighted import __version__
    from .qwen21_weighted.nodes import Qwen21WeightedExtension
except ImportError:
    from qwen21_weighted import __version__
    from qwen21_weighted.nodes import Qwen21WeightedExtension


async def comfy_entrypoint() -> Qwen21WeightedExtension:
    return Qwen21WeightedExtension()


__all__ = ["comfy_entrypoint", "__version__"]
