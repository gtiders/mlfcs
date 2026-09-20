from __future__ import annotations

from typing import Literal

import jax

JaxPlatform = Literal["auto", "cpu", "gpu"]


def resolve_jax_device(platform: JaxPlatform = "auto") -> jax.Device:
    """Select one fitting device without changing JAX's process-wide backend.

    The public fitting choice must not reconfigure another library's JAX
    workload in the same Python process.  The selected device is passed
    explicitly to MLFCS's persistent fitting buffers instead.
    """
    if platform not in {"auto", "cpu", "gpu"}:
        raise ValueError("jax_platform must be 'auto', 'cpu', or 'gpu'")
    jax.config.update("jax_enable_x64", True)
    try:
        devices = jax.devices() if platform == "auto" else jax.devices(platform)
    except RuntimeError as error:
        raise RuntimeError(
            f"JAX platform {platform!r} is unavailable; GPU execution requires "
            "a CUDA-enabled jaxlib"
        ) from error
    if not devices:
        raise RuntimeError(f"JAX did not provide a {platform!r} device")
    return devices[0]


def configure_jax(platform: JaxPlatform = "auto") -> None:
    """Validate legacy explicit JAX selection without changing global backend state."""
    resolve_jax_device(platform)
