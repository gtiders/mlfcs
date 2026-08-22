from __future__ import annotations

from typing import Literal

import jax

JaxPlatform = Literal["auto", "cpu", "gpu"]


def configure_jax(platform: JaxPlatform = "auto") -> None:
    """Configure JAX before the first numerical kernel initializes a backend.

    ``gpu`` requires a CUDA-enabled JAX installation. Backend selection is a
    process-wide JAX setting and cannot be changed after JAX is initialized.
    """
    if platform not in {"auto", "cpu", "gpu"}:
        raise ValueError("jax_platform must be 'auto', 'cpu', or 'gpu'")
    jax.config.update("jax_enable_x64", True)
    if platform != "auto":
        try:
            jax.config.update("jax_platforms", platform)
            devices = jax.devices()
        except RuntimeError as error:
            raise RuntimeError(
                f"JAX platform {platform!r} is unavailable or the backend is already "
                "initialized; GPU execution requires a CUDA-enabled jaxlib"
            ) from error
        if not devices or any(device.platform != platform for device in devices):
            raise RuntimeError(f"JAX did not activate the requested {platform!r} platform")
