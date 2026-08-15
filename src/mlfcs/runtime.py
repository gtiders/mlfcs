from __future__ import annotations

import os
from contextlib import nullcontext
from typing import Literal

import jax

JaxPlatform = Literal["auto", "cpu", "gpu"]
TransferGuard = Literal["log", "disallow", "log_explicit", "disallow_explicit"]


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


def transfer_guard():
    """Enable optional transfer auditing through ``MLFCS_JAX_TRANSFER_GUARD``.

    The default is inert.  CI or a developer can set the environment variable
    to a documented JAX guard level and turn accidental host/device copies
    into log entries or errors without adding a user-facing fitting parameter.
    """
    level = os.environ.get("MLFCS_JAX_TRANSFER_GUARD")
    if level is None:
        return nullcontext()
    if level not in {"log", "disallow", "log_explicit", "disallow_explicit"}:
        raise ValueError(
            "MLFCS_JAX_TRANSFER_GUARD must be log, disallow, log_explicit, or disallow_explicit"
        )
    return jax.transfer_guard(level)
