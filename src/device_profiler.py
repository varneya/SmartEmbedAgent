"""
Device Profiling Module
=======================

Inspects the host machine and returns a dictionary describing the available
compute resources. The recommendation agent uses this to filter out embedding
models that would not fit in memory or that would be unacceptably slow on
CPU-only hardware.

Returned dictionary keys
------------------------
    total_ram_gb        : float — total system RAM in GB
    available_ram_gb    : float — currently free RAM in GB
    ram_percentage_used : float — percentage of RAM in use (0–100)
    gpu_available       : bool  — whether a GPU is detected
    gpu_name            : str   — GPU model name, or "None" if no GPU
    gpu_memory_gb       : float — total GPU memory in GB, or 0.0 if no GPU
    compute_device      : str   — "cuda", "rocm", or "cpu"

GPU detection precedence
------------------------
    1. NVIDIA / CUDA via `torch.cuda`
    2. AMD / ROCm via `torch.version.hip` (PyTorch ROCm builds expose CUDA
       APIs against ROCm under the hood, so the same `torch.cuda` calls work)
    3. CPU-only fallback

All GPU code paths are wrapped in try/except — a missing or broken GPU stack
should never crash the profiler; it should just degrade to CPU-only.
"""

from __future__ import annotations

import json
from typing import Any, Dict


def _bytes_to_gb(b: int) -> float:
    """Convert a byte count to GB, rounded to 2 decimal places."""
    return round(b / (1024 ** 3), 2)


def _detect_nvidia_gpu() -> Dict[str, Any]:
    """Try to detect an NVIDIA GPU via CUDA. Returns {} on any failure."""
    try:
        import torch

        if not torch.cuda.is_available():
            return {}

        # `torch.version.hip` is set on ROCm builds, even though `cuda.is_available`
        # may also be True. We treat ROCm separately so the caller knows which
        # backend is in play.
        if getattr(torch.version, "hip", None):
            return {}

        device_index = 0
        gpu_name = torch.cuda.get_device_name(device_index)
        props = torch.cuda.get_device_properties(device_index)
        gpu_memory_gb = _bytes_to_gb(props.total_memory)

        return {
            "gpu_available": True,
            "gpu_name": gpu_name,
            "gpu_memory_gb": gpu_memory_gb,
            "compute_device": "cuda",
        }
    except Exception as e:
        # Surface the failure but don't propagate — caller will fall through.
        print(f"[device_profiler] NVIDIA detection failed: {type(e).__name__}: {e}")
        return {}


def _detect_amd_gpu() -> Dict[str, Any]:
    """Try to detect an AMD GPU via ROCm. Returns {} on any failure.

    PyTorch's ROCm builds reuse the `torch.cuda.*` API surface against ROCm,
    so the detection logic mirrors NVIDIA's. The distinguishing signal is
    `torch.version.hip`, which is non-empty on ROCm builds.
    """
    try:
        import torch

        if not getattr(torch.version, "hip", None):
            return {}
        if not torch.cuda.is_available():
            return {}

        device_index = 0
        gpu_name = torch.cuda.get_device_name(device_index)
        props = torch.cuda.get_device_properties(device_index)
        gpu_memory_gb = _bytes_to_gb(props.total_memory)

        return {
            "gpu_available": True,
            "gpu_name": gpu_name,
            "gpu_memory_gb": gpu_memory_gb,
            "compute_device": "rocm",
        }
    except Exception as e:
        print(f"[device_profiler] AMD/ROCm detection failed: {type(e).__name__}: {e}")
        return {}


def get_hardware_specs() -> Dict[str, Any]:
    """Inspect the host and return hardware specs as a dictionary.

    Always returns the full set of keys defined in the module docstring,
    falling back to safe defaults when a piece of hardware isn't detected.
    """
    import psutil

    # ----- RAM -----
    vm = psutil.virtual_memory()
    total_ram_bytes = vm.total
    used_ram_bytes = vm.used
    total_ram_gb = _bytes_to_gb(total_ram_bytes)
    # Compute available_ram_gb from total - used so the math is internally
    # consistent with ram_percentage_used (psutil's `available` uses a
    # different definition on Linux that includes reclaimable cache).
    available_ram_gb = round(total_ram_gb - _bytes_to_gb(used_ram_bytes), 2)
    ram_percentage_used = (
        round((used_ram_bytes / total_ram_bytes) * 100, 2) if total_ram_bytes else 0.0
    )

    # ----- GPU (NVIDIA → AMD → CPU) -----
    gpu_info = _detect_nvidia_gpu() or _detect_amd_gpu()
    if not gpu_info:
        gpu_info = {
            "gpu_available": False,
            "gpu_name": "None",
            "gpu_memory_gb": 0.0,
            "compute_device": "cpu",
        }

    return {
        "total_ram_gb": total_ram_gb,
        "available_ram_gb": available_ram_gb,
        "ram_percentage_used": ram_percentage_used,
        **gpu_info,
    }


def get_hardware_specs_json(indent: int = 2) -> str:
    """Convenience wrapper returning the specs as a JSON string."""
    return json.dumps(get_hardware_specs(), indent=indent)


# Backwards-compatible alias used elsewhere in the codebase.
profile_device = get_hardware_specs


if __name__ == "__main__":
    specs = get_hardware_specs()
    print("Hardware specs:")
    print(json.dumps(specs, indent=2))
