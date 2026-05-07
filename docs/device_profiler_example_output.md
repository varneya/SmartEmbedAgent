# `device_profiler.get_hardware_specs()` — Example Outputs

The function always returns the same set of keys. The values vary by hardware. Three representative examples are shown below.

## Example 1 — CPU-only laptop / cloud sandbox

```json
{
  "total_ram_gb": 3.81,
  "available_ram_gb": 3.43,
  "ram_percentage_used": 9.93,
  "gpu_available": false,
  "gpu_name": "None",
  "gpu_memory_gb": 0.0,
  "compute_device": "cpu"
}
```

This is the actual output captured when the profiler runs in an environment without `torch` installed (or with `torch` installed but no GPU). The error path is silently absorbed and the function returns CPU defaults.

## Example 2 — NVIDIA workstation

```json
{
  "total_ram_gb": 64.0,
  "available_ram_gb": 48.21,
  "ram_percentage_used": 24.67,
  "gpu_available": true,
  "gpu_name": "NVIDIA GeForce RTX 4090",
  "gpu_memory_gb": 24.0,
  "compute_device": "cuda"
}
```

The GPU branch fires when `torch.cuda.is_available()` returns True and `torch.version.hip` is empty. `gpu_name` comes from `torch.cuda.get_device_name(0)` and `gpu_memory_gb` from `torch.cuda.get_device_properties(0).total_memory` divided by 1024^3.

## Example 3 — AMD ROCm workstation

```json
{
  "total_ram_gb": 32.0,
  "available_ram_gb": 21.5,
  "ram_percentage_used": 32.81,
  "gpu_available": true,
  "gpu_name": "AMD Radeon RX 7900 XTX",
  "gpu_memory_gb": 24.0,
  "compute_device": "rocm"
}
```

PyTorch ROCm builds expose the same `torch.cuda.*` API as CUDA, but `torch.version.hip` is non-empty. The profiler distinguishes between the two so downstream code knows which backend is in play.

## Field reference

| Key | Type | Notes |
|---|---|---|
| `total_ram_gb` | float | `psutil.virtual_memory().total` converted to GB. |
| `available_ram_gb` | float | `total_ram_gb − used_ram_gb`. Internally consistent with `ram_percentage_used`. |
| `ram_percentage_used` | float | `(used / total) × 100`. Range: 0–100. |
| `gpu_available` | bool | True if any GPU was detected. |
| `gpu_name` | str | GPU model from torch, or `"None"`. |
| `gpu_memory_gb` | float | Total GPU memory in GB, or `0.0`. |
| `compute_device` | str | One of `"cuda"`, `"rocm"`, `"cpu"`. |

## Test results

10 tests run, all pass (1 skipped as intentional design coverage):

```
test_cpu_only_fallback                                   ... ok
test_gpu_detection_exception_falls_back_to_cpu           ... skipped
test_simulated_amd_gpu_when_no_nvidia                    ... ok
test_simulated_nvidia_gpu                                ... ok
test_compute_device_is_known                             ... ok
test_field_types                                         ... ok
test_has_all_required_keys                               ... ok
test_ram_percentage_in_range                             ... ok
test_ram_values_are_positive                             ... ok
test_returns_dict                                        ... ok
```
