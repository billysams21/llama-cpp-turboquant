# Change Summary

Date: 2026-04-14
Branch: feature/planarquant-kv-cache

## Scope
This document summarizes all current working-tree changes.

## Files Changed
- Deleted: AGENTS.md
- Deleted: CONTRIBUTING.md
- Modified: CMakeLists.txt
- Modified: ggml/src/ggml-cpu/ops.cpp
- Modified: ggml/src/ggml-quants.h
- Modified: ggml/src/ggml-turbo-quant.c
- Modified: ggml/src/ggml-cuda/turbo-innerq.cuh
- Modified: ggml/src/ggml-cuda/turbo-innerq.cu
- Modified: src/llama-kv-cache.cpp

## What Changed
- CMake MSVC setup:
  - Added _USE_MATH_DEFINES in CMakeLists.txt for MSVC builds.

- CPU turbo quant global symbol linkage fix:
  - ggml/src/ggml-quants.h:
    - Added GGML_API declaration for turbo3_cpu_wht_group_size.
  - ggml/src/ggml-turbo-quant.c:
    - Changed definition to GGML_API int turbo3_cpu_wht_group_size = 0.
  - ggml/src/ggml-cpu/ops.cpp:
    - Included ggml-quants.h.
    - Removed local extern int turbo3_cpu_wht_group_size declaration and now uses shared API declaration.

- CUDA InnerQ cross-DLL symbol export/import fix (Windows shared-library linking):
  - ggml/src/ggml-cuda/turbo-innerq.cuh:
    - Included ggml-backend.h.
    - Converted extern declarations and function declarations to GGML_BACKEND_API.
  - ggml/src/ggml-cuda/turbo-innerq.cu:
    - Converted global definitions and exported functions to GGML_BACKEND_API:
      - g_innerq_finalized
      - g_innerq_scale_inv_host
      - turbo_innerq_publish
      - turbo_innerq_needs_tensor_update
      - turbo_innerq_mark_tensor_updated
  - src/llama-kv-cache.cpp:
    - Included ggml-backend.h.
    - Replaced plain extern declarations under GGML_USE_CUDA with GGML_BACKEND_API declarations for:
      - g_innerq_finalized
      - g_innerq_scale_inv_host
      - turbo_innerq_needs_tensor_update
      - turbo_innerq_mark_tensor_updated

## Why These Changes
- Resolve unresolved external symbol errors on MSVC when building shared libraries across ggml-base, ggml-cpu, ggml-cuda, and llama targets.
- Ensure symbol names and visibility are consistent across translation units and DLL boundaries.

## Validation Status
- Confirmed successful build for ggml-cpu target after CPU turbo linkage fix.
- Full llama target re-check after CUDA InnerQ export/import changes is pending in this session (build was skipped by user choice).

## Notes
- AGENTS.md and CONTRIBUTING.md are currently deleted in the working tree and were not restored by these changes.
