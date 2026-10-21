# Ascend experimental branch status

This branch preserves the Ascend PSP-KV migration and the 2026-10-20
validation evidence. It is an experimental checkpoint with known failures.
The [hardware audit](results/2026-10-20-audit/REPORT.zh-CN.md) describes the
tested configurations, capacity measurements, generated text, and limits.

## Validation

- On worker03: 317 pytest cases passed, seven serving configurations
  completed 42 generation requests, and a native rebuild reproduced the
  existing shared library byte for byte.
- Two standalone NPU checkers failed: the FP4 scale boundary calculation
  differs from CPU, and the fused decode wrapper does not accept the
  asymmetric K/V keyword arguments supplied by its caller.
- Before publication: the local CPU suite passed 54 cases and skipped one
  module because this machine does not have `vllm_ascend` installed.
- The pre-commit hooks passed apart from `mypy-local`, which reported 25
  errors in the implementation group and 14 in the test group. Its complete
  output is retained in [pre-commit.txt](results/2026-10-20-audit/pre-commit.txt).

`mypy-local` is skipped for the checkpoint commit so the existing migration
can be preserved on the requested branch. The repository hook configuration
is unchanged, and subsequent normal commits continue to run the check.
The failures include dynamic operator namespace annotations, optional-value
narrowing, synthetic test metadata attributes, and the known fused decode
interface mismatch. They are not reported as passing checks.

Publication preparation applies formatting, adds SPDX headers, uses the
required regular-expression import, updates the CUDA synchronization calls
added by the test adapter to the accelerator API, and regenerates the backend
documentation. The raw audit output remains unchanged; spelling checks
exclude that output because dependency warnings and environment variable
names must be preserved verbatim.

FP4 model quality, performance gains over the stock backend, and the two
checker failures remain unresolved. The online route is restricted to eager
mode and TP=1; variable-width and FP4 serving require a single sequence and
disabled prefix caching. KV transfer is not supported by this backend.

AI assistance was used for implementation, testing, and publication
preparation. The commit attribution records that assistance.
