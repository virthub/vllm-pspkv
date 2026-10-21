# 910B3 Qwen3-0.6B cache-path measurements (2026-10-17)

These are measured serving results on `worker03`, a single Ascend 910B3 (host
NPU 4), using `quay.io/ascend/vllm-ascend:v0.20.2rc1`. All eight runs used
the same Qwen3-0.6B model, tensor parallelism 1, eager mode, a 128-token KV
block, `--max-model-len 256`, `--max-num-seqs 1`, prefix caching disabled,
and `--gpu-memory-utilization 0.35`. The PSP-KV runs loaded this vLLM fork
through `PYTHONPATH` and the vLLM-Ascend backend overlay. The stock BF16 run
used the image's original vLLM-Ascend backend. The native extension was built
for `dav-2201` from `ascend/native/pspkv_fp8.asc`.

| Cache path | KV cache capacity (tokens) | Output throughput (tokens/s) | Mean TTFT (ms) | Mean TPOT (ms) |
| --- | ---: | ---: | ---: | ---: |
| Stock vLLM-Ascend BF16 | 188,672 | 18.26 | 121.79 | 52.59 |
| PSP-KV native fixed-width FP8 scatter and decode | 377,344 | 16.32 | 136.65 | 58.84 |
| PSP-KV native uniform FP4 scatter and decode | 754,816 | 7.83 | 275.26 | 122.91 |
| PSP-KV PyTorch uniform FP4 reference | 754,816 | 2.02 | 869.97 | 483.91 |
| PSP-KV native FP4, separate K/V scale per dimension | 754,816* | 2.47 | 575.92 | 398.92 |
| PSP-KV FP4, separate K/V scale per dimension, torch SDPA | 754,816* | 0.98 | 1356.83 | 1010.28 |
| PSP-KV online variable-width arena, separate native score/value kernels | 377,344 | 9.86 | 257.19 | 96.35 |
| PSP-KV online variable-width arena, fused native decode kernel | 377,344 | 10.04 | 255.99 | 94.56 |

All eight benchmarks completed 12/12 requests. Each request had 32 input and
32 output tokens, with concurrency 1, two warmup requests, seed 123, and
`--ignore-eos`. The benchmark used the server's default sampling temperature;
it did not force greedy decoding. The exact vLLM benchmark command run inside
each serving container was:

```bash
vllm bench serve --backend openai --host 127.0.0.1 --port 8000 \
  --model Qwen/Qwen3-0.6B --dataset-name random \
  --num-prompts 12 --input-len 32 --output-len 32 \
  --max-concurrency 1 --ignore-eos --seed 123 --num-warmups 2 \
  --save-result --result-dir /tmp --result-filename <run>.json
```

The JSON files beside this document are the unedited `vllm bench serve`
result files. The cache capacity figures come from the corresponding serving
logs. Each run reserved about 20.16 GiB for KV cache, so the capacity ratios
show how many more token slots fit in that budget; they are **not** a measured
reduction of the entire server's HBM allocation. FP4 provided approximately
4 times BF16 token capacity and 2 times FP8 token capacity. The native FP4
path was 3.88 times faster than the PyTorch FP4 reference, but only 0.43
times the stock BF16 throughput and 0.48 times native FP8 throughput.
The grouped-scale FP4 runs used `PSPKV_ASCEND_FP4_GROUP_SIZE=1` and generated
readable Chinese for the short greedy prompt. The extra scale tensor is not
included in the reported KV slot capacity (*), so those two entries are
allocator capacity figures rather than total KV memory savings. On this
workload, torch SDPA was 2.52 times slower than the native grouped-FP4 path.
The opt-in fused variable-width kernel improved serving output throughput by
about 1.8% over the separate native score/value kernels in this benchmark.
Both variable-width runs used the conservative 0.05 completed-block error
threshold and a 1-byte-per-element arena budget. Their doubled allocator
capacity relative to BF16 is the FP8 budget; it does not demonstrate an
additional FP4 saving or a reduction of whole-server HBM use. The fused
kernel still reads compressed K/V values individually from global memory.

For a separate greedy smoke request, each path received the same prompt
`北京是中国的` with `max_tokens=16` and `temperature=0`. Stock BF16 returned
normal Chinese text starting `首都，也是中国最大的城市`; native fixed-width FP8 also
returned readable Chinese text. Both legacy per-head-scale FP4 paths produced
corrupted text after
the initial `首都` (including `IMITER` and replacement characters). This single
prompt is a failure case, not a complete accuracy evaluation. The FP4 quality
failure was present in both the PyTorch and native implementations; the native
kernel alone cannot be blamed from this observation.

The original repository tests were run inside the Ascend container on NPU 4
with `PSPKV_TEST_DEVICE=npu` and the native library path set:

```bash
PSPKV_TEST_DEVICE=npu \
  PSPKV_ASCEND_NATIVE_LIB=/workspace/pspkv-native-build/libpspkv_ascend_native.so \
  /workspace/pspkv-native-build/.venv/bin/python -m pytest \
  tests/v1/attention/test_psp_kv_correctness.py -q
PSPKV_TEST_DEVICE=npu \
  PSPKV_ASCEND_NATIVE_LIB=/workspace/pspkv-native-build/libpspkv_ascend_native.so \
  /workspace/pspkv-native-build/.venv/bin/python -m pytest \
  tests/v1/attention/test_psp_kv_mixed_precision.py -q
```

They passed 161 and 97 cases respectively. The first suite covered packed
scatter/dequantization and the basic, enhanced, BTKV and scaled basic
attention contracts, including token-major scaled FP4. The latter tests
exercise the original precision-selection logic. These checks establish
operator/reference agreement on their tested inputs; they do not establish
acceptable end-to-end FP4 language quality.

The first six rows measured fixed-stride routes. The last two measured the
online variable-width arena with and without fused decode. The opt-in
`PSPKV_ASCEND_VARIABLE_PRECISION=1` route connects an FP4/FP8 compact arena
to live vLLM allocation; its conservative completed-block policy did not
select FP4 for this model at the tested quality threshold. The native
basic/enhanced/BTKV contract kernels still use scalar arithmetic, while the
variable route has UB-staged score/value readers and an opt-in fused decode
kernel. The fused kernel is still scalar for compressed K/V loads. A faster
vectorized or Cube-based compressed attention implementation and a validated
quality-preserving FP4 policy remain necessary before claiming the original
PSP-KV performance goal.
