# Embedding parity (gate G2): SCIFACT

Cosine between torch (`sentence-transformers`) and ONNX Runtime vectors for the same
text. Chunks come from the production chunker and text format. Documents are encoded
in batches and each query alone, as in production. No database is involved.

- **Result**: PASS (fp32: mean >= 0.9999 and min >= 0.999 on both sets; int8 is not gated here)
- **Texts**: 2000 chunks sampled from 20672 (seed 0), 1109 queries
- **Measured**: 2026-11-07 14:28 UTC at commit `86dfc37`
- **Hardware**: Intel(R) Core(TM) i5-10200H CPU @ 2.40GHz, 8 logical CPUs, CPU-only, Windows-11-10.0.26300-SP0
- **Python**: 3.12.13
- **Versions**: torch 2.13.0, sentence-transformers 6.0.0, onnxruntime 1.31.1, tokenizers 0.22.2, numpy 2.5.2
- **Torch model**: `sentence-transformers/all-MiniLM-L6-v2@1110a243fdf4706b3f48f1d95db1a4f5529b4d41`
- **ONNX fp32**: `onnx/model.onnx`, SHA-256 `6fd5d72fe4589f189f8ebc006442dbb529bb7ce38f8082112682524616046452`
- **ONNX int8**: `onnx/model_quint8_avx2.onnx`, SHA-256 `b941bf19f1f1283680f449fa6a7336bb5600bdcd5f84d10ddc5cd72218a0fd21`

## Cosine with torch

| Variant | Set | Texts | Mean | Min | P1 |
|---|---|---:|---:|---:|---:|
| fp32 | chunks | 2000 | 1.00000000 | 1.00000000 | 1.00000000 |
| fp32 | queries | 1109 | 1.00000000 | 1.00000000 | 1.00000000 |
| int8 | chunks | 2000 | 0.99195241 | 0.97495508 | 0.98647238 |
| int8 | queries | 1109 | 0.99261718 | 0.98569447 | 0.98865556 |

## Encode time

Milliseconds per 100 texts on this laptop, after one unmeasured warm-up, one run each. Torch used 4 threads and ONNX used 1, as configured for serving, so these columns are not a like-for-like speed comparison between runtimes. Load-dependent and indicative only.

| Runtime | Threads | Chunks | Queries |
|---|---:|---:|---:|
| torch | 4 | 2710 | 1385 |
| ONNX fp32 | 1 | 5060 | 790 |
| ONNX int8 | 1 | 4389 | 677 |

## Reproduce

`uv sync --group ml`, then `uv run python -m scripts.benchmark parity-embeddings --dataset scifact --sample 2000 --seed 0`.

## Sanity checks (diagnostic, separate from the gate, same commit and environment)

- **The fp32 cosines round to 1 but the runtimes differ.** On the first 200 sampled chunks, no
  vector pair was byte-identical. The largest component difference was 1.75e-07, and the
  lowest cosine was 0.9999999999995.
- **Truncation was exercised.** `tokenizer.json` stores a limit of 128 tokens and the model
  uses 256. Of the 2,000 sampled chunks, 328 exceed 128 tokens (longest 228), and none exceed
  256. Those 328 have a lowest cosine of 0.9999999999995. No query exceeds 128 (longest 79).
- **Past 256 tokens.** Chunks never reach 256 here, so 200 whole documents over 256 tokens
  were also compared. Their lowest cosine was 0.9999999999996. In the corpus, 3,681 of 5,183
  documents exceed 256 tokens, but only as whole documents, not as chunks.
