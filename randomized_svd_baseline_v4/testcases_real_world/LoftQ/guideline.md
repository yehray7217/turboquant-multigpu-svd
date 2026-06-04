# Guideline：LoftQ-inspired LLM Quantization Residual Testcase

## Goal

Prepare a real AI / LLM testcase matrix for `randomized_svd_multigpu_v4`.

This testcase is based on LoftQ:

> Instead of directly applying low-rank approximation to the original LLM weight matrix (W), construct a quantization residual matrix (R = W - Q(W)), where (Q(W)) is the dequantized low-bit quantized version of (W). Then run distributed randomized SVD on (R).

The purpose is to benchmark:

```text
distributed randomized SVD on a real LLM quantization residual matrix
```

and compare:

```text
no compression
TQ 8-bit
TQ 4-bit
```

This is a **LoftQ-inspired residual SVD benchmark**, not a full reproduction of LoftQ fine-tuning.

---

## Current Directory

The relevant working directory is:

```text
/work/pbr03617/turboquant-multigpu-svd/randomized_svd_baseline_v4/testcases_real_world/LoftQ
```

The reference materials are already placed under:

```text
/work/pbr03617/turboquant-multigpu-svd/randomized_svd_baseline_v4/testcases_real_world/LoftQ/references
```

Current reference files:

```text
references/
├── LoftQ/
└── LoftQ.md
```

where:

```text
LoftQ/     = official LoftQ GitHub repository
LoftQ.md   = LoftQ paper text parsed by MarkItDown
```

Please inspect both before implementing the preprocessing script.

---

## Expected Final Output

Please produce:

```text
testcases_real_world/LoftQ/
├── .gitignore
├── matrix.f32
├── matrix_debug.f32
├── meta.json
├── README.md
├── exp_turboquant/
│   ├── README.md
│   ├── ctrl.slurm
│   ├── b8-exp.slurm
│   ├── b4-exp.slurm
│   ├── output_logs/
│   └── error_logs/
├── scripts/
│   └── prepare_loftq_residual_matrix.py
└── references/
    ├── LoftQ/
    └── LoftQ.md
```

where:

```text
.gitignore       = ignore large model shards, cache, and raw .f32 matrices
matrix.f32       = formal benchmark residual matrix
matrix_debug.f32 = optional small/debug residual matrix
meta.json        = human-readable metadata, not consumed by the CUDA program
README.md        = source, preprocessing, sanity check, and experiment notes
exp_turboquant/  = no-compression / TQ 8-bit / TQ 4-bit experiment configs and logs
scripts/         = preprocessing scripts
references/      = LoftQ paper/repository references
```

Do not commit large model shards, Hugging Face cache files, or generated `.f32` matrices unless the team explicitly decides to version the dataset.

Recommended `.gitignore`:

```gitignore
matrix*.f32
raw_data/
cache/
hf_cache/
*.safetensors
*.bin
*.pt
*.pth
*.npy
*.npz
```

The final matrix file must be:

```text
matrix.f32
```

Format:

```text
raw float32 binary
row-major layout
no header
```

Do not use `.npy`, `.pt`, `.safetensors`, `.hdf5`, or `.mat` as the final CUDA input file.

Those formats may be used during preprocessing, but the final input for `randomized_svd_multigpu_v4` must be `matrix.f32`.

---

## Important Conceptual Clarification

Do not describe this testcase as directly reproducing LoRA.

LoRA learns a low-rank update:

[
\Delta W = BA
]

while keeping the pretrained weight (W) frozen.

LoftQ is relevant here because it initializes LoRA adapters to compensate quantization error. A key low-rank approximation target is the quantization residual:

[
R = W - Q(W)
]

where (Q(W)) is the dequantized quantized weight.

Therefore, this testcase should be described as:

```text
LoftQ-inspired quantization residual matrix benchmark
```

not:

```text
LoRA reproduction
```

and not simply:

```text
SVD compression of the original pretrained weight
```

---

## Matrix Definition

Let:

[
W \in \mathbb{R}^{m \times n}
]

be one dense LLM weight matrix, for example an MLP projection weight.

Apply low-bit quantization and dequantization:

[
Q(W) = \operatorname{dequantize}(\operatorname{quantize}(W))
]

Then define:

[
R = W - Q(W)
]

The final file `matrix.f32` should store (R), not (W).

So:

```text
rows    = rows of selected LLM weight tensor
columns = columns of selected LLM weight tensor
entry   = quantization residual W[i,j] - Q(W)[i,j]
```

---

## Recommended Model and Tensor

Use a model that is easy to access on Hugging Face and has a reasonably large dense weight matrix.

Preferred first target:

```text
Mistral-7B
tensor: model.layers.0.mlp.gate_proj.weight
expected shape: about 14336 x 4096
```

This shape is good for the current distributed rSVD pipeline:

```text
m > n
dense real AI matrix
large enough to benchmark communication
not too large to preprocess
```

If Mistral-7B access is inconvenient, use a smaller debug model first, such as TinyLlama.

Debug option:

```text
TinyLlama-1.1B
tensor: model.layers.0.mlp.gate_proj.weight
expected shape: about 5632 x 2048
```

If using the debug model, write it to:

```text
matrix_debug.f32
```

and keep the formal benchmark file name:

```text
matrix.f32
```

---

## Quantization Method

Prefer to follow the LoftQ repository’s quantization behavior if feasible.

Please inspect:

```text
references/LoftQ/
references/LoftQ.md
```

and identify the relevant quantization / LoftQ initialization code.

However, do not attempt to reproduce full LoftQ training.

The preprocessing script only needs to produce:

[
R = W - Q(W)
]

If integrating the exact LoftQ quantizer is too time-consuming, use a clearly documented fallback quantizer:

```text
group-wise or row-wise symmetric 4-bit quantization
```

Fallback example:

For each row or group:

[
s = \frac{\max |W|}{7}
]

[
q = \operatorname{clip}(\operatorname{round}(W/s), -7, 7)
]

[
Q(W) = s q
]

Then:

[
R = W - Q(W)
]

If using fallback quantization, clearly write in `README.md` and `meta.json`:

```text
This is LoftQ-inspired, not an exact LoftQ quantizer.
```

---

## Suggested Preprocessing Script

Create:

```text
scripts/prepare_loftq_residual_matrix.py
```

The script should:

1. Load selected LLM weight tensor (W).

2. Convert it to FP32.

3. Apply 4-bit quantization and dequantization to obtain (Q(W)).

4. Compute residual:

   [
   R = W - Q(W)
   ]

5. Save (R) as row-major FP32:

   ```python
   R = np.asarray(R, dtype=np.float32, order="C")
   R.tofile("matrix.f32")
   ```

6. Write `meta.json`.

7. Print sanity-check statistics.

---

## Avoid Downloading Full Model if Possible

If the model is stored in sharded `safetensors`, prefer reading only the shard containing the target tensor.

Possible approach:

1. Inspect `model.safetensors.index.json`.

2. Locate the shard containing:

   ```text
   model.layers.0.mlp.gate_proj.weight
   ```

3. Download only that shard if possible.

4. Use `safetensors` to load only the target tensor.

If that is too complicated, loading through `transformers` is acceptable for the first version, but document the method.

---

## Final File Format

The final file must be:

```text
matrix.f32
```

with:

```text
dtype  = float32
layout = row-major
shape  = rows x cols
```

Example:

```python
R = np.asarray(R, dtype=np.float32, order="C")
R.tofile("matrix.f32")
```

Do not silently transpose.

The shape must match what will be passed to the CUDA program with:

```bash
--m <rows>
--n <cols>
```

---

## meta.json

Create:

```text
meta.json
```

Example:

```json
{
  "name": "loftq_residual_mistral7b_layer0_gate_proj",
  "rows": 14336,
  "cols": 4096,
  "dtype": "float32",
  "layout": "row_major",
  "file": "matrix.f32",
  "debug_file": "matrix_debug.f32",
  "source_model": "Mistral-7B or chosen model",
  "source_tensor": "model.layers.0.mlp.gate_proj.weight",
  "matrix_meaning": "R = W - dequantize(quantize(W)), quantization residual of one LLM weight matrix",
  "quantization": "describe exact LoftQ quantizer or fallback row-wise/group-wise symmetric 4-bit quantization",
  "preprocessing": "load W, quantize/dequantize to Q(W), compute residual R",
  "notes": "LoftQ-inspired residual SVD benchmark; not a full LoftQ reproduction"
}
```

---

## Sanity Checks

After creating `matrix.f32`, run:

```python
import json
import numpy as np

with open("meta.json") as f:
    meta = json.load(f)

m = meta["rows"]
n = meta["cols"]

R = np.fromfile("matrix.f32", dtype=np.float32).reshape(m, n)

expected_bytes = m * n * np.dtype(np.float32).itemsize
actual_bytes = R.nbytes

print("shape:", R.shape)
print("dtype:", R.dtype)
print("mean:", float(R.mean()))
print("std:", float(R.std()))
print("mean abs:", float(np.mean(np.abs(R))))
print("max abs:", float(np.max(np.abs(R))))
print("has NaN:", bool(np.isnan(R).any()))
print("has Inf:", bool(np.isinf(R).any()))
print("file size bytes:", actual_bytes)
print("expected bytes:", expected_bytes)
assert actual_bytes == expected_bytes
```

Expected:

```text
shape matches meta.json
dtype = float32
has NaN = False
has Inf = False
```

Also report:

[
\frac{|R|_F}{|W|_F}
]

This measures how large the quantization residual is compared with the original weight.

If possible, also report:

[
\frac{|W - Q(W)|_F}{|W|_F}
]

which is the same residual ratio.

---

## CUDA Smoke Test

After producing `matrix.f32`, run a small smoke test.

If using a debug model / debug tensor, run the smoke test against `matrix_debug.f32` and pass the corresponding `--m` and `--n`.

Example for Mistral-like shape:

```bash
srun --mpi=pmix "$BIN" \
    --m 14336 \
    --n 4096 \
    --k 250 \
    --oversample 6 \
    --ngpus 16 \
    --gpus-per-rank 8 \
    --input-file testcases_real_world/LoftQ/matrix.f32 \
    --compress-b-mode none \
    --compress-b-bits 0 \
    --compress-subspace-mode none \
    --compress-subspace-bits 0 \
    --subspace-iter 1 \
    --skip-form-u \
    --summary-only \
    --repeat 5
```

If this works, run the full experiments.

---

## Formal Experiment Settings

Run three groups:

### 1. No TQ

```text
--compress-b-mode none
--compress-b-bits 0
--compress-subspace-mode none
--compress-subspace-bits 0
```

### 2. TQ 8-bit

```text
--compress-b-mode tq
--compress-b-bits 8
--compress-subspace-mode tq
--compress-subspace-bits 8
```

### 3. TQ 4-bit

```text
--compress-b-mode tq
--compress-b-bits 4
--compress-subspace-mode tq
--compress-subspace-bits 4
```

Suggested common parameters:

```text
k = 250
oversample = 6
l = 256
subspace_iter = 1
repeat = 50
input_file = testcases_real_world/LoftQ/matrix.f32
```

Do not add `--device-random-input`. Currently `--input-file` and `--device-random-input` are mutually exclusive. In file mode, matrix A is fixed from the file, while randomized SVD may regenerate Omega across repeats.

Each group should be split into two phases:

1. Timing phase: add `--no-check-error`; record only runtime / communication metrics.
2. Final-error phase: do not add `--no-check-error`; record only accuracy metrics.

The reason is that v4 prints timing and final-error summaries separately. File loading and row-major to column-major conversion are setup work and should not be counted in `Total Time`.

Timing phase example:

```bash
srun --mpi=pmix "$BIN" \
    --m 14336 \
    --n 4096 \
    --k 250 \
    --oversample 6 \
    --ngpus 16 \
    --gpus-per-rank 8 \
    --input-file testcases_real_world/LoftQ/matrix.f32 \
    --compress-b-mode tq \
    --compress-b-bits 8 \
    --compress-subspace-mode tq \
    --compress-subspace-bits 8 \
    --subspace-iter 1 \
    --skip-form-u \
    --summary-only \
    --repeat 50 \
    --no-check-error
```

Final-error phase example:

```bash
srun --mpi=pmix "$BIN" \
    --m 14336 \
    --n 4096 \
    --k 250 \
    --oversample 6 \
    --ngpus 16 \
    --gpus-per-rank 8 \
    --input-file testcases_real_world/LoftQ/matrix.f32 \
    --compress-b-mode tq \
    --compress-b-bits 8 \
    --compress-subspace-mode tq \
    --compress-subspace-bits 8 \
    --subspace-iter 1 \
    --skip-form-u \
    --summary-only \
    --repeat 50
```

---

## Metrics to Record

Please collect:

```text
Total Time
GPU Compute Time
Host/Staging Time
NVLink Time
InfiniBand Time
NVLink Payload
InfiniBand Payload
Global B Relative Error
Final Reconstruction Error
Top-k singular values, if printed
```

Real data has no synthetic theoretical error, so do not use polynomial / exponential theoretical error. In `--input-file` mode, the program should report `theoretical = n/a` and `err ratio = n/a`; the main accuracy metric is raw `Final Reconstruction Error` and its difference from the no-compression baseline.

`Global B Relative Error` is a compression diagnostic. Enable it with `--check-b-error` only when studying compression damage to the intermediate B matrix. Do not enable it in the default timing run, because the extra copy/check work can affect runtime.

Main comparison:

```text
No TQ baseline
vs
TQ 8-bit
vs
TQ 4-bit
```

---

## Report Story

Suggested explanation:

> Directly applying low-rank approximation to pretrained LLM weights can be problematic because important outlier directions may be lost. Inspired by LoftQ, we instead construct the quantization residual matrix (R = W - Q(W)). Running randomized SVD on (R) corresponds to the low-rank residual approximation step used to initialize LoRA adapters for quantized models. This gives us a real AI matrix benchmark for distributed rSVD and TurboQuant communication compression.

---

## Deliverables Checklist

Please provide:

```text
.gitignore
matrix.f32
matrix_debug.f32, optional
meta.json
README.md
scripts/prepare_loftq_residual_matrix.py
exp_turboquant/{ctrl,b8-exp,b4-exp}.slurm
sanity-check output
commands used to run no TQ / TQ 8-bit / TQ 4-bit
```

`README.md` should include:

```text
1. model name
2. tensor name
3. tensor shape
4. quantization method
5. residual norm ratio ||R||_F / ||W||_F
6. final matrix shape
7. sanity-check output
8. exact commands used for experiments
```
