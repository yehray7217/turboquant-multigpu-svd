# Experiment Guideline

This guideline is for teammates and their AI assistants when creating, running, and summarizing experiments under:

```text
/work/pbr03617/turboquant-multigpu-svd/randomized_svd_baseline_v4
```

Before starting an experiment, give the AI both files:

```text
project-context/PROJECT_CONTEXT.md
project-context/EXPERIMENT_GUIDELINE.md
```

The AI should follow the structure and conventions below unless the human explicitly overrides them.

## 1. Experiment Folder Structure

Each independent experiment should live in its own folder:

```text
exp_{experiment_topic}/
├── README.md
├── ctrl.slurm
├── exp.slurm
├── output_logs/
│   ├── ctrl.out
│   └── exp.out
└── error_logs/
    ├── ctrl.err
    └── exp.err
```

If there are more than two variants, use clear names:

```text
q0_ctrl.slurm
q1_exp.slurm
q2_exp.slurm
tq4_exp.slurm
tq2_exp.slurm
```

Avoid vague names like:

```text
exp_group2.slurm
test.slurm
run.slurm
```

unless this is only a temporary scratch experiment.

## 2. Recommended Baseline Config

For current v4 experiments, use this baseline unless the experiment topic requires changing it:

```text
m = 32768
n = 8192
k = 250
oversample = 6
l = 256
ngpus = 16
gpus-per-rank = 8
seed = 1234
spectrum-decay-mode = polynomial
spectrum-decay-param = 0.6
spectrum-rank = 8192
repeat = 50
device-random-input = on
skip-form-u = on
summary-only = on
```

Important notes:

- `spectrum_rank` should be `min(m,n)` for the main synthetic spectrum experiments.
- Do not go back to `spectrum_rank = 1024` unless the experiment specifically studies effective rank.
- `repeat = 50` is the current standard.
- Timing runs skip the first cold run internally.
- Accuracy runs use all 50 randomized trials with different seeds.
- `spectrum-decay-param = 0.6` is the recommended first-pass stress-test setting.

Recommended polynomial decay sweep:

| p | Theoretical Error, k=250, rank=8192 | Why run it |
|---:|---:|---|
| 0.4 | 76.70% | Slow decay; hard case |
| 0.6 | 41.77% | Main stress-test setting |
| 0.8 | 15.27% | Medium-easy case |
| 1.0 | 4.85% | Sanity check / easy case |

`p=1.0` is a useful sanity check, but it is too easy to be the only standard config. A complete TQ experiment should ideally sweep `p = 0.4, 0.6, 0.8, 1.0`.

## 3. Standard Slurm Pattern

Every experiment usually has two `srun` phases:

1. Timing phase:
   ```bash
   --repeat 50 \
   --no-check-error
   ```

2. Final-error phase:
   ```bash
   --repeat 50
   ```

The timing phase should report only `Total Time`.

The final-error phase should report:

```text
Final Reconstruction Error:
  mean
  min
  stddev
  theoretical
  err ratio
```

`B Relative Error` is mostly a compression diagnostic and is disabled by default. Enable it only when the experiment specifically studies compression damage to the intermediate `B` matrix:

```bash
--check-b-error
```

## 4. Metrics to Record

The most important final quality metric is:

```text
err ratio = mean final reconstruction error / theoretical best rank-k error
```

This is preferred over raw final reconstruction error because different spectrum parameters have different theoretical difficulty.

Record at least:

| Metric | Source | Meaning |
|---|---|---|
| `Total Time mean` | Timing phase | Main runtime metric |
| `Total Time min` | Timing phase | Best observed warm runtime |
| `Total Time stddev` | Timing phase | Runtime stability |
| `Final Error mean` | Final-error phase | Average reconstruction error |
| `Final Error stddev` | Final-error phase | Accuracy variability across seeds |
| `Theoretical` | Final-error phase | Best possible rank-k error for synthetic spectrum |
| `Error Ratio` | Final-error phase | Main accuracy comparison metric |

If compression is enabled and detailed logs are needed, also record:

| Metric | Meaning |
|---|---|
| payload bytes / MiB | Actual transmitted payload |
| compression ratio | FP32 payload size divided by compressed payload size |
| `B Relative Error` | Optional diagnostic error caused by compressing/decompressing `B_i`; enable with `--check-b-error` |

## 5. README Format

Each experiment folder must include a short `README.md`.

Recommended format:

```markdown
# Experiment: {Short Title}

One sentence describing the purpose of the experiment.

## Config

| Parameter | Value |
|---|---:|
| m | 32768 |
| n | 8192 |
| k | 250 |
| oversample | 6 |
| spectrum | polynomial, sigma_i = i^-0.6 |
| spectrum_rank | 8192 |
| repeat | 50 |

## Results

| Output | Group | Variable | Total Time Mean (ms) | Final Error Mean | Theoretical | Error Ratio | Notes |
|---|---|---|---:|---:|---:|---:|---|
| ctrl.out | control | none | TBD | TBD | TBD | TBD |  |
| exp.out | experiment | TBD | TBD | TBD | TBD | TBD |  |

## Conclusion

Short interpretation. Say clearly whether the result is positive, neutral, or negative.
```

When filling the table:

- Use values from the summary output, not manually estimated values.
- Prefer `Error Ratio` for final accuracy interpretation.
- Keep notes concise and grounded in the actual logs.

## 6. Recommended TQ Experiment

This is the next important experiment family.

Goal:

```text
Measure whether TurboQuant improves runtime without unacceptable final reconstruction error on realistic synthetic spectra.
```

Suggested folder:

```text
exp_tq_b_reduce/
```

Fixed config:

```text
m = 32768
n = 8192
k = 250
oversample = 6
spectrum-decay-mode = polynomial
spectrum-decay-param = 0.6
spectrum-rank = 8192
subspace-iter = 1
repeat = 50
```

Compare:

| Group | `compress-b-mode` | `compress-b-bits` |
|---|---|---:|
| control | `none` | `0` |
| TQ 4-bit | `tq` | `4` |
| TQ 2-bit | `tq` | `2` |

Expected interpretation:

- TQ 4-bit is likely the best tradeoff.
- TQ 2-bit may be faster but can damage accuracy.
- With `subspace-iter = 1`, B-only TQ may not produce full speedup because the subspace iteration `Z` reduce is still uncompressed.
- Start with `p=0.6`; if time allows, sweep `p=0.4, 0.6, 0.8, 1.0`.

## 7. Recommended Subspace Iteration Settings

Current observation:

| q | Result |
|---:|---|
| 0 | Baseline; less accurate |
| 1 | Best current accuracy/cost tradeoff |
| 2 | Worse than q=1 in current FP32 + small oversample setup |

Default recommendation:

```text
--subspace-iter 1
```

Do not use `--stabilize-subspace-z` unless the experiment explicitly studies Z-side QR. The current stabilization experiment was negative.

## 8. Slurm Template

Use this as a starting point and edit only the experiment variable.

```bash
#!/bin/bash
#SBATCH --job-name=rsvd_v4_{short_name}
#SBATCH --account=ACD115064
#SBATCH --partition=nycugpu_queue
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=8
#SBATCH --time=01:00:00
#SBATCH --output=/work/pbr03617/turboquant-multigpu-svd/randomized_svd_baseline_v4/exp_{topic}/output_logs/{ctrl_or_exp}.out
#SBATCH --error=/work/pbr03617/turboquant-multigpu-svd/randomized_svd_baseline_v4/exp_{topic}/error_logs/{ctrl_or_exp}.err

set -euo pipefail

module load cuda/12.8
module load ucx/1.14.1
module load openmpi/5.0.2_ucx1.14.1_cuda12.3

unset OMPI_MCA_opal_cuda_support || true
unset OMPI_MCA_btl_smcuda_use_cuda_ipc || true

PROJECT_ROOT=/work/{username}/turboquant-multigpu-svd/randomized_svd_baseline_v4
BIN="${PROJECT_ROOT}/.build/randomized_svd_multigpu_v4"
cd "$PROJECT_ROOT"

make
cd "${PROJECT_ROOT}/exp_{topic}"

BASE_ARGS=(
    --m 32768
    --n 8192
    --k 250
    --oversample 6
    --ngpus 16
    --gpus-per-rank 8
    --seed 1234
    --spectrum-decay-mode polynomial
    --spectrum-decay-param 0.6
    --spectrum-rank 8192
    --subspace-iter 1
    --device-random-input
    --skip-form-u
    --compress-b-mode none
    --compress-b-bits 0
    --summary-only
)

echo "Starting {control/experiment}: {short description}"
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST}"
nvidia-smi -L
echo "====================================="

echo "Timing: {control/experiment}"
srun --mpi=pmix "$BIN" \
    "${BASE_ARGS[@]}" \
    --repeat 50 \
    --no-check-error
echo "-------------------------------------"

echo "Final-error: {control/experiment}"
srun --mpi=pmix "$BIN" \
    "${BASE_ARGS[@]}" \
    --repeat 50

echo "{Control/Experiment} complete!"
```

## 9. AI Prompt Template

Teammates can use this prompt with an AI assistant:

```text
Please read:
/work/{username}/turboquant-multigpu-svd/randomized_svd_baseline_v4/skills/PROJECT_CONTEXT.md
/work/{username}/turboquant-multigpu-svd/randomized_svd_baseline_v4/skills/EXPERIMENT_GUIDELINE.md

Create a new experiment folder under:
/work/{username}/turboquant-multigpu-svd/randomized_svd_baseline_v4

Experiment topic:
{describe topic}

Control variable:
{fixed config}

Manipulated variable:
{what changes between control and experiment}

Please create:
- README.md
- ctrl_group.slurm
- exp_group.slurm
- output_logs/
- error_logs/

Do not submit the Slurm jobs. I will submit them manually.
```

After jobs finish:

```text
Please parse:
{path to ctrl.out}
{path to exp.out}

Update the experiment README with:
- config
- Total Time mean/min/stddev
- Final Reconstruction Error mean/min/stddev
- theoretical
- error ratio
- concise conclusion
```

## 10. Common Pitfalls

- Do not compare raw final error across different spectrum parameters without also comparing `Error Ratio`.
- Do not use `spectrum_rank = 1024` for main experiments unless intentionally studying low effective rank.
- Do not use only `p=1.0` as the main TQ result; include at least `p=0.6`, and ideally sweep `0.4/0.6/0.8/1.0`.
- Do not trust Slurm `echo` text alone; always confirm actual parameters from the program banner in the output log.
- Do not enable `B Relative Error` by default; use `--check-b-error` only for compression diagnostics.
- Do not interpret q=2 as automatically better than q=1. Current experiments show q=1 is better.
- Do not use `--stabilize-subspace-z` by default.
- Do not submit Slurm jobs from AI unless the human explicitly asks for it.

## 11. Checklist Before Handing Back to Human

Before saying the experiment setup is ready, the AI should verify:

- `README.md` exists and describes the purpose.
- Slurm output/error paths point to the correct experiment folder.
- `bash -n *.slurm` passes.
- The program path is `.build/randomized_svd_multigpu_v4`.
- `repeat = 50`.
- `spectrum_rank = 8192` for the standard 32k x 8k setup.
- `spectrum-decay-param = 0.6` for the first-pass TQ stress test, unless intentionally sweeping p.
- The manipulated variable is the only intended difference between control and experiment.
- The AI did not submit any Slurm job unless explicitly instructed.
