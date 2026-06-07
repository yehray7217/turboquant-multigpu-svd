# OISST Notes

## Python Environment

The OISST preprocessing workflow needs a Python virtual environment.

Recommended setup:

```bash
cd randomized_svd_baseline_v4/testcases_real_world/OISST

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install numpy xarray netCDF4 dask toolz partd fsspec
```

If the environment already exists:

```bash
cd randomized_svd_baseline_v4/testcases_real_world/OISST
source .venv/bin/activate
```

## Preprocessing

Debug matrix example:

```bash
python scripts/prepare_oisst_matrix.py \
    --rows 8192 \
    --cols 4096 \
    --start-date 2000-01-01 \
    --download-missing \
    --raw-dir raw_data \
    --batch-size 2048 \
    --retries 5 \
    --output matrix_debug.f32 \
    --meta meta_debug.json
```

## Smoke Test

Submit the debug smoke test with Slurm:

```bash
cd exp_turboquant
sbatch smoke-debug.slurm
```

Check logs:

```bash
cat output_logs/smoke-debug.out
cat error_logs/smoke-debug.err
```
