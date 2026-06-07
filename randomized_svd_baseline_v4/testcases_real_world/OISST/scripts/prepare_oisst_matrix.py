#!/usr/bin/env python3

import argparse
import datetime as dt
import json
import time
from pathlib import Path
from typing import List
from urllib.error import URLError
from urllib.request import urlopen

try:
    import numpy as np
except ImportError:
    np = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare an OISST SST anomaly matrix in raw row-major FP32 format."
    )
    parser.add_argument("--rows", type=int, default=32768, help="Number of ocean grid points.")
    parser.add_argument("--cols", type=int, default=8192, help="Number of daily snapshots.")
    parser.add_argument("--start-year", type=int, default=2000, help="First year to query.")
    parser.add_argument("--end-year", type=int, default=2023, help="Last year to query, inclusive.")
    parser.add_argument("--start-date", type=str, default="2000-01-01", help="First day to keep.")
    parser.add_argument("--end-date", type=str, default=None, help="Optional final day to keep.")
    parser.add_argument("--seed", type=int, default=1234, help="Sampling seed.")
    parser.add_argument("--batch-size", type=int, default=2048, help="Selected points fetched per OPeNDAP batch.")
    parser.add_argument("--retries", type=int, default=3, help="Retries per yearly OPeNDAP request.")
    parser.add_argument("--raw-dir", type=Path, default=Path("raw_data"), help="Directory for cached daily .nc files.")
    parser.add_argument(
        "--download-missing",
        action="store_true",
        help="Download missing daily OISST files from the NOAA NCEI direct-download root before processing.",
    )
    parser.add_argument(
        "--local-only",
        action="store_true",
        help="Use only local cached .nc files and fail if any required day is missing.",
    )
    parser.add_argument("--output", type=Path, default=Path("matrix.f32"), help="Output .f32 path.")
    parser.add_argument("--meta", type=Path, default=Path("meta.json"), help="Output metadata path.")
    return parser.parse_args()


def require_numpy():
    if np is None:
        raise SystemExit("numpy is required. Install it with `pip install numpy`.")
    return np


def build_url(year: int) -> str:
    return "https://psl.noaa.gov/thredds/dodsC/Datasets/noaa.oisst.v2.highres/sst.day.mean.{0}.nc".format(year)


def build_daily_filename(date_obj: dt.date) -> str:
    return "oisst-avhrr-v02r01.{0}.nc".format(date_obj.strftime("%Y%m%d"))


def build_daily_download_url(date_obj: dt.date) -> str:
    yyyymm = date_obj.strftime("%Y%m")
    return (
        "https://www.ncei.noaa.gov/data/sea-surface-temperature-optimum-interpolation/v2.1/access/avhrr/{0}/{1}".format(
            yyyymm, build_daily_filename(date_obj)
        )
    )


def parse_date(date_str: str) -> dt.date:
    return dt.datetime.strptime(date_str, "%Y-%m-%d").date()


def resolve_date_window(args: argparse.Namespace):
    start_date = parse_date(args.start_date)
    if args.end_date:
        end_date = parse_date(args.end_date)
    else:
        end_date = start_date + dt.timedelta(days=args.cols - 1)

    if end_date < start_date:
        raise SystemExit("end_date must be on or after start_date.")

    first_year = max(args.start_year, start_date.year)
    last_year = min(args.end_year, end_date.year)
    if last_year < first_year:
        raise SystemExit("Requested date window falls outside the configured year range.")

    return start_date, end_date, list(range(first_year, last_year + 1))


def iter_dates(start_date: dt.date, end_date: dt.date):
    current = start_date
    while current <= end_date:
        yield current
        current += dt.timedelta(days=1)


def daily_local_path(raw_dir: Path, date_obj: dt.date) -> Path:
    return raw_dir / date_obj.strftime("%Y%m") / build_daily_filename(date_obj)


def download_daily_file(date_obj: dt.date, destination: Path, retries: int) -> None:
    url = build_daily_download_url(date_obj)
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = destination.with_suffix(destination.suffix + ".part")
    last_error = None

    for attempt in range(1, retries + 1):
        try:
            with urlopen(url, timeout=120) as response:
                data = response.read()
            if not data:
                raise RuntimeError("empty response body")
            with tmp_path.open("wb") as handle:
                handle.write(data)
            tmp_path.replace(destination)
            return
        except Exception as exc:
            last_error = exc
            if tmp_path.exists():
                tmp_path.unlink()
            if attempt < retries:
                print(
                    "Warning: failed to download {0} on attempt {1}/{2}: {3}".format(
                        destination.name, attempt, retries, exc
                    )
                )
                time.sleep(min(5, attempt))

    raise SystemExit("Failed to download {0}: {1}".format(url, last_error))


def ensure_local_daily_files(
    start_date: dt.date,
    end_date: dt.date,
    raw_dir: Path,
    retries: int,
    download_missing: bool,
):
    paths = []
    missing = []
    for date_obj in iter_dates(start_date, end_date):
        local_path = daily_local_path(raw_dir, date_obj)
        paths.append(local_path)
        if not local_path.exists():
            missing.append((date_obj, local_path))

    if missing and not download_missing:
        raise SystemExit(
            "Missing {0} daily OISST files under {1}. Re-run with --download-missing.".format(
                len(missing), raw_dir
            )
        )

    for date_obj, local_path in missing:
        print("Downloading {0} -> {1}".format(date_obj.isoformat(), local_path))
        download_daily_file(date_obj, local_path, retries)

    return paths


def open_dataset_for_year(year: int, retries: int):
    try:
        import xarray as xr
    except ImportError as exc:
        raise SystemExit("xarray is required. Install it with `pip install xarray netCDF4 dask`.") from exc

    url = build_url(year)
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            ds = xr.open_dataset(url, engine="netcdf4", chunks=None, cache=False)
            if "sst" not in ds:
                ds.close()
                raise RuntimeError("Expected variable `sst` in NOAA OISST dataset.")
            return ds
        except Exception as exc:
            last_error = exc
            if attempt < retries:
                print(
                    "Warning: failed to open year {0} from OPeNDAP on attempt {1}/{2}: {3}".format(
                        year, attempt, retries, exc
                    )
                )
                time.sleep(min(5, attempt))
    raise SystemExit("Failed to open OISST year {0}: {1}".format(year, last_error))


def sample_valid_points(first_year: int, start_date: dt.date, rows: int, seed: int, retries: int):
    np_mod = require_numpy()

    ds = open_dataset_for_year(first_year, retries)
    if "sst" not in ds:
        ds.close()
        raise SystemExit("Expected variable `sst` in NOAA OISST dataset.")

    first_day = ds["sst"].sel(time=slice(start_date.isoformat(), start_date.isoformat()))
    if int(first_day.sizes.get("time", 0)) == 0:
        ds.close()
        raise SystemExit("Could not find start date {0} in OISST year {1}.".format(start_date, first_year))

    stacked = first_day.isel(time=0).stack(point=("lat", "lon"))
    first_snapshot = stacked.load().values
    valid_point_indices = np_mod.flatnonzero(np_mod.isfinite(first_snapshot))
    if valid_point_indices.size < rows:
        ds.close()
        raise SystemExit(
            f"Requested {rows} ocean points, but only found {valid_point_indices.size} valid points."
        )

    rng = np_mod.random.default_rng(seed)
    shuffled_indices = rng.permutation(valid_point_indices)
    point_indices = np_mod.sort(shuffled_indices[:rows])
    latitudes = np_mod.asarray(stacked["lat"].values)[point_indices]
    longitudes = np_mod.asarray(stacked["lon"].values)[point_indices]
    ds.close()
    return point_indices, latitudes, longitudes


def load_time_series_for_points(
    years: List[int],
    start_date: dt.date,
    end_date: dt.date,
    point_indices,
    batch_size: int,
    retries: int,
):
    import xarray as xr

    np_mod = require_numpy()
    point_count = int(point_indices.size)
    yearly_blocks = []
    selected_start = None
    selected_end = None

    for year in years:
        year_start = max(start_date, dt.date(year, 1, 1))
        year_end = min(end_date, dt.date(year, 12, 31))
        ds = open_dataset_for_year(year, retries)
        if "sst" not in ds:
            ds.close()
            raise SystemExit("Expected variable `sst` in OISST year {0}.".format(year))

        sst = ds["sst"].sel(time=slice(year_start.isoformat(), year_end.isoformat()))
        if int(sst.sizes.get("time", 0)) == 0:
            ds.close()
            continue

        stacked = sst.stack(point=("lat", "lon")).transpose("point", "time")
        batch_blocks = []
        for start in range(0, point_count, batch_size):
            stop = min(start + batch_size, point_count)
            batch = point_indices[start:stop]
            batch_da = stacked.isel(point=xr.DataArray(batch, dims="point")).load()
            batch_values = np_mod.asarray(batch_da.values, dtype=np_mod.float32)
            if batch_values.shape[0] != (stop - start):
                ds.close()
                raise SystemExit("Unexpected batch shape while loading year {0}.".format(year))
            if not np_mod.isfinite(batch_values).all():
                ds.close()
                raise SystemExit(
                    "Encountered NaN or Inf in selected OISST points for year {0}. Try a different seed.".format(year)
                )
            batch_blocks.append(batch_values)

        year_block = np_mod.concatenate(batch_blocks, axis=0)
        yearly_blocks.append(year_block)
        if selected_start is None:
            selected_start = str(np_mod.asarray(sst["time"].values[0], dtype="datetime64[D]"))
        selected_end = str(np_mod.asarray(sst["time"].values[-1], dtype="datetime64[D]"))
        ds.close()

    if not yearly_blocks:
        raise SystemExit("No OISST data loaded for the requested date window.")

    matrix = np_mod.concatenate(yearly_blocks, axis=1)
    return matrix, selected_start, selected_end


def open_local_daily_sst(nc_path: Path):
    try:
        import xarray as xr
    except ImportError as exc:
        raise SystemExit("xarray is required. Install it with `pip install xarray netCDF4 dask`.") from exc

    ds = xr.open_dataset(str(nc_path), engine="netcdf4", chunks=None, cache=False)
    if "sst" not in ds:
        ds.close()
        raise SystemExit("Expected variable `sst` in {0}.".format(nc_path))
    sst = ds["sst"].squeeze(drop=True)
    return ds, sst


def sample_valid_points_from_local(first_path: Path, rows: int, seed: int):
    np_mod = require_numpy()

    ds, sst = open_local_daily_sst(first_path)
    stacked = sst.stack(point=("lat", "lon"))
    first_snapshot = stacked.load().values
    valid_point_indices = np_mod.flatnonzero(np_mod.isfinite(first_snapshot))
    if valid_point_indices.size < rows:
        ds.close()
        raise SystemExit(
            "Requested {0} ocean points, but only found {1} valid points in {2}.".format(
                rows, valid_point_indices.size, first_path
            )
        )

    rng = np_mod.random.default_rng(seed)
    shuffled_indices = rng.permutation(valid_point_indices)
    point_indices = np_mod.sort(shuffled_indices[:rows])
    latitudes = np_mod.asarray(stacked["lat"].values)[point_indices]
    longitudes = np_mod.asarray(stacked["lon"].values)[point_indices]
    ds.close()
    return point_indices, latitudes, longitudes


def load_local_time_series_for_points(local_paths, point_indices, batch_size: int):
    np_mod = require_numpy()
    point_count = int(point_indices.size)
    columns = []
    selected_start = None
    selected_end = None

    for nc_path in local_paths:
        ds, sst = open_local_daily_sst(nc_path)
        stacked = sst.stack(point=("lat", "lon"))
        column_batches = []
        for start in range(0, point_count, batch_size):
            stop = min(start + batch_size, point_count)
            batch_indices = point_indices[start:stop]
            batch_values = np_mod.asarray(stacked.isel(point=batch_indices).load().values, dtype=np_mod.float32)
            if batch_values.ndim != 1:
                ds.close()
                raise SystemExit("Expected 1D daily SST slice in {0}, got shape {1}.".format(nc_path, batch_values.shape))
            if not np_mod.isfinite(batch_values).all():
                ds.close()
                raise SystemExit("Encountered NaN or Inf in selected OISST points for {0}.".format(nc_path))
            column_batches.append(batch_values)

        column = np_mod.concatenate(column_batches, axis=0)
        if column.shape[0] != point_count:
            ds.close()
            raise SystemExit("Unexpected point count while reading {0}.".format(nc_path))
        columns.append(column)

        file_date = nc_path.stem.split(".")[-1]
        selected_start = selected_start or "{0}-{1}-{2}".format(file_date[0:4], file_date[4:6], file_date[6:8])
        selected_end = "{0}-{1}-{2}".format(file_date[0:4], file_date[4:6], file_date[6:8])
        ds.close()

    if not columns:
        raise SystemExit("No local OISST files were loaded.")

    matrix = np_mod.stack(columns, axis=1)
    return matrix, selected_start, selected_end


def temporal_mean_center(matrix):
    np_mod = require_numpy()
    row_means = matrix.mean(axis=1, dtype=np.float64, keepdims=True)
    centered = matrix - row_means.astype(np_mod.float32)
    return np_mod.asarray(centered, dtype=np_mod.float32, order="C")


def write_outputs(
    matrix,
    output_path: Path,
    meta_path: Path,
    args: argparse.Namespace,
    start_date: str,
    end_date: str,
    latitudes,
    longitudes,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.parent.mkdir(parents=True, exist_ok=True)

    matrix.tofile(output_path)
    meta = {
        "name": "OISST_SST_anomaly",
        "rows": int(matrix.shape[0]),
        "cols": int(matrix.shape[1]),
        "dtype": "float32",
        "layout": "row_major",
        "file": output_path.name,
        "source": "NOAA OISST daily sea surface temperature via PSL OPeNDAP",
        "matrix_meaning": "rows are selected ocean grid points, columns are daily time snapshots",
        "preprocessing": "temporal mean removed per grid point",
        "nan_policy": "sample only grid points with finite values across the selected time window",
        "selection": f"global valid ocean points sampled with seed {args.seed}",
        "selection_seed": args.seed,
        "start_date": start_date,
        "end_date": end_date,
        "number_of_days": int(matrix.shape[1]),
        "number_of_points": int(matrix.shape[0]),
        "url_years": [args.start_year, args.end_year],
        "latitude_range": [float(latitudes.min()), float(latitudes.max())],
        "longitude_range": [float(longitudes.min()), float(longitudes.max())],
        "notes": "SVD corresponds to EOF/PCA analysis of the SST anomaly field.",
    }
    with meta_path.open("w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)
        handle.write("\n")


def print_sanity_stats(matrix, output_path: Path) -> None:
    np_mod = require_numpy()
    expected_bytes = matrix.shape[0] * matrix.shape[1] * np_mod.dtype(np_mod.float32).itemsize
    actual_bytes = output_path.stat().st_size
    row_means = matrix.mean(axis=1)

    print(f"shape: {matrix.shape}")
    print(f"dtype: {matrix.dtype}")
    print(f"mean abs: {float(np_mod.mean(np_mod.abs(matrix))):.8f}")
    print(f"max abs: {float(np_mod.max(np_mod.abs(matrix))):.8f}")
    print(f"has NaN: {bool(np_mod.isnan(matrix).any())}")
    print(f"has Inf: {bool(np_mod.isinf(matrix).any())}")
    print(f"mean of row means: {float(np_mod.mean(row_means)):.8e}")
    print(f"max abs row mean: {float(np_mod.max(np_mod.abs(row_means))):.8e}")
    print(f"file size bytes: {actual_bytes}")
    print(f"expected bytes: {expected_bytes}")
    if actual_bytes != expected_bytes:
        raise SystemExit(f"File size mismatch for {output_path}.")


def main() -> None:
    args = parse_args()
    np_mod = require_numpy()
    start_date_obj, end_date_obj, years = resolve_date_window(args)

    use_local_mode = args.local_only or args.download_missing or args.raw_dir.exists()
    if use_local_mode:
        local_paths = ensure_local_daily_files(
            start_date=start_date_obj,
            end_date=end_date_obj,
            raw_dir=args.raw_dir,
            retries=args.retries,
            download_missing=args.download_missing,
        )
        point_indices, latitudes, longitudes = sample_valid_points_from_local(
            first_path=local_paths[0],
            rows=args.rows,
            seed=args.seed,
        )
        matrix, start_date, end_date = load_local_time_series_for_points(
            local_paths=local_paths,
            point_indices=point_indices,
            batch_size=args.batch_size,
        )
    else:
        point_indices, latitudes, longitudes = sample_valid_points(
            first_year=years[0],
            start_date=start_date_obj,
            rows=args.rows,
            seed=args.seed,
            retries=args.retries,
        )
        matrix, start_date, end_date = load_time_series_for_points(
            years=years,
            start_date=start_date_obj,
            end_date=end_date_obj,
            point_indices=point_indices,
            batch_size=args.batch_size,
            retries=args.retries,
        )
    if matrix.shape[1] < args.cols:
        raise SystemExit(
            "Requested {0} days, but only loaded {1} days. Adjust --start-date/--end-date or year range.".format(
                args.cols, matrix.shape[1]
            )
        )
    matrix = np_mod.asarray(matrix[:, : args.cols], dtype=np_mod.float32, order="C")
    matrix = temporal_mean_center(matrix)

    if np_mod.isnan(matrix).any() or np_mod.isinf(matrix).any():
        raise SystemExit("Output matrix contains NaN or Inf after preprocessing.")

    write_outputs(
        matrix=matrix,
        output_path=args.output,
        meta_path=args.meta,
        args=args,
        start_date=start_date,
        end_date=end_date,
        latitudes=latitudes,
        longitudes=longitudes,
    )
    print_sanity_stats(matrix, args.output)


if __name__ == "__main__":
    main()
