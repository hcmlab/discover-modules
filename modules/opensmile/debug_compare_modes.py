#!/usr/bin/env python3
"""Benchmark and compare openSMILE extraction modes on a single audio file.

Modes:
1) dataset_iterator chunked processing with process_signal()
2) direct file-level chunking with process_files() (single worker)
3) direct file-level chunking with process_files() (multi-worker, repeated file entries)
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import os
import sys
import time
import wave
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from discover_utils.data.provider.dataset_iterator import DatasetIterator

INPUT_ID = "input_audio"
OUTPUT_ID = "output_stream"

# Avoid shadowing by local 'opensmile.py' in this directory.
def import_opensmile_package():
    script_dir = str(Path(__file__).resolve().parent)
    old_path = list(sys.path)
    try:
        sys.path = [p for p in sys.path if Path(p).resolve().as_posix() != Path(script_dir).as_posix()]
        if "opensmile" in sys.modules:
            del sys.modules["opensmile"]
        mod = importlib.import_module("opensmile")
    finally:
        sys.path = old_path

    if not hasattr(mod, "FeatureSet"):
        raise RuntimeError("Imported 'opensmile' is not the opensmile package (missing FeatureSet).")
    return mod


OSMILE = import_opensmile_package()
warnings.filterwarnings(
    "ignore",
    message=r"Segment too short, filling with NaN\.",
    category=UserWarning,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", type=Path, required=True, help="Path to wav file")
    parser.add_argument("--feature-set", default="eGeMAPSv02", help="opensmile.FeatureSet member name")
    parser.add_argument("--feature-lvl", default="Functionals", help="opensmile.FeatureLevel member name")
    parser.add_argument("--win-ms", type=float, default=40.0, help="Window length in ms")
    parser.add_argument("--hop-ms", type=float, default=40.0, help="Hop length in ms")
    parser.add_argument("--workers", type=int, default=4, help="Workers for multi-worker mode")
    parser.add_argument("--inspect-workers", action="store_true", help="Print effective worker config for multiple Smile init variants")
    parser.add_argument("--benchmark-worker-variants", action="store_true", help="Time process_files() for each worker variant")
    parser.add_argument("--sweep-start-ms", type=float, default=None, help="If set, sweep window/hop from this value upward until NaNs disappear")
    parser.add_argument("--sweep-step-ms", type=float, default=100.0, help="Sweep step size in ms")
    parser.add_argument("--sweep-max-ms", type=float, default=5000.0, help="Sweep max window/hop in ms")
    return parser.parse_args()


def wav_duration_seconds(path: Path) -> float:
    with wave.open(str(path), "rb") as wf:
        frames = wf.getnframes()
        sr = wf.getframerate()
    return frames / sr


def mk_smile(
    feature_set: str,
    feature_lvl: str,
    num_workers: int | None = None,
    multiprocessing: bool | None = None,
):
    fs = OSMILE.FeatureSet.__members__[feature_set]
    fl = OSMILE.FeatureLevel.__members__[feature_lvl]
    kwargs = {"feature_set": fs, "feature_level": fl}
    sig = inspect.signature(OSMILE.Smile.__init__)
    if num_workers is not None and "num_workers" in sig.parameters:
        kwargs["num_workers"] = int(num_workers)
    if multiprocessing is not None and "multiprocessing" in sig.parameters:
        kwargs["multiprocessing"] = bool(multiprocessing)
    return OSMILE.Smile(**kwargs)


def _resolve_worker_count(smile) -> int | None:
    candidates = [
        ("num_workers",),
        ("process", "num_workers"),
        ("_process", "num_workers"),
        ("process", "process", "num_workers"),
    ]
    for chain in candidates:
        obj = smile
        ok = True
        for attr in chain:
            if not hasattr(obj, attr):
                ok = False
                break
            obj = getattr(obj, attr)
        if ok and isinstance(obj, int):
            return obj
    return None


def inspect_worker_variants(feature_set: str, feature_lvl: str, requested_workers: int) -> None:
    sig = inspect.signature(OSMILE.Smile.__init__)
    supports_num_workers = "num_workers" in sig.parameters
    supports_multiprocessing = "multiprocessing" in sig.parameters
    print(
        f"[WORKERS] cpu_count={os.cpu_count()} "
        f"supports_num_workers={supports_num_workers} "
        f"supports_multiprocessing={supports_multiprocessing}"
    )

    variants = [
        ("default", None, None),
        ("multiprocessing_true_default_workers", None, True),
        ("multiprocessing_false_default_workers", None, False),
        ("multiprocessing_true_workers_cli", requested_workers, True),
        ("multiprocessing_true_workers_cpu_count", os.cpu_count() or requested_workers, True),
        ("multiprocessing_false_workers_cli", requested_workers, False),
    ]
    for name, workers, mp in variants:
        smile = mk_smile(
            feature_set=feature_set,
            feature_lvl=feature_lvl,
            num_workers=workers,
            multiprocessing=mp,
        )
        effective = _resolve_worker_count(smile)
        print(
            f"[WORKERS] variant={name} requested_workers={workers} "
            f"requested_multiprocessing={mp} effective_workers={effective if effective is not None else 'unknown'}"
        )


def benchmark_worker_variants(
    audio_path: Path,
    feature_set: str,
    feature_lvl: str,
    starts: np.ndarray,
    ends: np.ndarray,
    requested_workers: int,
) -> None:
    variants = [
        ("default", None, None),
        ("multiprocessing_true_default_workers", None, True),
        ("multiprocessing_false_default_workers", None, False),
        ("multiprocessing_true_workers_cli", requested_workers, True),
        ("multiprocessing_true_workers_cpu_count", os.cpu_count() or requested_workers, True),
        ("multiprocessing_false_workers_cli", requested_workers, False),
    ]
    files = [str(audio_path)] * len(starts)
    print(f"[BENCH] variants={len(variants)} segments={len(starts)}")
    for name, workers, mp in variants:
        smile = mk_smile(
            feature_set=feature_set,
            feature_lvl=feature_lvl,
            num_workers=workers,
            multiprocessing=mp,
        )
        effective = _resolve_worker_count(smile)
        t0 = time.perf_counter()
        out = smile.process_files(
            files,
            starts=starts.tolist(),
            ends=ends.tolist(),
        )
        dt = time.perf_counter() - t0
        out = out.reset_index(drop=True)
        nr = nan_ratio(out)
        print(
            f"[BENCH] variant={name} requested_workers={workers} requested_multiprocessing={mp} "
            f"effective_workers={effective if effective is not None else 'unknown'} "
            f"time={dt:.3f}s rows={len(out)} nan_ratio={nr:.6f}"
        )

    # Experimental variant: keep len(files)=cpu_count and split starts/ends per file.
    # This only works if the installed opensmile/audinterface version supports
    # nested starts/ends (one list of segments per file).
    cpu_n = os.cpu_count() or requested_workers
    starts_split = [x.tolist() for x in np.array_split(starts, cpu_n) if len(x) > 0]
    ends_split = [x.tolist() for x in np.array_split(ends, cpu_n) if len(x) > 0]
    files_split = [str(audio_path)] * len(starts_split)
    smile = mk_smile(
        feature_set=feature_set,
        feature_lvl=feature_lvl,
        num_workers=requested_workers,
        multiprocessing=True,
    )
    effective = _resolve_worker_count(smile)
    try:
        t0 = time.perf_counter()
        out = smile.process_files(
            files_split,
            starts=starts_split,
            ends=ends_split,
        )
        dt = time.perf_counter() - t0
        out = out.reset_index(drop=True)
        nr = nan_ratio(out)
        print(
            f"[BENCH] variant=split_by_cpu_files requested_workers={requested_workers} "
            f"requested_multiprocessing=True effective_workers={effective if effective is not None else 'unknown'} "
            f"len_files={len(files_split)} time={dt:.3f}s rows={len(out)} nan_ratio={nr:.6f}"
        )
    except Exception as exc:
        print(
            f"[BENCH] variant=split_by_cpu_files len_files={len(files_split)} unsupported: {type(exc).__name__}: {exc}"
        )


def make_segments(duration_s: float, win_s: float, hop_s: float) -> tuple[np.ndarray, np.ndarray]:
    # Drop tail shorter than win_s to avoid short-segment NaNs from incomplete windows.
    starts = np.arange(0.0, duration_s - win_s + 1e-12, hop_s, dtype=np.float64)
    ends = starts + win_s
    return starts, ends


def run_dataset_iterator_mode(
    audio_path: Path,
    feature_set: str,
    feature_lvl: str,
    win_ms: float,
    hop_ms: float,
) -> tuple[pd.DataFrame, float]:
    smile = mk_smile(feature_set=feature_set, feature_lvl=feature_lvl)
    input_audio = {"src": "file:stream:audio", "type": "input", "id": INPUT_ID, "uri": str(audio_path)}
    output_audio = {"src": "file:stream:SSIStream:feature", "type": "output", "id": OUTPUT_ID, "uri": str(audio_path.parent / "tmp.debug.stream")}
    # DatasetIterator interprets floats as seconds and ints as milliseconds.
    ds = DatasetIterator(
        data_description=[input_audio, output_audio],
        frame_size=f"{int(round(win_ms))}ms",
        stride=f"{int(round(hop_ms))}ms",
    )
    ds.load()

    t0 = time.perf_counter()
    out = []
    for sample in ds:
        sig = np.asarray(sample[INPUT_ID])
        sr = int(ds.current_session.input_data[INPUT_ID].meta_data.sample_rate)
        out.append(smile.process_signal(sig, sr))
    dt = time.perf_counter() - t0

    if not out:
        return pd.DataFrame(columns=smile.feature_names), dt
    return pd.concat(out, ignore_index=True), dt


def run_file_segments_single_worker(
    audio_path: Path,
    feature_set: str,
    feature_lvl: str,
    starts: np.ndarray,
    ends: np.ndarray,
) -> tuple[pd.DataFrame, float]:
    smile = mk_smile(
        feature_set=feature_set,
        feature_lvl=feature_lvl,
        num_workers=1,
        multiprocessing=False,
    )
    t0 = time.perf_counter()
    files = [str(audio_path)] * len(starts)
    out = smile.process_files(
        files,
        starts=starts.tolist(),
        ends=ends.tolist(),
    )
    dt = time.perf_counter() - t0
    return out.reset_index(drop=True), dt


def run_file_segments_multi_worker(
    audio_path: Path,
    feature_set: str,
    feature_lvl: str,
    starts: np.ndarray,
    ends: np.ndarray,
    workers: int,
) -> tuple[pd.DataFrame, float]:
    smile = mk_smile(
        feature_set=feature_set,
        feature_lvl=feature_lvl,
        num_workers=workers,
        multiprocessing=True,
    )
    files = [str(audio_path)] * len(starts)
    t0 = time.perf_counter()
    out = smile.process_files(
        files,
        starts=starts.tolist(),
        ends=ends.tolist(),
    )
    dt = time.perf_counter() - t0

    out = out.reset_index()
    if "start" in out.columns and "end" in out.columns:
        out = out.sort_values(["start", "end"], kind="mergesort")
    out = out.drop(columns=[c for c in ("file", "start", "end") if c in out.columns], errors="ignore")
    return out.reset_index(drop=True), dt


def compare_frames(a: pd.DataFrame, b: pd.DataFrame, label_a: str, label_b: str) -> None:
    summary = compare_frames_summary(a, b)
    if summary is None:
        print(f"[COMPARE] {label_a} vs {label_b}: no overlap")
        return
    rows, cols, mean_abs, max_abs = summary
    if mean_abs is None:
        print(f"[COMPARE] {label_a} vs {label_b}: rows={rows}, cols={cols}, all values are NaN/Inf")
        return
    print(
        f"[COMPARE] {label_a} vs {label_b}: "
        f"rows={rows}, cols={cols}, "
        f"mean_abs_diff={mean_abs:.6g}, "
        f"max_abs_diff={max_abs:.6g}"
    )


def compare_frames_summary(a: pd.DataFrame, b: pd.DataFrame) -> tuple[int, int, float | None, float | None] | None:
    common_cols = [c for c in a.columns if c in b.columns]
    rows = min(len(a), len(b))
    if rows == 0 or not common_cols:
        return None

    xa = a.loc[: rows - 1, common_cols].to_numpy(dtype=np.float64, copy=False)
    xb = b.loc[: rows - 1, common_cols].to_numpy(dtype=np.float64, copy=False)
    diff = np.abs(xa - xb)
    finite = np.isfinite(diff)
    if not np.any(finite):
        return rows, len(common_cols), None, None
    return rows, len(common_cols), float(np.mean(diff[finite])), float(np.max(diff[finite]))


def describe(name: str, frame: pd.DataFrame, seconds: float) -> None:
    arr = frame.to_numpy(dtype=np.float64, copy=False) if len(frame) else np.array([], dtype=np.float64)
    nan_ratio = float(np.isnan(arr).sum() / arr.size) if arr.size else 0.0
    print(
        f"[{name}] rows={len(frame)}, cols={len(frame.columns)}, "
        f"time={seconds:.3f}s, nan_ratio={nan_ratio:.6f}"
    )


def nan_ratio(frame: pd.DataFrame) -> float:
    arr = frame.to_numpy(dtype=np.float64, copy=False) if len(frame) else np.array([], dtype=np.float64)
    if arr.size == 0:
        return 0.0
    return float(np.isnan(arr).sum() / arr.size)


def main() -> None:
    args = parse_args()
    if not args.audio.exists():
        raise FileNotFoundError(args.audio)

    win_s = args.win_ms / 1000.0
    hop_s = args.hop_ms / 1000.0
    dur_s = wav_duration_seconds(args.audio)
    if args.inspect_workers:
        inspect_worker_variants(
            feature_set=args.feature_set,
            feature_lvl=args.feature_lvl,
            requested_workers=args.workers,
        )
    if args.benchmark_worker_variants:
        starts, ends = make_segments(
            duration_s=dur_s,
            win_s=args.win_ms / 1000.0,
            hop_s=args.hop_ms / 1000.0,
        )
        benchmark_worker_variants(
            audio_path=args.audio,
            feature_set=args.feature_set,
            feature_lvl=args.feature_lvl,
            starts=starts,
            ends=ends,
            requested_workers=args.workers,
        )

    if args.sweep_start_ms is not None:
        print(
            f"[SWEEP] file={args.audio} duration={dur_s:.3f}s "
            f"start={args.sweep_start_ms}ms step={args.sweep_step_ms}ms max={args.sweep_max_ms}ms"
        )
        current_ms = float(args.sweep_start_ms)
        found = False
        while current_ms <= args.sweep_max_ms + 1e-9:
            cur_win_s = current_ms / 1000.0
            starts, ends = make_segments(duration_s=dur_s, win_s=cur_win_s, hop_s=cur_win_s)
            ds_out, ds_t = run_dataset_iterator_mode(
                audio_path=args.audio,
                feature_set=args.feature_set,
                feature_lvl=args.feature_lvl,
                win_ms=current_ms,
                hop_ms=current_ms,
            )
            st_out, st_t = run_file_segments_single_worker(
                audio_path=args.audio,
                feature_set=args.feature_set,
                feature_lvl=args.feature_lvl,
                starts=starts,
                ends=ends,
            )
            mt_out, mt_t = run_file_segments_multi_worker(
                audio_path=args.audio,
                feature_set=args.feature_set,
                feature_lvl=args.feature_lvl,
                starts=starts,
                ends=ends,
                workers=args.workers,
            )
            ds_nan = nan_ratio(ds_out)
            st_nan = nan_ratio(st_out)
            mt_nan = nan_ratio(mt_out)
            ds_st = compare_frames_summary(ds_out, st_out)
            st_mt = compare_frames_summary(st_out, mt_out)
            ds_mt = compare_frames_summary(ds_out, mt_out)
            def _fmt_cmp(cmp_summary):
                if cmp_summary is None:
                    return "no-overlap"
                _, _, mean_abs, max_abs = cmp_summary
                if mean_abs is None:
                    return "all-NaN"
                return f"mean={mean_abs:.3g},max={max_abs:.3g}"
            print(
                f"[SWEEP] win/hop={current_ms:.1f}ms segments={len(starts)} "
                f"ds_rows={len(ds_out)} ds_nan={ds_nan:.6f} ds_t={ds_t:.3f}s "
                f"single_rows={len(st_out)} single_nan={st_nan:.6f} single_t={st_t:.3f}s "
                f"multi_rows={len(mt_out)} multi_nan={mt_nan:.6f} multi_t={mt_t:.3f}s "
                f"cmp(ds,s)={_fmt_cmp(ds_st)} cmp(s,m)={_fmt_cmp(st_mt)} cmp(ds,m)={_fmt_cmp(ds_mt)}"
            )
            if ds_nan == 0.0 and st_nan == 0.0 and mt_nan == 0.0:
                print(f"[SWEEP] first no-NaN window/hop across all modes: {current_ms:.1f}ms")
                found = True
                break
            current_ms += args.sweep_step_ms
        if not found:
            print("[SWEEP] no no-NaN window found in configured range")
        return

    starts, ends = make_segments(duration_s=dur_s, win_s=win_s, hop_s=hop_s)
    print(
        f"[INPUT] file={args.audio} duration={dur_s:.3f}s "
        f"win={args.win_ms}ms hop={args.hop_ms}ms segments={len(starts)}"
    )

    ds_out, ds_t = run_dataset_iterator_mode(
        audio_path=args.audio,
        feature_set=args.feature_set,
        feature_lvl=args.feature_lvl,
        win_ms=args.win_ms,
        hop_ms=args.hop_ms,
    )
    st_out, st_t = run_file_segments_single_worker(
        audio_path=args.audio,
        feature_set=args.feature_set,
        feature_lvl=args.feature_lvl,
        starts=starts,
        ends=ends,
    )
    mt_out, mt_t = run_file_segments_multi_worker(
        audio_path=args.audio,
        feature_set=args.feature_set,
        feature_lvl=args.feature_lvl,
        starts=starts,
        ends=ends,
        workers=args.workers,
    )

    describe("dataset_iterator", ds_out, ds_t)
    describe("file_segments_single", st_out, st_t)
    describe("file_segments_multi", mt_out, mt_t)

    compare_frames(ds_out, st_out, "dataset_iterator", "file_segments_single")
    compare_frames(st_out, mt_out, "file_segments_single", "file_segments_multi")
    compare_frames(ds_out, mt_out, "dataset_iterator", "file_segments_multi")


if __name__ == "__main__":
    main()
