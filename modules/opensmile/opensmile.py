import warnings
import numpy as np
import opensmile
import pandas as pd
import inspect
import os
from time import perf_counter
from discover_utils.interfaces.server_module import Processor
from discover_utils.data.stream import SSIStream
from discover_utils.utils.log_utils import log

INPUT_ID = "input_audio"
OUTPUT_ID = "output_stream"

_default_options = {
    "feature_set": "eGeMAPSv02",
    "feature_lvl": "Functionals",
    "file_multiprocessing": True,
    "file_num_workers": "0",
}


class OpenSmile(Processor):
    chainable = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.options = _default_options | self.options

        self.feature_set = opensmile.FeatureSet.__members__[
            self.options["feature_set"]

        ].value
        self.feature_lvl = opensmile.FeatureLevel.__members__[
            self.options["feature_lvl"]
        ].value
        self.input_sr = 16000
        self.media_type_id = (
            self.options["feature_set"] + "_" + self.options["feature_lvl"]
        )
        self.file_multiprocessing = bool(self.options["file_multiprocessing"])
        try:
            self.file_num_workers = int(self.options["file_num_workers"])
        except (TypeError, ValueError):
            self.file_num_workers = 0
        if self.file_num_workers <= 0:
            self.file_num_workers = os.cpu_count() or 1
        self._file_level_segmented = False

        log(f'Initialized OpenSmile processor with feature set {self.feature_set} and feature level {self.feature_lvl}')

        smile_kwargs = {
            "feature_set": opensmile.FeatureSet(self.feature_set),
            "feature_level": opensmile.FeatureLevel(self.feature_lvl),
        }
        smile_init_sig = inspect.signature(opensmile.Smile.__init__)
        if "num_workers" in smile_init_sig.parameters:
            smile_kwargs["num_workers"] = self.file_num_workers
        if "multiprocessing" in smile_init_sig.parameters:
            smile_kwargs["multiprocessing"] = self.file_multiprocessing

        self.smile = opensmile.Smile(**smile_kwargs)
        configured_workers = self._resolve_worker_count()
        log(
            "OpenSmile worker config: "
            f"multiprocessing={self.file_multiprocessing}, "
            f"requested_workers={self.file_num_workers}, "
            f"workers={configured_workers if configured_workers is not None else 'unknown'}, "
            f"cpu_count={os.cpu_count()}"
        )

        self._dl = self.smile.feature_names
        self._dim_labels = [{"id": i, "name": x} for i, x in enumerate(self._dl)]

    @staticmethod
    def _to_seconds(ms: int | None) -> float:
        if ms is None or ms <= 0:
            return 0.0
        return ms / 1000.0

    @staticmethod
    def _estimate_segments(duration_s: float, win_s: float, hop_s: float) -> int:
        if duration_s <= 0 or win_s <= 0 or hop_s <= 0:
            return 1
        return max(1, int(np.floor((duration_s - win_s + 1e-12) / hop_s) + 1))

    def _resolve_worker_count(self):
        candidates = [
            ("num_workers",),
            ("process", "num_workers"),
            ("_process", "num_workers"),
            ("process", "process", "num_workers"),
        ]
        for chain in candidates:
            obj = self.smile
            ok = True
            for attr in chain:
                if not hasattr(obj, attr):
                    ok = False
                    break
                obj = getattr(obj, attr)
            if ok and isinstance(obj, int):
                return obj
        return None

    @staticmethod
    def _frame_starts(duration_s: float, hop_s: float) -> np.ndarray:
        if duration_s <= 0 or hop_s <= 0:
            return np.asarray([], dtype=np.float64)
        num_frames = int(np.floor(duration_s / hop_s))
        if num_frames <= 0:
            return np.asarray([], dtype=np.float64)
        return np.arange(num_frames, dtype=np.float64) * hop_s

    @staticmethod
    def _pad_signal_edges(signal: np.ndarray, pad_left: int, pad_right: int) -> np.ndarray:
        if pad_left <= 0 and pad_right <= 0:
            return signal
        if signal.size == 0:
            total = max(1, pad_left + pad_right)
            return np.zeros((total,), dtype=np.float32)
        if signal.ndim == 1:
            pad_width = (pad_left, pad_right)
        else:
            pad_width = ((pad_left, pad_right),) + tuple((0, 0) for _ in signal.shape[1:])
        return np.pad(signal, pad_width=pad_width, mode="edge")

    def _segment_file_with_padding(
        self,
        file_path: str,
        audio_data: np.ndarray | None,
        sample_rate: int,
        duration_s: float,
        left_s: float,
        frame_s: float,
        right_s: float,
        hop_s: float,
    ) -> pd.DataFrame:
        frame_starts = self._frame_starts(duration_s, hop_s)
        if frame_starts.size == 0:
            return self.smile.process_file(str(file_path))

        win_starts = frame_starts - left_s
        win_ends = frame_starts + frame_s + right_s
        inside_mask = (win_starts >= 0.0) & (win_ends <= duration_s)
        inside_idx = np.where(inside_mask)[0]
        edge_idx = np.where(~inside_mask)[0]

        rows: list[pd.DataFrame | None] = [None] * len(frame_starts)

        if inside_idx.size > 0:
            inside_files = [str(file_path)] * int(inside_idx.size)
            inside = self.smile.process_files(
                inside_files,
                starts=win_starts[inside_idx].tolist(),
                ends=win_ends[inside_idx].tolist(),
            ).reset_index(drop=True)
            for j, idx in enumerate(inside_idx):
                rows[int(idx)] = inside.iloc[[j]]

        if edge_idx.size > 0:
            if audio_data is None:
                raise ValueError("Audio samples are required for edge-padding in segmented file-level mode.")
            duration_samples = int(round(duration_s * sample_rate))
            audio_data = np.asarray(audio_data)
            for idx in edge_idx:
                ws = float(win_starts[idx])
                we = float(win_ends[idx])
                s0 = int(round(max(0.0, ws) * sample_rate))
                s1 = int(round(min(duration_s, we) * sample_rate))
                s0 = min(max(0, s0), duration_samples)
                s1 = min(max(0, s1), duration_samples)
                chunk = audio_data[s0:s1]
                pad_left = int(round(max(0.0, -ws) * sample_rate))
                pad_right = int(round(max(0.0, we - duration_s) * sample_rate))
                chunk = self._pad_signal_edges(chunk, pad_left, pad_right)
                rows[int(idx)] = self.smile.process_signal(chunk, sample_rate).reset_index(drop=True)

        out = [r for r in rows if r is not None]
        if not out:
            return pd.DataFrame(columns=self._dl)
        return pd.concat(out, ignore_index=True)

    def process_sample(self, sample):
        # DatasetIterator returns Audio as (num_samples, num_channels)
        # opensmile expects (num_channels, num_samples) or (num_samples,)
        if sample.ndim == 2:
            sample = sample.T
        features = self.smile.process_signal(sample, self.input_sr)
        return features

    def process_data(self, ds_iterator) -> np.ndarray:
        """Returning a dictionary that contains the original keys from the dataset iterator and a list of processed
        samples as value. Can be overwritten to customize the processing"""
        ds_iterator: DatasetIterator
        self.ds_iterator = ds_iterator
        left_s = self._to_seconds(ds_iterator.left_context)
        frame_s = self._to_seconds(ds_iterator.frame_size)
        right_s = self._to_seconds(ds_iterator.right_context)
        win_s = left_s + frame_s + right_s
        hop_s = self._to_seconds(ds_iterator.stride)
        self._file_level_segmented = win_s > 0 and hop_s > 0
        log(
            "OpenSmile file-level mode: "
            f"segmented={self._file_level_segmented}, "
            f"left_s={left_s:.3f}, frame_s={frame_s:.3f}, right_s={right_s:.3f}, "
            f"win_s={win_s:.3f}, hop_s={hop_s:.3f}, "
            f"multiprocessing={self.file_multiprocessing}"
        )
        output_list = []
        total_sessions = len(ds_iterator.sessions)
        for session_idx, (session_name, session) in enumerate(ds_iterator.sessions.items(), start=1):
            t_start_session = perf_counter()
            session_manager = session.get("manager")
            if session_manager is None:
                log(f"Skipping session {session_name}: missing session manager")
                continue
            session_manager.load()
            ds_iterator.current_session = session_manager
            ds_iterator.current_session_info = session.get("info")
            audio = session_manager.input_data[INPUT_ID]
            file_path = getattr(audio.meta_data, "file_path", None)
            if file_path:
                if self._file_level_segmented:
                    duration_ms = audio.meta_data.duration
                    if duration_ms is None:
                        duration_ms = getattr(ds_iterator.current_session_info, "duration", None)
                    duration_s = self._to_seconds(duration_ms)
                    est_segments = self._estimate_segments(duration_s, win_s, hop_s)
                    log(
                        f"Processing session {session_idx}/{total_sessions}: "
                        f"{session_name}, duration={duration_s:.3f}s, segments~{est_segments}"
                    )
                    if duration_s <= 0:
                        features = self.smile.process_file(str(file_path))
                    else:
                        features = self._segment_file_with_padding(
                            file_path=str(file_path),
                            audio_data=audio.data,
                            sample_rate=int(audio.meta_data.sample_rate),
                            duration_s=duration_s,
                            left_s=left_s,
                            frame_s=frame_s,
                            right_s=right_s,
                            hop_s=hop_s,
                        )
                else:
                    log(f"Processing session {session_idx}/{total_sessions}: {session_name}, full-file extraction")
                    features = self.smile.process_file(str(file_path))
            else:
                log(f"Processing session {session_idx}/{total_sessions}: {session_name}, in-memory signal fallback")
                self.input_sr = int(audio.meta_data.sample_rate)
                features = self.smile.process_signal(audio.data, self.input_sr)
            output_list.append(features)
            # For very long files on constrained hardware, we can process segment jobs in batches
            # and log progress/ETA per batch. This adds call overhead but improves observability.
            elapsed_session = perf_counter() - t_start_session
            rows = len(features) if hasattr(features, "__len__") else 0
            log(
                f"Completed session {session_idx}/{total_sessions}: {session_name} "
                f"in {elapsed_session:.2f}s, rows={rows}"
            )
        output_list = pd.concat(output_list, ignore_index=True).to_numpy()
        return output_list

    def to_output(self, data: np.ndarray) -> dict:
        output_templates = self.ds_iterator.current_session.output_data_templates
        if self._file_level_segmented and self.ds_iterator.stride and self.ds_iterator.stride > 0:
            sample_rate = 1000.0 / self.ds_iterator.stride
            chunks = None
        else:
            duration_ms = None
            current_input = self.ds_iterator.current_session.input_data.get(INPUT_ID)
            if current_input is not None:
                duration_ms = current_input.meta_data.duration
            if duration_ms is None:
                current_info = getattr(self.ds_iterator, "current_session_info", None)
                duration_ms = getattr(current_info, "duration", None)
            if duration_ms and duration_ms > 0:
                sample_rate = 1000.0 / duration_ms
                chunks = np.asarray(
                    [(0.0, duration_ms / 1000.0, 0, 1)],
                    dtype=SSIStream.CHUNK_DTYPE,
                )
            else:
                sample_rate = 1.0
                chunks = None
        output_templates[OUTPUT_ID] = SSIStream(
            data=data,
            sample_rate=sample_rate,
            chunks=chunks,
            dim_labels=self._dim_labels,
            media_type=self.media_type_id,
            role=output_templates[OUTPUT_ID].meta_data.role,
            dataset=output_templates[OUTPUT_ID].meta_data.dataset,
            name=output_templates[OUTPUT_ID].meta_data.name,
            session=output_templates[OUTPUT_ID].meta_data.session,
        )

        return output_templates


if __name__ == "__main__":
    import os
    from pathlib import Path

    PYTORCH_ENABLE_MPS_FALLBACK = 1
    from discover_utils.utils.ssi_xml_utils import Trainer
    from discover_utils.data.provider.dataset_iterator import DatasetIterator
    import dotenv

    env = dotenv.load_dotenv(r"../.env")
    data_dir = Path(os.getenv("TEST_DIR"))

    os_trainer = Trainer()
    os_trainer.load_from_file("opensmile.trainer")
    os_extractor = OpenSmile(model_io=None, opts={}, trainer=os_trainer)

    for audio in data_dir.glob("*.wav"):

        # Inputs
        input_audio = {
            "src": "file:stream:audio",
            "type": "input",
            "id": INPUT_ID,
            "uri": str(audio),
        }

        # Outputs
        output_audio = {
            "src": "file:stream:SSIStream:feature",
            "type": "output",
            "id": OUTPUT_ID,
            "uri": str(audio.parent / "gemaps.stream"),
        }

        dm_audio = DatasetIterator(
            data_description=[input_audio, output_audio],
            frame_size=os_trainer.meta_frame_step,
            left_context=os_trainer.meta_left_ctx,
        )
        dm_audio.load()
        data = os_extractor.process_data(dm_audio)
        dm_audio.current_session.output_data_templates = os_extractor.to_output(data)
        dm_audio.save()
        breakpoint()
