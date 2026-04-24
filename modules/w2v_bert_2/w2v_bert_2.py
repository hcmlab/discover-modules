# Copyright (c) 2024, Dominik Schiller
import os
from time import perf_counter

import numpy as np
import torch
import torchaudio
from discover_utils.data.stream import SSIStream
from discover_utils.interfaces.server_module import Processor
from discover_utils.utils.log_utils import log
from discover_utils.utils.type_definitions import SSINPDataType
from transformers import AutoFeatureExtractor, Wav2Vec2BertModel
from transformers.utils import logging as hf_logging

INPUT_ID = "audio"
OUTPUT_ID = "embeddings"

_default_options = {"batch_size": 250}


class W2VBert2(Processor):
    def __init__(self, *args, **kwargs):
        # Setting options
        super().__init__(*args, **kwargs)
        self.options = _default_options | self.options
        self.batch_size = int(self.options["batch_size"])

        # Build model
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        log(f"Using device {self._device}")
        hf_logging.disable_progress_bar()

        # https://github.com/huggingface/blog/blob/main/fine-tune-w2v2-bert.md
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(
            "facebook/w2v-bert-2.0"
        )
        self.model = Wav2Vec2BertModel.from_pretrained("facebook/w2v-bert-2.0").to(
            self._device
        )
        self.model.eval()

        self.model_sr = 16000
        self.ds_iterator = None
        self._output_sr = None

    def _process_batch(self, batch):
        preprocessed = self.feature_extractor(
            batch, sampling_rate=self.model_sr, return_tensors="pt"
        ).to(self._device)
        with torch.no_grad():
            pred = self.model(**preprocessed)
        return pred

    @staticmethod
    def _ms_to_seconds(ms: int | None) -> float:
        if ms is None or ms <= 0:
            return 0.0
        return ms / 1000.0

    @staticmethod
    def _pad_embeddings_edges(embeddings: np.ndarray, pad_left: int, pad_right: int) -> np.ndarray:
        if pad_left <= 0 and pad_right <= 0:
            return embeddings
        if embeddings.shape[0] == 0:
            # Fallback for degenerate cases.
            return np.zeros((max(1, pad_left + pad_right), 1024), dtype=np.float32)
        return np.pad(embeddings, ((pad_left, pad_right), (0, 0)), mode="edge")

    def _extract_native_embeddings(self, signal: np.ndarray) -> np.ndarray:
        t_start = perf_counter()
        outputs = self._process_batch([signal])
        # Sequence output at model-native temporal resolution.
        embeddings = outputs[0].squeeze(0).cpu().numpy()
        log(
            f"Native extraction finished in {perf_counter() - t_start:.2f}s "
            f"with {embeddings.shape[0]} frames."
        )
        return embeddings

    @staticmethod
    def _to_mono(signal: np.ndarray) -> np.ndarray:
        if signal.ndim == 1:
            return signal
        # support common shapes (num_samples, num_channels) and (num_channels, num_samples)
        if signal.shape[0] <= 8 and signal.shape[1] > signal.shape[0]:
            return np.mean(signal, axis=0)
        return np.mean(signal, axis=1)

    @staticmethod
    def _resample(signal: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        if orig_sr == target_sr:
            return signal
        wav = torch.as_tensor(signal, dtype=torch.float32)
        wav = torchaudio.functional.resample(wav, orig_freq=orig_sr, new_freq=target_sr)
        return wav.cpu().numpy()

    def _pool_embeddings(
        self,
        embeddings: np.ndarray,
        native_sr: float,
        duration_s: float,
        left_ms: int,
        frame_ms: int,
        right_ms: int,
        stride_ms: int,
    ) -> tuple[np.ndarray, float]:
        # If pooling params are invalid, keep native embeddings.
        if native_sr <= 0 or duration_s <= 0 or frame_ms <= 0 or stride_ms <= 0:
            return embeddings, native_sr

        # Iterator-style windows:
        # frame_start = i * stride, frame_end = frame_start + frame_size
        # window = [frame_start-left, frame_end+right]
        n_windows = int(np.floor((duration_s * 1000.0) / stride_ms))
        if n_windows <= 0:
            return embeddings, native_sr

        pooled = []
        for i in range(n_windows):
            frame_start_ms = i * stride_ms
            frame_end_ms = frame_start_ms + frame_ms
            win_start_ms = frame_start_ms - left_ms
            win_end_ms = frame_end_ms + right_ms

            raw_start = (win_start_ms / 1000.0) * native_sr
            raw_end = (win_end_ms / 1000.0) * native_sr
            start_idx = int(np.floor(raw_start))
            end_idx = int(np.ceil(raw_end))
            expected = max(1, int(np.ceil(((win_end_ms - win_start_ms) / 1000.0) * native_sr)))

            left_pad = max(0, -start_idx)
            right_pad = max(0, end_idx - embeddings.shape[0])
            start_idx = max(0, start_idx)
            end_idx = min(embeddings.shape[0], end_idx)

            chunk = embeddings[start_idx:end_idx]
            chunk = self._pad_embeddings_edges(chunk, left_pad, right_pad)
            if chunk.shape[0] > expected:
                chunk = chunk[:expected]
            elif chunk.shape[0] < expected:
                chunk = self._pad_embeddings_edges(chunk, 0, expected - chunk.shape[0])
            pooled.append(chunk.mean(axis=0))

        pooled = np.asarray(pooled, dtype=np.float32)
        return pooled, (1000.0 / stride_ms)

    def process_data(self, ds_iterator) -> np.ndarray:
        self.ds_iterator = ds_iterator
        output_list = []
        for session_name, session in ds_iterator.sessions.items():
            session_manager = session.get("manager")
            if session_manager is None:
                continue
            session_manager.load()
            ds_iterator.current_session = session_manager
            ds_iterator.current_session_info = session.get("info")

            audio = session_manager.input_data[INPUT_ID]
            sig = np.squeeze(np.asarray(audio.data))
            if sig.ndim > 1:
                sig = self._to_mono(sig)
            sr = int(audio.meta_data.sample_rate)
            if sr != self.model_sr:
                sig = self._resample(sig, orig_sr=sr, target_sr=self.model_sr)
                sr = self.model_sr

            if sig.size == 0:
                raise ValueError(f"Empty audio signal in session {session_name}")

            duration_s = float(sig.shape[0]) / float(sr)
            native = self._extract_native_embeddings(sig)
            native_sr = native.shape[0] / duration_s if duration_s > 0 else 0.0

            left_ms = int(ds_iterator.left_context or 0)
            frame_ms = int(ds_iterator.frame_size or 0)
            right_ms = int(ds_iterator.right_context or 0)
            stride_ms = int(ds_iterator.stride or frame_ms)

            pooled, output_sr = self._pool_embeddings(
                embeddings=native,
                native_sr=native_sr,
                duration_s=duration_s,
                left_ms=left_ms,
                frame_ms=frame_ms,
                right_ms=right_ms,
                stride_ms=stride_ms,
            )
            self._output_sr = output_sr
            output_list.append(pooled)
            log(
                f"Session {session_name}: native_sr={native_sr:.3f}Hz, "
                f"output_sr={output_sr:.3f}Hz, rows={pooled.shape[0]}"
            )

        return np.concatenate(output_list, axis=0) if output_list else np.empty((0, 1024), dtype=np.float32)

    def to_output(self, data):
        # Embeddings
        sm = self.ds_iterator.current_session
        stream_template = sm.output_data_templates[OUTPUT_ID]

        sm.output_data_templates[OUTPUT_ID] = SSIStream(
            data=np.array(data, SSINPDataType.FLOAT.value),
            sample_rate=float(self._output_sr) if self._output_sr else (1000 / self.ds_iterator.stride),
            role=stream_template.meta_data.role,
            dataset=stream_template.meta_data.dataset,
            name=stream_template.meta_data.name,
            session=stream_template.meta_data.session,
        )
        return sm.output_data_templates


if __name__ == "__main__":
    from pathlib import Path

    from dotenv import load_dotenv
    from discover_utils.data.provider.dataset_iterator import DatasetIterator
    from discover_utils.utils.ssi_xml_utils import Trainer

    load_dotenv(r"../.env")

    ew2v_trainer = Trainer()
    ew2v_trainer.load_from_file("w2v_bert_2.trainer")
    en = W2VBert2(model_io=None, opts={"batch_size": 5}, trainer=ew2v_trainer)

    data_dir = Path(os.getenv("TEST_DIR"))

    for audio in data_dir.rglob("*.wav"):
        # Inputs
        dd_input_audio = {
            "src": "file:stream:audio",
            "type": "input",
            "id": INPUT_ID,
            "uri": str(audio),
        }

        dd_output_embeddings = {
            "src": "file:stream:SSIStream",
            "type": "output",
            "id": OUTPUT_ID,
            "uri": str(audio.parent / "w2v2bert.stream"),
        }

        ds_iterator = DatasetIterator(
            data_description=[dd_input_audio, dd_output_embeddings],
            frame_size=ew2v_trainer.meta_frame_step,
            left_context=ew2v_trainer.meta_left_ctx,
            start=0,
        )
        ds_iterator.load()
        emotions = en.process_data(ds_iterator)
        output = en.to_output(emotions)
        breakpoint()
