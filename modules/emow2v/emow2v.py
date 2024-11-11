# Copyright (c) 2023, Dominik Schiller
import sys
import os

# Add local dir to path for relative imports
sys.path.insert(0, os.path.dirname(__file__))

import librosa
from discover_utils.interfaces.server_module import Processor
from discover_utils.utils.type_definitions import SSINPDataType
from discover_utils.data.provider.dataset_iterator import DatasetIterator
from discover_utils.utils.log_utils import log
from discover_utils.utils.ssi_xml_utils import Trainer
from emow2v_source import EmotionModel
from transformers import Wav2Vec2Processor
from time import perf_counter
from discover_utils.data.stream import SSIStream
from discover_utils.data.annotation import ContinuousAnnotation, ContinuousAnnotationScheme


import torch
import numpy as np

INPUT_ID = "audio"
OUTPUT_ID_VALENCE = "valence"
OUTPUT_ID_AROUSAL = "arousal"
OUTPUT_ID_DOMINANCE = "dominance"
OUTPUT_ID_EMBEDDINGS = "embeddings"

_default_options = {"batch_size": 250}

PYTORCH_ENABLE_MPS_FALLBACK = 1


class EmoW2V(Processor):
    def __init__(self, *args, **kwargs):
        # Setting options
        super().__init__(*args, **kwargs)
        self.options = _default_options | self.options
        self.batch_size = int(self.options["batch_size"])

        # Build model
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        log(f"Using device {self._device}")
        model_name = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = EmotionModel.from_pretrained(model_name).to(self._device)
        self.model.eval()
        self.model_sr = 16000

    def _process_batch(self, batch):
        preprocessed = self.processor(
            batch, sampling_rate=self.model_sr, return_tensors="pt"
        ).to(self._device)
        with torch.no_grad():
            pred = self.model(preprocessed.input_values)
        return pred

    def process_data(self, ds_iterator: DatasetIterator) -> tuple:
        self.ds_iterator = ds_iterator
        current_batch = []
        output_embeddings = []
        output_predictions = []
        nan_idx = []
        t_start = perf_counter()

        # Placeholder array to keep sample order for predictions on garbage data
        garbage_input = np.zeros(
            (
                int(
                    (self.model_sr / 1000)
                    * (
                        ds_iterator.left_context
                        + ds_iterator.frame_size
                        + ds_iterator.right_context
                    )
                ),
            ),
            dtype=np.float32,
        )
        garbage_output_embedding = np.zeros((1024,), dtype=np.float32)
        garbage_output_prediction = np.zeros((3,), dtype=np.float32)

        for i, sample in enumerate(ds_iterator):
            sr = int(
                ds_iterator.current_session.input_data[INPUT_ID].meta_data.sample_rate
            )

            if isinstance(sample[INPUT_ID], type(np.NAN)) and np.isnan(
                sample[INPUT_ID]
            ):
                nan_idx.append(i)
                sig_pp = garbage_input
            else:
                # Preprocess input
                sig_pp = np.squeeze(sample[INPUT_ID])
                if len(sig_pp.shape) > 1:
                    sig_pp = librosa.to_mono(np.swapaxes(sig_pp, 0, 1))
                sig_pp = librosa.resample(sig_pp, orig_sr=sr, target_sr=self.model_sr)

            current_batch.append(sig_pp)

            if i > 0 and i % self.batch_size == 0:
                pred = self._process_batch(current_batch)
                log(
                    f"Batch {i / self.batch_size} : {int((self.batch_size / (perf_counter() - t_start)))} samples/s"
                )
                output_embeddings.append(pred[0].cpu().numpy())
                output_predictions.append(pred[1].cpu().numpy())
                current_batch = []
                t_start = perf_counter()

        # Process last batch
        if current_batch:
            pred = self._process_batch(current_batch)
            log(
                f"Partial batch with {len(current_batch)} samples: {int((self.batch_size / (perf_counter() - t_start)))} samples/s"
            )
            output_embeddings.append(pred[0].cpu().numpy())
            output_predictions.append(pred[1].cpu().numpy())

        output_embeddings = np.vstack(output_embeddings)
        output_predictions = np.vstack(output_predictions)

        if nan_idx:
            log(f"Replacing {len(nan_idx)} nan-vectors with zero-vectors")
            output_embeddings[nan_idx] = garbage_output_embedding
            output_predictions[nan_idx] = garbage_output_prediction

        return (output_embeddings, output_predictions)

    def to_output(self, data):
        embeddings, anno_data = data
        anno_data = np.clip(anno_data, 0, 1)
        output = {}

        # Annotations in output order a,d,v
        for i, output_id in enumerate(
            [OUTPUT_ID_AROUSAL, OUTPUT_ID_DOMINANCE, OUTPUT_ID_VALENCE]
        ):
            scheme = ContinuousAnnotationScheme(
                sample_rate=1000 / self.ds_iterator.stride,
                min_val=0,
                max_val=1,
                name=output_id,
            )
            annotation = ContinuousAnnotation(
                scheme=scheme,
                data=np.array(anno_data[:, i], dtype=scheme.label_dtype),
            )
            output[output_id] = annotation

        # Embeddings
        output[OUTPUT_ID_EMBEDDINGS] = SSIStream(
            data=np.array(embeddings, SSINPDataType.FLOAT.value),
            sample_rate=self.ds_iterator.stride,
        )

        return output


if __name__ == "__main__":

    from pathlib import Path
    from dotenv import load_dotenv

    load_dotenv(r"../.env")

    ew2v_trainer = Trainer()
    ew2v_trainer.load_from_file("emow2v.trainer")
    en = EmoW2V(model_io=None, opts={}, trainer=ew2v_trainer)

    data_dir = Path(os.getenv("TEST_DIR"))

    for audio in data_dir.rglob("*.wav"):
        # Inputs
        dd_input_audio = {
            "src": "file:stream:audio",
            "type": "input",
            "id": INPUT_ID,
            "uri": str(audio),
        }

        # Outputs
        dd_output_arousal = {
            "src": "file:annotation:continuous",
            "type": "output",
            "id": OUTPUT_ID_AROUSAL,
            "uri": str(audio.parent / "emow2v_arousal.annotation"),
        }

        dd_output_valence = {
            "src": "file:annotation:continuous",
            "type": "output",
            "id": OUTPUT_ID_VALENCE,
            "uri": str(audio.parent / "emow2v_valence.annotation"),
        }

        dd_output_dominance = {
            "src": "file:annotation:continuous",
            "type": "output",
            "id": OUTPUT_ID_DOMINANCE,
            "uri": str(audio.parent / "emow2v_dominance.annotation"),
        }

        dd_output_embeddings = {
            "src": "file:stream:SSIStream",
            "type": "output",
            "id": OUTPUT_ID_EMBEDDINGS,
            "uri": str(audio.parent / "emow2v.stream"),
        }

        ds_iterator = DatasetIterator(
            data_description=[
                dd_input_audio,
                dd_output_arousal,
                dd_output_valence,
                dd_output_dominance,
                dd_output_embeddings,
            ],
            frame_size=ew2v_trainer.meta_frame_step,
            left_context=ew2v_trainer.meta_left_ctx,
        )
        ds_iterator.load()
        emotions = en.process_data(ds_iterator)
        output = en.to_output(emotions)
        breakpoint()
