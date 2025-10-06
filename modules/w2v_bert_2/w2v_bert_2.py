# Copyright (c) 2024, Dominik Schiller
import os

import librosa
from discover_utils.interfaces.server_module import Processor
from discover_utils.utils.type_definitions import SSINPDataType
from discover_utils.data.provider.dataset_iterator import DatasetIterator
from discover_utils.utils.log_utils import log
from discover_utils.utils.ssi_xml_utils import Trainer
from transformers import AutoFeatureExtractor, Wav2Vec2BertModel

from time import perf_counter
from discover_utils.data.stream import SSIStream

import torch
import numpy as np

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

        # https://github.com/huggingface/blog/blob/main/fine-tune-w2v2-bert.md
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(
            "facebook/w2v-bert-2.0"
        )
        self.model = Wav2Vec2BertModel.from_pretrained("facebook/w2v-bert-2.0").to(
            self._device
        )

        self.model_sr = 16000

    def _process_batch(self, batch):
        preprocessed = self.feature_extractor(
            batch, sampling_rate=self.model_sr, return_tensors="pt"
        ).to(self._device)
        with torch.no_grad():
            pred = self.model(**preprocessed)
        return pred

    def process_data(self, ds_iterator: DatasetIterator) -> tuple:
        self.ds_iterator = ds_iterator
        current_batch = []
        output_embeddings = []
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
        garbage_output = np.zeros((1024,), dtype=np.float32)

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
                # https://bagustris.wordpress.com/2022/08/23/acoustic-feature-extraction-with-transformers/
                outputs = self._process_batch(current_batch)
                # Average last hidden states
                hidden_states = torch.mean(outputs[0], dim=1)
                output_embeddings.append(hidden_states)
                log(
                    f"Batch {int(i / self.batch_size)} : {int((self.batch_size / (perf_counter() - t_start)))} samples/s"
                )
                current_batch = []
                t_start = perf_counter()

        # Process last batch
        if current_batch:
            outputs = self._process_batch(current_batch)
            # Average last hidden states
            hidden_states = torch.mean(outputs[0], dim=1)
            output_embeddings.append(hidden_states)
            log(
                f"Partial batch with {len(current_batch)} samples: {int(len(current_batch) / (perf_counter() - t_start))} samples/s"
            )


        output_embeddings = torch.concatenate(output_embeddings).cpu().numpy()
        if nan_idx:
            log(f"Replacing {len(nan_idx)} nan-vectors with zero-vectors")
            output_embeddings[nan_idx] = garbage_output
        return output_embeddings

    def to_output(self, data):
        # Embeddings
        sm = self.ds_iterator.current_session
        stream_template = sm.output_data_templates[OUTPUT_ID]

        sm.output_data_templates[OUTPUT_ID] = SSIStream(
            data=np.array(data, SSINPDataType.FLOAT.value),
            sample_rate= 1000 / self.ds_iterator.stride,
            role=stream_template.meta_data.role,
            dataset=stream_template.meta_data.dataset,
            name=stream_template.meta_data.name,
            session=stream_template.meta_data.session,
        )
        return sm.output_data_templates


if __name__ == "__main__":

    from pathlib import Path
    from dotenv import load_dotenv

    load_dotenv(r"../.env")

    ew2v_trainer = Trainer()
    ew2v_trainer.load_from_file("w2v_bert_2.trainer")
    en = W2VBert2(model_io=None, opts={"batch_size" : 5}, trainer=ew2v_trainer)

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
