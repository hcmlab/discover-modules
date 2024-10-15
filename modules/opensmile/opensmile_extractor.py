import numpy as np
import opensmile
import pandas as pd
from time import perf_counter
from nova_utils.interfaces.server_module import Processor
from nova_utils.data.stream import SSIStream
from nova_utils.utils.log_utils import log

INPUT_ID = "input_audio"
OUTPUT_ID = "output_stream"

_default_options = {
    "feature_set": "eGeMAPSv02",
    "feature_lvl": "Functionals",
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

        log(f'Initialized OpenSmile processor with feature set {self.feature_set} and feature level {self.feature_lvl}')

        self.smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet(self.feature_set),
            feature_level=opensmile.FeatureLevel(self.feature_lvl),
        )

        self._dl = self.smile.feature_names
        self._dim_labels = [{"id": i, "name": x} for i, x in enumerate(self._dl)]

    def process_sample(self, sample):
        features = self.smile.process_signal(sample, self.input_sr)
        return features

    def process_data(self, ds_iterator) -> np.ndarray:
        """Returning a dictionary that contains the original keys from the dataset iterator and a list of processed
        samples as value. Can be overwritten to customize the processing"""
        ds_iterator: DatasetIterator
        self.ds_iterator = ds_iterator
        # Start the stopwatch / counter
        pc_start = perf_counter()
        output_list = []
        for i, sample in enumerate(ds_iterator):
            self.input_sr = int(
                ds_iterator.current_session.input_data[INPUT_ID].meta_data.sample_rate
            )
            if i % 100 == 0:
                log(
                    f"Processing sample {i}. {i / (perf_counter() - pc_start)} samples / s. Processed {i * ds_iterator.stride / 1000} Seconds of data."
                )
            # for id, output_list in processed.items():
            #     data_for_id = {id: sample[id]}
            out = self.preprocess_sample(sample)
            out = self.process_sample(out)
            out = self.postprocess_sample(out)
            output_list.append(out)

        output_list = pd.concat(output_list).to_numpy()

        return output_list

    def to_output(self, data: np.ndarray) -> dict:
        output_templates = self.ds_iterator.current_session.output_data_templates
        output_templates[OUTPUT_ID] = SSIStream(
            data=data,
            sample_rate=1000/self.ds_iterator.stride,
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
    from nova_utils.utils.ssi_xml_utils import Trainer
    from nova_utils.data.provider.dataset_iterator import DatasetIterator
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
