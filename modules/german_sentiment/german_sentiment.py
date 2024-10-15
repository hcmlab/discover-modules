import numpy as np
import os
from discover_utils.interfaces.server_module import Processor
from germansentiment import SentimentModel
from discover_utils.utils.anno_utils import resample

#TODO Seems to be not working at the moment
os.environ["PYTORCH_PRETRAINED_BERT_CACHE"] = os.getenv('CACHE_DIR', 'cache')
os.environ["PYTORCH_TRANSFORMERS_CACHE"] = os.getenv('CACHE_DIR', 'cache')
os.environ["HUGGINGFACE_HUB_CACHE"] = os.getenv('CACHE_DIR', 'cache')
os.environ["HF_MODULES_CACHE"] = os.getenv('CACHE_DIR', 'cache')

_default_options = {}
_dim_labels = [
    "sentiment",
]

INPUT_ID = 'transcript'
OUTPUT_ID = 'sentiment'

class GermanSentiment(Processor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.options = _default_options | self.options


        self.options = _default_options | self.options
        self.model = SentimentModel()
        self.ds_iter = None

    def preprocess_sample(self, sample):
        return sample

    def process_sample(self, sample):
        return sample

    def postprocess_sample(self, sample):
        return sample

    def process_data(self, ds_iter) -> dict:
        self.ds_iter = ds_iter

        #anno_key = self.options["transcript"]
        #if not anno_key:
        #    self.logger.error("Option 'transcript_in' not set.")
        #    raise ValueError
        #if anno_key not in ds_iter.anno_schemes.keys():
        #    self.logger.error(f" Key '{anno_key}' not found in loaded Annoations. Valid keys are f{[x for x in ds_iter.anno_schemes.keys()]} ")
        #    raise ValueError


        # Concatenating all labels to one string per window
        sample_data = [" ".join(window["transcript"]) for window in list(ds_iter)]

        # Predicting all windows
        _, preds = self.model.predict_sentiment(sample_data, output_probabilities=True)

        # Calculating sentiment from pos / neg / neutral: 1*pos - 1*neg + 0*neutr
        expect = [x[0][1] - x[1][1] for x in preds]

        # Set empty string predictions zu 0
        expect_clean = np.asarray([e if s else 0.0 for e,s in zip(expect, sample_data)])

        return expect_clean

    def to_output(self, data) -> dict:
        # Append necessary meta information
        output_anno = self.ds_iter.current_session.output_data_templates[OUTPUT_ID]

        src_sr = (1 / self.ds_iter.stride) * 1000
        trgt_sr = output_anno.annotation_scheme.sample_rate
        output_anno.data = data.astype(output_anno.annotation_scheme.label_dtype)
        output_anno.data = resample(output_anno.data,src_sr, trgt_sr)
        return self.ds_iter.current_session.output_data_templates

if __name__ == '__main__':
    import dotenv
    import os
    from discover_utils.utils.ssi_xml_utils import Trainer
    from pathlib import Path
    from discover_utils.data.provider.dataset_iterator import DatasetIterator
    from discover_utils.data.provider.data_manager import NovaDatasetManager

    env = dotenv.load_dotenv(r'../.env')
    IP = os.getenv("NOVA_IP", "")
    PORT = int(os.getenv("NOVA_PORT", 0))
    USER = os.getenv("NOVA_USER", "")
    PASSWORD = os.getenv("NOVA_PASSWORD", "")
    DATA_DIR = Path(os.getenv("NOVA_DATA_DIR", None))

    test_fd = os.getenv('TEST_DIR')
    test_file = Path(test_fd) / 'german_sentiment' / 'transcript.annotation'

    # Trainer
    gs_trainer = Trainer()
    gs_trainer.load_from_file("german_sentiment.trainer")
    gs = GermanSentiment(model_io=None, opts={}, trainer=gs_trainer)

    # Inputs
    transcript = {
        "src": "file:annotation",
        "type": "input",
        "id": INPUT_ID,
        "uri": str(test_file),
    }

    # Output
    sentiment = {
        "src": "db:annotation:Continuous",
        "scheme": "sentiment",
        "role": "testrole",
        "annotator": "schildom",
        "type": "output",
        "id": OUTPUT_ID,
        "uri": str(test_file),
    }

    ctx = {
        "db": {
            "db_host": IP,
            "db_port": PORT,
            "db_user": USER,
            "db_password": PASSWORD,
            "data_dir": DATA_DIR,
        },
    }


    ds_iterator = DatasetIterator(
        dataset_manager= NovaDatasetManager(data_description=[transcript, sentiment], source_context=ctx, dataset='test'),
        frame_size=gs_trainer.meta_frame_step,
        left_context=gs_trainer.meta_left_ctx
    )
    #ds_iterator.load()
    emotions = gs.process_data(ds_iterator)
    output = gs.to_output(emotions)
    breakpoint()


