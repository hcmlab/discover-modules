import numpy as np
import os
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoConfig
from discover_utils.interfaces.server_module import Processor
from discover_utils.utils.anno_utils import resample
from scipy.special import softmax
import torch
from discover_utils.utils.type_definitions import SSINPDataType
from discover_utils.data.stream import SSIStream

_default_options = {'model_path': "cardiffnlp/twitter-xlm-roberta-base-sentiment"}
_dim_labels = [
    "sentiment",
]

INPUT_TRANSCRIPT = 'transcript'
OUTPUT_SENTIMENT = 'sentiment'
OUTPUT_EMBEDDING = 'embedding'

class Sentiment(Processor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.options = _default_options | self.options


        self.options = _default_options | self.options
        self.ds_iter = None

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        #self.model_path = f"cardiffnlp/twitter-xlm-roberta-base-sentiment"

        self.tokenizer = AutoTokenizer.from_pretrained(self.options['model_path'])
        self.config = AutoConfig.from_pretrained(self.options['model_path'])

        # PT
        self.model = AutoModelForSequenceClassification.from_pretrained(self.options['model_path'])
        self.model.to(self.device)
        #self.model.save_pretrained(self.model_path)


    def preprocess_sample(self, sample):
        return sample

    def process_sample(self, sample):
        return sample

    def postprocess_sample(self, sample):
        return sample

    def process_data(self, ds_iter) -> dict:
        self.ds_iter = ds_iter

        # Concatenating all labels to one string per window
        sample_data = [" ".join(window["transcript"]) for window in list(ds_iter)]
        

        # Build cache to avoid OOM
        with torch.no_grad():
            sample_data_set = set(sample_data)
            model_out = [self.model(**self.tokenizer(x, return_tensors='pt').to(self.device), output_hidden_states=True) for x in sample_data_set]
            cache_map = {x: softmax(y.logits[0].cpu().numpy()) @ [-1, 0, 1] for x, y in zip(sample_data_set, model_out)}
            embed_map = {x: y.hidden_states[-1][:, 0, :].squeeze().cpu().numpy() for x, y in zip(sample_data_set, model_out)}
        
        preds = np.array([cache_map[x] for x in sample_data])
        embeds = np.array([embed_map[x] for x in sample_data])
        return (preds, embeds)

    def to_output(self, data) -> dict:
        preds, embeds = data
        templates = self.ds_iter.current_session.output_data_templates

        # annotations
        output_anno = templates[OUTPUT_SENTIMENT]

        # sample rates
        src_sr = (1 / self.ds_iter.stride) * 1000
        trgt_sr = output_anno.annotation_scheme.sample_rate

        output_anno.data = preds.astype(output_anno.annotation_scheme.label_dtype)
        output_anno.data = resample(output_anno.data,src_sr, trgt_sr)

        # embeddings
        templates[OUTPUT_EMBEDDING] = SSIStream(
            data=np.array(embeds, SSINPDataType.FLOAT.value),
            sample_rate=src_sr
        )

        return self.ds_iter.current_session.output_data_templates

if __name__ == '__main__':
    import dotenv
    import os
    from discover_utils.utils.ssi_xml_utils import Trainer
    from pathlib import Path
    from discover_utils.data.provider.dataset_iterator import DatasetIterator
    from discover_utils.data.provider.data_manager import DatasetManager

    os.chdir(r'../discover-modules/modules/sentiment')
    env = dotenv.load_dotenv(r'../../.env')
    IP = os.getenv("NOVA_IP", "")
    PORT = int(os.getenv("NOVA_PORT", 0))
    USER = os.getenv("NOVA_USER", "")
    PASSWORD = os.getenv("NOVA_PASSWORD", "")
    DATA_DIR = Path(os.getenv("NOVA_DATA_DIR", None))

    test_fd = os.getenv('TEST_DIR')
    test_file = Path(test_fd) / 'sentiment' / 'transcript.annotation'
    
    # Trainer
    gs_trainer = Trainer()
    gs_trainer.load_from_file("sentiment.trainer")
    gs = Sentiment(model_io=None, opts={}, trainer=gs_trainer)

    # Inputs
    transcript = {
        "src": "file:annotation",
        "type": "input",
        "id": INPUT_TRANSCRIPT,
        "uri": str(test_file),
    }

    # Output
    sentiment = {
        "src": "db:annotation:Continuous",
        "scheme": "sentiment",
        "role": "testrole",
        "annotator": "schildom",
        "type": "output",
        "id": OUTPUT_SENTIMENT,
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
        dataset_manager= DatasetManager(data_description=[transcript, sentiment], source_context=ctx, dataset='test'),
        frame_size=gs_trainer.meta_frame_step,
        left_context=gs_trainer.meta_left_ctx
    )
    #ds_iterator.load()
    emotions = gs.process_data(ds_iterator)
    output = gs.to_output(emotions)
    breakpoint()


