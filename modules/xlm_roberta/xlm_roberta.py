import numpy as np
import os
import torch
from discover_utils.interfaces.server_module import Processor
from discover_utils.utils.log_utils import log
from discover_utils.utils.type_definitions import SSINPDataType
from discover_utils.data.stream import SSIStream
from transformers import AutoTokenizer, XLMRobertaModel
from time import perf_counter

#TODO Seems to be not working at the moment
os.environ["PYTORCH_PRETRAINED_BERT_CACHE"] = os.getenv('CACHE_DIR', 'cache')
os.environ["PYTORCH_TRANSFORMERS_CACHE"] = os.getenv('CACHE_DIR', 'cache')
os.environ["HUGGINGFACE_HUB_CACHE"] = os.getenv('CACHE_DIR', 'cache')
os.environ["HF_MODULES_CACHE"] = os.getenv('CACHE_DIR', 'cache')

_default_options = {
    'batch_size' : 1
}

INPUT_ID = 'transcript'
OUTPUT_ID = 'embeddings'

class XLMRoBERTa(Processor):
    def __init__(self, *args, **kwargs):

        # Setting options
        super().__init__(*args, **kwargs)
        self.options = _default_options | self.options
        self.batch_size = int(self.options["batch_size"])

        # Build model
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        log(f"Using device {self._device}")

        self.tokenizer = AutoTokenizer.from_pretrained("FacebookAI/xlm-roberta-base")
        self.model = XLMRobertaModel.from_pretrained("FacebookAI/xlm-roberta-base").to(
            self._device
        )

    def preprocess_sample(self, sample):
        return sample

    def process_sample(self, sample):
        return sample

    def postprocess_sample(self, sample):
        return sample

    def _process_batch(self, batch):
        preprocessed = self.tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(self._device)
        with torch.no_grad():
            try:
                pred = self.model(**preprocessed)
            except:
                breakpoint()
        return pred

    def process_data(self, ds_iter) -> dict:
        self.ds_iterator = ds_iter
        current_batch = []
        output_embeddings = []
        t_start = perf_counter()

        # Placeholder array to keep sample order for predictions on garbage data
        # garbage_input = np.zeros(
        #     (
        #         int(
        #             (self.model_sr / 1000)
        #             * (
        #                     ds_iterator.left_context
        #                     + ds_iterator.frame_size
        #                     + ds_iterator.right_context
        #             )
        #         ),
        #     ),
        #     dtype=np.float32,
        # )
        # garbage_output = np.zeros((1024,), dtype=np.float32)

        for i, sample in enumerate(self.ds_iterator):
            sample_flat = ''.join(sample.get(INPUT_ID, [])).strip()
            current_batch.append(sample_flat)

            if i > 0 and i % self.batch_size == 0:
                outputs = self._process_batch(current_batch)#self.model(**inputs)
                pooled_output = outputs['pooler_output']
                output_embeddings.append(pooled_output)
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
        return output_embeddings

    def to_output(self, data) -> dict:
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

if __name__ == '__main__':
    import dotenv
    import os
    from discover_utils.utils.ssi_xml_utils import Trainer
    from pathlib import Path
    from discover_utils.data.provider.dataset_iterator import DatasetIterator

    env = dotenv.load_dotenv(r'../.env')
    IP = os.getenv("NOVA_IP", "")
    PORT = int(os.getenv("NOVA_PORT", 0))
    USER = os.getenv("NOVA_USER", "")
    PASSWORD = os.getenv("NOVA_PASSWORD", "")
    DATA_DIR = Path(os.getenv("NOVA_DATA_DIR", None))

    test_fd = os.getenv('TEST_DIR')
    test_file = Path(test_fd) / 'nova_server_test' / 'transcript.annotation'

    # Trainer
    xlm_roberta_trainer = Trainer()
    xlm_roberta_trainer.load_from_file("xlm_roberta.trainer")
    xlm_roberta = XLMRoBERTa(model_io=None, opts={}, trainer=xlm_roberta_trainer)

    # Inputs
    dd_input_transcript = {
        "src": "file:annotation:Free",
        "type": "input",
        "id": INPUT_ID,
        "uri": str(test_file),
    }

    # Output
    dd_output_embeddings = {
        "src": "file:stream:SSIStream",
        "type": "output",
        "id": OUTPUT_ID,
        "uri": str(test_file.parent / "xlm_roberta.stream"),
    }

    ds_iterator = DatasetIterator(
        data_description=[dd_input_transcript, dd_output_embeddings],
        frame_size=xlm_roberta_trainer.meta_frame_step,
        left_context=xlm_roberta_trainer.meta_left_ctx,
        start=0,

    )
    ds_iterator.load()
    emotions = xlm_roberta.process_data(ds_iterator)
    output = xlm_roberta.to_output(emotions)

    breakpoint()


