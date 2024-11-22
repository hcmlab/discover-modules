import numpy as np
import requests
from discover_utils.interfaces.server_module import Processor
from discover_utils.data.annotation import DiscreteAnnotation
from discover_utils.utils.log_utils import log
import json


_default_options = {
    'group_turns': False,
}
_dim_labels = [
    "sentiment",
]

INPUT_ID = 'transcript'
INPUT_ID_CONTEXT = 'transcript_context'
OUTPUT_ID = 'output'
ATTRIBUTES = ['Explanation']
UNK = '-'

class LensFreePrompt(Processor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.options = _default_options | self.options
        self.ip = self.options['ip']
        self.port = str(self.options['port'])
        self.model = self.options['model']
        self.provider = self.options['provider']
        self.prompt = self.options.get('prompt', '')
        self.group_turns = self.options.get('group_turns', False)

    def preprocess_sample(self, sample):
        return sample

    def predict_sample(self, sample, prompt):
        request = {
            'system_prompt': '',
            'provider': self.provider,
            'model': self.model,
            'message': f'{prompt}. \n """{sample}""". Value:\n',
            'temperature': 0,
            #'top_k': 1,
            #"top_p" : 0.01
            'resp_format' : 'json',
            'max_new_tokens': 128,
            'enforce_determinism': True,
            'stream': True
        }
        with requests.post('http://' + self.ip + ':' + self.port + '/assist', json=request) as response:
            resp = response.content

        try:
            resp = json.loads(resp.decode('UTF-8'))
        except Exception as e:
            log(str(e))
            resp = {}

        return resp

    def process_data(self, ds_manager) -> np.ndarray:
        self.session_manager = self.get_session_manager(ds_manager, ignore_ambiguity=True)

        odt = self.session_manager.output_data_templates[OUTPUT_ID]
        if not self.prompt.endswith('.'):
            self.prompt = self.prompt + '.'
        prompt = self.prompt + f' Respond in JSON. Only use one key called "label".'
        data_object = self.session_manager.input_data[INPUT_ID]
        data = data_object.data

        ret = []
        attributes = {
            a : [] for a in ATTRIBUTES
        }

        if self.group_turns:
            data_object_context = self.session_manager.input_data[INPUT_ID_CONTEXT]
            data_context = data_object_context.data

            import pandas as pd
            transcript = pd.DataFrame(data)
            context = pd.DataFrame(data_context)
            transcript['id']  = data_object.meta_data.role
            context['id']  = data_object_context.meta_data.role
            df = pd.concat([transcript, context])
            df = df.sort_values('from')
            df['turn'] = df['id'].ne(df['id'].shift(1)).cumsum()
            df = df.groupby('turn').agg(
                {
                    'from' : 'min',
                    'to' : 'max',
                    'name' : ' '.join,
                    'conf' : 'min',
                    'id' : 'first'
                }
            )

            # Merge with previous context turn
            # Check if first row contains context
            if df.iloc[0]['id'] != data_object_context.meta_data.role:
                ctx = df.iloc[:1]
                ctx['name'] = ''
                df = pd.concat([ctx, df])

            # If last label is context drop it
            if df.iloc[-1]['id'][1] == data_object_context.meta_data.role:
                df = df.drop(df[-1:].index, axis=0)

            # Group every two rows
            df.reset_index()

            # Join role with transcript
            df['name'] = df[['id', 'name']].apply(lambda row: ': '.join(row.values.astype(str)), axis=1)

            df = df.groupby(np.arange(len(df)) // 2).agg(
                {
                    'from' : 'min',
                    'to' : 'max',
                    'name' : ' \n '.join,
                    'conf' : 'min',
                    'id' : 'first'
                }
            )
            data = df.to_numpy()

        for i,d in enumerate(data):
            resp = self.predict_sample(d[2], prompt=prompt)
            log(f'"{d[2]}" : {resp}')
            resp = resp.get('label', UNK)
            ret.append((d[0], d[1], resp, d[3]))

        return ret

    def to_output(self, data) -> dict:
        # Append necessary meta information
        out = self.session_manager.output_data_templates[OUTPUT_ID]
        #out.data, out.meta_data.attributes = data
        out.data = data
        return self.session_manager.output_data_templates

if __name__ == '__main__':
    import dotenv
    import os
    from discover_utils.utils.ssi_xml_utils import Trainer
    from pathlib import Path
    from discover_utils.data.provider.dataset_iterator import DatasetIterator
    from discover_utils.utils.ssi_xml_utils import Trainer


    env = dotenv.load_dotenv(r'../../.env')
    IP = os.getenv("NOVA_IP", "")
    PORT = int(os.getenv("NOVA_PORT", 0))
    USER = os.getenv("NOVA_USER", "")
    PASSWORD = os.getenv("NOVA_PASSWORD", "")
    DATA_DIR = Path(os.getenv("NOVA_DATA_DIR", None))
    LENS_IP = os.getenv("LENS_IP", "")
    LENS_PORT = os.getenv("LENS_PORT", 0)
    DISCOVER_OUT_DIR = Path(os.getenv("DISCOVER_OUT_DIR", None))

    # Trainer
    na_trainer = Trainer()
    na_trainer.load_from_file("lens_predict.trainer")
    opts = {
        'ip': LENS_IP,
        'port': LENS_PORT,
        'model': 'mistral-nemo',
        'provider': 'ollama',
        'prompt': 'translate the provided text to english',
        'group_turns': True,
    }
    na = LensFreePrompt(model_io=None, opts=opts, trainer=na_trainer)

    # Inputs
    transcript = {
        "src": "db:annotation:Free",
        "scheme": "transcript",
        "role": "patient",
        "annotator": "whisperx_segment",
        "type": "input",
        "id": INPUT_ID,
    }

    transcript_ctx = {
        "src": "db:annotation:Free",
        "scheme": "transcript",
        "role": "therapeut",
        "annotator": "whisperx_segment",
        "type": "input",
        "id": INPUT_ID_CONTEXT
    }


    # Output
    translation = {
        "src": "file:annotation:Free",
        "type": "output",
        "id": OUTPUT_ID,
        "uri": str(DISCOVER_OUT_DIR / 'Lens' / 'lens_predict.py.annotation'),
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
        data_description=[transcript, transcript_ctx, translation],
        source_context=ctx,
        dataset='therapai',
        frame_size=na_trainer.meta_frame_step,
        left_context=na_trainer.meta_left_ctx,
        session_names=['88Y8_S03']
    )
    ds_iterator.load()
    emotions = na.process_data(ds_iterator)
    output = na.to_output(emotions)

    breakpoint()


