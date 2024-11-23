import numpy as np
import requests
from discover_utils.interfaces.server_module import Processor
from discover_utils.utils.log_utils import log
import sys
import os
# Add local dir to path for relative imports
sys.path.insert(0, os.path.dirname(__file__))
import json
import templates

_default_options = {
    'language' : 'en',
    'group_turns' : False,
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
        self.system_prompt = self.options.get('system_prompt')
        self.prompt = self.options.get('prompt')
        self.lang = self.options.get('language')
        self.group_turns = self.options.get('group_turns')

    def preprocess_sample(self, sample):
        return sample

    def process_sample(self, message, system_prompt):
        request = {
            'system_prompt': system_prompt,
            'provider': self.provider,
            'model': self.model,
            'message': message,
            'temperature': 0,
            # 'top_k': 1,
            # "top_p" : 0.01
            'resp_format': 'json',
            'max_new_tokens': 1024,
            'enforce_determinism': True
        }
        with requests.post('http://' + self.ip + ':' + self.port + '/assist', json=request) as response:
            resp = response.content

        try:
            resp = json.loads(resp.decode('UTF-8'))
        except Exception as e:
            log(e)
            resp = {}

        return resp

    def process_data(self, ds_manager) -> np.ndarray:
        self.session_manager = self.get_session_manager(ds_manager, ignore_ambiguity=True)

        odt = self.session_manager.output_data_templates[OUTPUT_ID]
        main_role = self.session_manager.input_data[INPUT_ID].meta_data.role
        annotation_scheme_name = odt.annotation_scheme.name
        class_ids = odt.annotation_scheme.classes
        annotation_scheme_description = odt.meta_data.description
        annotation_scheme_examples = odt.meta_data.examples

        response_schema_json = templates.response_scheme_template(self.lang)
        system_prompt = templates.system_prompt_template(self.lang, annotation_scheme_name, [x["name"] for x in class_ids.values()])
        description = ''
        examples = ''
        if annotation_scheme_description:
            description = templates.description_template(self.lang, annotation_scheme_description)
        if annotation_scheme_examples:
            examples = templates.example_template(self.lang, [x["value"] + " : " + x["label"] for x in annotation_scheme_examples])

        system_prompt = '\n'.join([system_prompt, description, examples, response_schema_json])

        data_object = self.session_manager.input_data[INPUT_ID]

        data = data_object.data

        ret = []
        attributes = {
            a: [] for a in ATTRIBUTES
        }

        if self.group_turns:

            data_object_context = self.session_manager.input_data[INPUT_ID_CONTEXT]
            data_context = data_object_context.data

            import pandas as pd
            transcript = pd.core.frame.DataFrame(data)
            context = pd.core.frame.DataFrame(data_context)
            transcript['id'] = data_object.meta_data.role
            context['id'] = data_object_context.meta_data.role
            df = pd.concat([transcript, context])
            df = df.sort_values('from')
            df['turn'] = df['id'].ne(df['id'].shift(1)).cumsum()
            df = df.groupby('turn').agg(
                {
                    'from': 'min',
                    'to': 'max',
                    'name': ' '.join,
                    'conf': 'min',
                    'id': 'first'
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
                    'from': 'min',
                    'to': 'max',
                    'name': ' \n '.join,
                    'conf': 'min',
                    'id': 'first'
                }
            )
            data = df.to_numpy()

        for i, d in enumerate(data):
            resp = self.process_sample(message=templates.message_template(self.lang, d[2], main_role=main_role), system_prompt=system_prompt)
            print(resp)

            label_candidates = [k for k, v in odt.annotation_scheme.classes.items() if v['name'] == resp.get('label')]
            if label_candidates:
                label_id = label_candidates[0]
            else:
                label_id = odt.rest_label_id
            label_id = int(label_id)
            conf = float(resp.get('confidence', 0.0))
            ret.append((d[0], d[1], label_id, conf))

            for k, v in attributes.items():
                v.append(resp.get(k.lower(), ''))
        return ret, attributes

    def to_output(self, data) -> dict:
        # Append necessary meta information
        out = self.session_manager.output_data_templates[OUTPUT_ID]
        out.data, out.meta_data.attribute_values = data
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
    lens_trainer = Trainer()
    lens_trainer.load_from_file("lens_predict.trainer")
    opts = {
        'ip': LENS_IP,
        'port': LENS_PORT,
        'model': 'llama2',
        'provider': 'ollama'
    }
    lens = LensFreePrompt(model_io=None, opts=opts, trainer=lens_trainer)

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

    # Output
    sentiment = {
        "src": "db:annotation:Discrete",
        "scheme": "sentiment",
        "role": "patient",
        "annotator": "schildom",
        "type": "output",
        "id": OUTPUT_ID
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
        data_description=[transcript, transcript_ctx, sentiment],
        source_context=ctx,
        dataset='therapai',
        frame_size=lens_trainer.meta_frame_step,
        left_context=lens_trainer.meta_left_ctx,
        session_names=['88Y8_S03']
    )
    ds_iterator.load()
    emotions = lens.process_data(ds_iterator)
    output = lens.to_output(emotions)
