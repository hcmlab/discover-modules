import numpy as np
import requests
from discover_utils.interfaces.server_module import Processor
from discover_utils.data.annotation import DiscreteAnnotation
from discover_utils.utils.log_utils import log
import json


_default_options = {}
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

    def preprocess_sample(self, sample):
        return sample

    def process_sample(self, sample, system_prompt):
        request = {
            'system_prompt': system_prompt,
            'provider': self.provider,
            'model': self.model,
            'message': f'Predict the following data: <{sample}>. Value:\n',
            'temperature': 0,
            #'top_k': 1,
            #"top_p" : 0.01
            'resp_format' : 'json',
            'max_new_tokens': 1024,
            'enforce_determinism': True
        }
        with requests.post('http://' + self.ip + ':' + self.port + '/assist', json=request) as response:
            resp = response.content

        try:
            resp = json.loads(resp.decode('UTF-8'))
        except Exception as e:
            log (e)
            resp = {}

        return resp

    def process_data(self, ds_manager) -> np.ndarray:
        self.session_manager = self.get_session_manager(ds_manager, ignore_ambiguity=True)

        odt = self.session_manager.output_data_templates[OUTPUT_ID]
        if isinstance(odt, DiscreteAnnotation):
            annotation_scheme_name = odt.annotation_scheme.name
            class_ids = odt.annotation_scheme.classes
            annotation_scheme_description = odt.meta_data.description
            annotation_scheme_examples = odt.meta_data.examples


            response_schema_json = """Respond in JSON. Your response should be in the following format:
                ```json
                {
                  "label": "The label you assigned to the text segment.",
                  "confidence": "A float value on a continuous scale from 0.0 to 1.0 that indicates how confident you are in your prediction. 0.0 means your are just guessing and 1.0 means you are absolut certain.",
                  "explanation": "Explain why you assigned the label to the input."
                }
                ```
            """

            sp = f'You are a classifier that assigns labels to text segments.' \
                 f'You label every segment with respect to {annotation_scheme_name}. ' \
                 f'To assign the label chose one of the following categories:{", ".join([x["name"] for x in class_ids.values()])}.' \

            if annotation_scheme_description:
                 sp += f'Follow the following description of the labels:\n{annotation_scheme_description}'
            if annotation_scheme_examples:
                ex = "\n".join([x["value"] + " : " +  x["label"] for x in annotation_scheme_examples])
                sp += f'Use the following examples as a guideline for your labeling process:\n { ex }'

            sp += f'{response_schema_json}'
        else:
            sp = self.system_prompt + f'Respond in JSON. Only use one key called <label>.'

        data_object = self.session_manager.input_data[INPUT_ID]
        data_object_context = self.session_manager.input_data[INPUT_ID_CONTEXT]

        data = data_object.data
        data_context = data_object_context.data

        ret = []
        attributes = {
            a : [] for a in ATTRIBUTES
        }

        # TODO MAKE OPTION
        group_turns = True
        if group_turns:
            import pandas as pd
            transcript = pd.core.frame.DataFrame(data)
            context = pd.core.frame.DataFrame(data_context)
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
            resp = self.process_sample(d[2], system_prompt=sp)
            print(resp)

            if isinstance(odt, DiscreteAnnotation):
                label_candidates = [k for k,v in odt.annotation_scheme.classes.items() if v['name'] == resp.get('label')]
                if label_candidates:
                    label_id = label_candidates[0]
                else:
                    label_id = odt.rest_label_id
                label_id = int(label_id)
                conf = float(resp.get('confidence', 0.0))
                ret.append((d[0], d[1], label_id, conf))
            else:
                resp = resp.get('label', UNK)
                ret.append((d[0], d[1], resp, d[3]))

            for k,v in attributes.items():
                v.append(resp.get(k.lower(), ''))
        return ret, attributes

    def to_output(self, data) -> dict:
        # Append necessary meta information
        out = self.session_manager.output_data_templates[OUTPUT_ID]
        out.data, out.meta_data.attributes = data
        return self.session_manager.output_data_templates

if __name__ == '__main__':
    import dotenv
    import os
    from discover_utils.utils.ssi_xml_utils import Trainer
    from pathlib import Path
    from discover_utils.data.provider.dataset_iterator import DatasetIterator
    from discover_utils.utils.ssi_xml_utils import Trainer


    env = dotenv.load_dotenv(r'../.env')
    IP = os.getenv("NOVA_IP", "")
    PORT = int(os.getenv("NOVA_PORT", 0))
    USER = os.getenv("NOVA_USER", "")
    PASSWORD = os.getenv("NOVA_PASSWORD", "")
    DATA_DIR = Path(os.getenv("NOVA_DATA_DIR", None))
    ASSISTANT_IP = os.getenv("NOVA_ASSISTANT_IP", "")
    ASSISTANT_PORT = os.getenv("NOVA_ASSISTANT_PORT", 0)



# amh = AnnotationHandler(db_host=IP, db_port=PORT, db_user=USER, db_password=PASSWORD)

    # load
    #fs = "Loading {} took {}ms"
    #t_start = perf_counter()
    #discrete_anno = amh.load(
    #     dataset="test",
    #     scheme="emotion_categorical",
    #     annotator="schildom",
    #     session="01_AffWild2_video1",
    #     role="testrole2",
    #     header_only=True
    # )
    #t_stop = perf_counter()
    #print(fs.format("Discrete annotation", int((t_stop - t_start) * 1000)))

    # Trainer
    na_trainer = Trainer()
    na_trainer.load_from_file("nova_assistant_predict.trainer")
    opts = {
        'ip': ASSISTANT_IP,
        'port': ASSISTANT_PORT,
        'model': 'llama2',
        'provider': 'ollama'
    }
    na = LensFreePrompt(model_io=None, opts=opts, trainer=na_trainer)

    # Inputs
    transcript = {
        "src": "db:annotation:Free",
        "scheme": "transcript",
        "role": "testrole2",
        "annotator": "baurtobi",
        "type": "input",
        "id": INPUT_ID,

    }

    # Output
    sentiment = {
        "src": "db:annotation:Discrete",
        "scheme": "emotion_categorical",
        "role": "testrole2",
        "annotator": "baurtobi",
        "type": "output",
        "id": OUTPUT_ID,
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
        data_description=[transcript, sentiment],
        source_context=ctx,
        dataset='test',
        frame_size=na_trainer.meta_frame_step,
        left_context=na_trainer.meta_left_ctx,
        session_names=['01_AffWild2_video1']
    )
    ds_iterator.load()
    emotions = na.process_data(ds_iterator)
    output = na.to_output(emotions)


