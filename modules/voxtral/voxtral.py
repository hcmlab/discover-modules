""" Voxtral Transcription

Author:
    Tobias Hallmen <tobias.hallmen@uni-a.de>
Date:
    11.08.2025

"""

from discover_utils.interfaces.server_module import Processor
from discover_utils.utils.log_utils import log
import sys


INPUT_ID = "audio"
OUTPUT_ID = "transcript"

# Setting defaults
_default_options = {"model": "mistralai/Voxtral-Mini-3B-2507", 'language': None, 'compute_type': 'bfloat16'}

# Voxtral:
# Natively multilingual: Automatic language detection and state-of-the-art performance in the world’s most widely used languages
# (English, Spanish, French, Portuguese, Hindi, German, Dutch, Italian)

# apparently it is using WhisperTokenizer, so maybe supported language codes are identical, cf. whisperx/alignment.py
# DEFAULT_ALIGN_MODELS_TORCH.keys() | DEFAULT_ALIGN_MODELS_HF.keys() | {None} ==
# {'nl', 'tr', 'es', 'cs', 'de', 'en', 'ko', 'zh', 'he', 'fr', 'ca', 'ru', 'el', 'uk', 'fi', 'it', 'pt', 'ar', 'ur',
# 'te', 'fa', None, 'hu', 'vi', 'hi', 'pl', 'da', 'ja', 'sk', 'sl', 'hr'}


# TODO: add log infos, 
class Voxtral(Processor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.options = _default_options | self.options
        
        import torch
        
        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            raise ValueError("No CUDA enabled GPU found - aborting")
        
        # set dtypes
        if self.options['compute_type'] == 'bfloat16':
            self.options['compute_type'] = torch.bfloat16
        elif self.options['compute_type'] == 'float16':
            self.options['compute_type'] = torch.float16
        else:
            raise ValueError(f'Unsupported compute_type {self.options['compute_type']}')

        # expand string shorthands
        if self.options['language'] == 'auto':
            self.options['language'] = None
        if self.options['model'] == 'voxtral-mini-3b':
            self.options['model'] = 'mistralai/Voxtral-Mini-3B-2507'
        elif self.options['model'] == 'voxtral-small-24b':
            self.options['model'] = 'mistralai/Voxtral-Small-24B-2507'
        
        self.session_manager = None
        self.length = None

    def process_data(self, ds_manager) -> dict:
        from transformers import VoxtralForConditionalGeneration, AutoProcessor
        import torch

        self.session_manager = self.get_session_manager(ds_manager)
        meta_data = self.session_manager.input_data[INPUT_ID].meta_data
        input_audio = str(meta_data.file_path)
        self.length = meta_data.duration if meta_data.duration else meta_data.num_samples / meta_data.sample_rate
        
        log('Loading Audio Processor..')
        processor = AutoProcessor.from_pretrained(self.options['model'])

        try:
            log(f'Loading Model in {self.options['compute_type']}..')
            model = VoxtralForConditionalGeneration.from_pretrained(self.options['model'], torch_dtype=self.options['compute_type'], device_map=self.device)
        except ValueError:  # TODO: cant test this, dont have old enough GPU
            log(f'Your device {self.device} does not support {self.options["compute_type"]} - fallback to float16')
            self.options['compute_type'] = torch.float16
            model = VoxtralForConditionalGeneration.from_pretrained(self.options['model'], torch_dtype=self.options['compute_type'], device_map=self.device)
            
        inputs = processor.apply_transcription_request(language=[self.options['language']], audio=[input_audio], model_id=self.options['model'])  # expects equal long lists of languages and audios
        inputs = inputs.to(self.device, dtype=self.options['compute_type'])

        log('Transcribing..')
        outputs = model.generate(**inputs, max_new_tokens=torch.inf)
        decoded_outputs = processor.batch_decode(outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]

        return decoded_outputs

    def to_output(self, data: str):
        # convert to milliseconds
        # from, to, text, conf
        out = self.session_manager.output_data_templates[OUTPUT_ID]
        out.data = [(0, self.length, data, 1.0)]
        return self.session_manager.output_data_templates
