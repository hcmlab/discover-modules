"""WhisperX Module
Author: Dominik Schiller <dominik.schiller@uni-a.de>, Tobias Hallmen <tobias.hallmen@uni-a.de>
Date: 18.10.2023

"""

from discover_utils.interfaces.server_module import Processor
import sys


INPUT_ID = "audio"
OUTPUT_ID = "transcript"

# Setting defaults
_default_options = {"model": "tiny", "alignment_mode": "segment", "batch_size": 16, 'language': None, 'compute_type': 'float16', 'vad_onset': 0.500, 'vad_offset': 0.363}

# supported language codes, cf. whisperx/alignment.py
# DEFAULT_ALIGN_MODELS_TORCH.keys() | DEFAULT_ALIGN_MODELS_HF.keys() | {None} ==
# {'nl', 'tr', 'es', 'cs', 'de', 'en', 'ko', 'zh', 'he', 'fr', 'ca', 'ru', 'el', 'uk', 'fi', 'it', 'pt', 'ar', 'ur',
# 'te', 'fa', None, 'hu', 'vi', 'hi', 'pl', 'da', 'ja', 'sk', 'sl', 'hr'}

# TODO: add log infos, 
#  add whisperx' diarisation? no, it's just pyannote
#  apparently whisperx is also a dead project by now (october 2024)
class WhisperX(Processor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.options = _default_options | self.options
        
        # string treatment for number extraction
        try:
            self.options['batch_size'] = int(self.options['batch_size'])
        except ValueError:
            print(f'batch_size field contains invalid characters - using default {_default_options["batch_size"]}')
            self.options['batch_size'] = _default_options['batch_size']
        try:
            self.options['vad_onset'] = float(self.options['vad_onset'])
        except ValueError:
            print(f'vad_onset field contains invalid characters - using default {_default_options["vad_onset"]}')
            self.options['vad_onset'] = _default_options['vad_onset']
        try:
            self.options['vad_offset'] = float(self.options['vad_offset'])
        except ValueError:
            print(f'vad_offset field contains invalid characters - using default {_default_options["vad_offset"]}')
            self.options['vad_offset'] = _default_options['vad_offset']
        
        self.options = _default_options | self.options
        
        # expand string shorthands
        if self.options['language'] == 'auto':
            self.options['language'] = None
        if self.options['model'] == 'large-v3-turbo':
            self.options['model'] = 'deepdml/faster-whisper-large-v3-turbo-ct2'
        
        self.device = None
        self.session_manager = None

    def process_data(self, ds_manager) -> dict:
        import whisperx
        import torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.session_manager = self.get_session_manager(ds_manager)
        input_audio = self.session_manager.input_data['audio']

        # sliding window will be applied by WhisperX
        audio = whisperx.load_audio(input_audio.meta_data.file_path)

        # transcribe with original whisper
        try:
            model = whisperx.load_model(self.options["model"], self.device, compute_type=self.options['compute_type'],
                                        language=self.options['language'], vad_options={'vad_onset': self.options['vad_onset'], 'vad_offset': self.options['vad_offset']})
        except ValueError:
            print(f'Your device {self.device} does not support {self.options["compute_type"]} - fallback to float32')
            sys.stdout.flush()
            model = whisperx.load_model(self.options["model"], self.device, compute_type='float32',
                                        language=self.options['language'], vad_options={'vad_onset': self.options['vad_onset'], 'vad_offset': self.options['vad_offset']})
            
        result = model.transcribe(audio, batch_size=self.options["batch_size"])

        # delete model if low on GPU resources
        import gc; gc.collect(); torch.cuda.empty_cache(); del model

        if not self.options["alignment_mode"] == "raw":
            # load alignment model and metadata
            model_a, metadata = whisperx.load_align_model(
                language_code=result["language"], device=self.device
            )

            # align whisper output
            result_aligned = whisperx.align(
                result["segments"], model_a, metadata, audio, self.device
            )
            result = result_aligned

            # delete model if low on GPU resources
            import gc; gc.collect(); torch.cuda.empty_cache(); del model_a

        return result

    def to_output(self, data: dict):
        def _fix_missing_timestamps(data):
            """
            https://github.com/m-bain/whisperX/issues/253
            Some characters might miss timestamps and recognition scores. This function adds estimated time stamps assuming a fixed time per character of 65ms.
            Confidence for each added timestamp will be 0.
            Args:
                data (dictionary): output dictionary as returned by process_data
            """
            last_end = 0
            for s in data["segments"]:
                for w in s["words"]:
                    if "end" in w.keys():
                        last_end = w["end"]
                    else:
                        #TODO: rethink lower bound for confidence; place word centred instead of left aligned
                        w["start"] = last_end
                        last_end += 0.065
                        w["end"] = last_end
                        #w["score"] = 0.000
                        w['score'] = _hmean([x['score'] for x in s['words'] if len(x) == 4])
        
        def _hmean(scores):
            if len(scores) > 0:
                prod = scores[0]
                for s in scores[1:]:
                    prod *= s
                prod = prod**(1/len(scores))
            else:
                prod = 0
            return prod
        
        if (
            self.options["alignment_mode"] == "word"
            or self.options["alignment_mode"] == "segment"
        ):
            _fix_missing_timestamps(data)

        if self.options["alignment_mode"] == "word":
            anno_data = [
                (w["start"], w["end"], w["word"], w["score"])
                for w in data["word_segments"]
            ]
        else:
            anno_data = [
                #(w["start"], w["end"], w["text"], _hmean([x['score'] for x in w['words']])) for w in data["segments"]
                (w["start"], w["end"], w["text"], 1) for w in data["segments"]  # alignment 'raw' no longer contains a score(?)
            ]

        # convert to milliseconds
        anno_data = [(x[0]*1000, x[1]*1000, x[2], x[3]) for x in anno_data]
        out = self.session_manager.output_data_templates[OUTPUT_ID]
        out.data = anno_data
        return self.session_manager.output_data_templates
