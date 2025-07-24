"""CrisperWhisper Module
Author: Tobias Hallmen <tobias.hallmen@uni-a.de>
Date: 23.07.2025
Based on: https://github.com/nyrahealth/CrisperWhisper

CrisperWhisper provides advanced speech recognition with verbatim transcriptions 
and precise word-level timestamps, minimizing hallucinations.
"""

import sys
import os
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from discover_utils.interfaces.server_module import Processor
from discover_utils.utils.log_utils import log

INPUT_ID = "audio"
OUTPUT_ID = "transcript"

MEDIA_TYPE_ID = "annotation:Free:transcript"

_default_options = {
    "model_id": "nyrahealth/CrisperWhisper",
    "batch_size": 1,
    "chunk_length_s": 30,
    "device": "auto",
    "torch_dtype": "auto",
    "language": "de",
    "return_timestamps": "word",
    "confidence": 1.0,
    "segmentation": "word",
}

class CrisperWhisper(Processor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.options = _default_options | self.options
        self.device = None
        self.model = None
        self.processor = None
        self.pipeline = None
        self.session_manager = None
        self.nlp_models = {}

    def _setup_device_and_dtype(self):
        if self.options["device"] == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = self.options["device"]
            
        if self.options["torch_dtype"] == "auto":
            if self.device == "cuda" and torch.cuda.is_available():
                self.torch_dtype = torch.float16
            else:
                self.torch_dtype = torch.float32
        else:
            self.torch_dtype = getattr(torch, self.options["torch_dtype"])

    def _load_model(self):
        from transformers.models.whisper import generation_whisper
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
        from utils import _patched_extract_token_timestamps
        generation_whisper.WhisperGenerationMixin._extract_token_timestamps = _patched_extract_token_timestamps
        self._setup_device_and_dtype()
        
        log(f"Loading CrisperWhisper model on {self.device} with dtype {self.torch_dtype}")
        
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.options["model_id"],
            torch_dtype=self.torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
            use_cache=False,
            attn_implementation="eager"
        )
        self.model.to(self.device)
        
        self.processor = AutoProcessor.from_pretrained(self.options["model_id"])
        
        self.pipeline = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            #max_new_tokens=128,
            chunk_length_s=self.options["chunk_length_s"],
            batch_size=self.options["batch_size"],
            return_timestamps=self.options["return_timestamps"],
            torch_dtype=self.torch_dtype,
            device=self.device,
            generate_kwargs={'language': f'<|{self.options["language"]}|>'}
        )

    def _get_spacy_sentencizer(self, language):
        if language not in self.nlp_models:
            try:
                import spacy
                self.nlp_models[language] = spacy.blank(language)
                self.nlp_models[language].add_pipe("sentencizer")
                log(f"Loaded spaCy sentencizer for language: {language}")
            except Exception as e:
                log(f"Failed to load spaCy sentencizer for {language}: {str(e)}")
                self.nlp_models[language] = None
        return self.nlp_models[language]

    def _apply_sentence_segmentation(self, result):
        if self.options["segmentation"] != "sentence":
            return result
            
        language = self.options["language"]
        full_text = result.get("text", "")
        
        if not full_text.strip():
            return result
            
        nlp = self._get_spacy_sentencizer(language)
        if nlp is None:
            log("spaCy sentencizer not available, falling back to word segmentation")
            return result
            
        doc = nlp(full_text)
        sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
        
        if not sentences:
            return result
            
        # Create sentence-level chunks by combining word-level timestamps
        word_chunks = result.get("chunks", [])
        if not word_chunks:
            return {"text": full_text, "chunks": []}
            
        sentence_chunks = []
        word_idx = 0
        
        for sentence in sentences:
            sentence_words = sentence.split()
            if not sentence_words:
                continue
                
            # Find matching word chunks for this sentence
            sentence_word_chunks = []
            remaining_words = len(sentence_words)
            
            while word_idx < len(word_chunks) and remaining_words > 0:
                chunk = word_chunks[word_idx]
                chunk_text = chunk["text"].strip()
                
                # Check if this chunk's text matches any word in our sentence
                if any(word.lower() in chunk_text.lower() or chunk_text.lower() in word.lower() 
                       for word in sentence_words):
                    sentence_word_chunks.append(chunk)
                    remaining_words -= len(chunk_text.split())
                
                word_idx += 1
                
                # Break if we've likely covered all words in the sentence
                if remaining_words <= 0:
                    break
            
            if sentence_word_chunks:
                # Create sentence chunk with combined timestamps
                start_time = sentence_word_chunks[0]["timestamp"][0]
                end_time = sentence_word_chunks[-1]["timestamp"][1]
                
                sentence_chunks.append({
                    "text": sentence,
                    "timestamp": [start_time, end_time]
                })
        
        return {"text": full_text, "chunks": sentence_chunks}

    def process_data(self, ds_manager) -> dict:
        if self.pipeline is None:
            self._load_model()
            
        self.session_manager = self.get_session_manager(ds_manager)
        input_audio = self.session_manager.input_data['audio']
        
        log(f"Processing audio file: {input_audio.meta_data.file_path}")
        
        # TODO: Remove exception handling - let errors propagate to DISCOVER for ungraceful job termination
        try:
            from utils import _adjust_pauses_for_hf_pipeline_output
            # Use file path instead of numpy data for better compatibility
            result = self.pipeline(str(input_audio.meta_data.file_path))
            result = _adjust_pauses_for_hf_pipeline_output(result)
            
            # Apply sentence segmentation if requested
            result = self._apply_sentence_segmentation(result)
            
            log(f"Transcription completed. Text length: {len(result.get('text', ''))}")
            
            return result
            
        except Exception as e:
            log(f"Error during transcription: {str(e)}")
            return {"text": "", "chunks": []}  # TODO: Remove this fallback, let exception propagate

    def to_output(self, data: dict):
        annotation = self.session_manager.output_data_templates[OUTPUT_ID]
        annotation.data = np.array([(int(x['timestamp'][0]*1000), int(x['timestamp'][1]*1000), x['text'], self.options['confidence']) for x in data['chunks']], dtype=annotation.annotation_scheme.label_dtype)        
        return self.session_manager.output_data_templates


if __name__ == "__main__":
    from discover_utils.utils.ssi_xml_utils import Trainer
    from discover_utils.data.provider.data_manager import DatasetManager
    from pathlib import Path
    import tempfile
    import wave
    
    print("Testing CrisperWhisper module...")
    
    try:
        # Load trainer configuration
        trainer = Trainer()
        trainer.load_from_file("crisperwhisper.trainer")
        
        # Create CrisperWhisper processor with proper initialization
        processor = CrisperWhisper(model_io=None, opts={}, trainer=trainer)
        
        # Generate test audio
        duration = 5
        sample_rate = 16000
        samples = np.random.randn(duration * sample_rate).astype(np.float32)
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            # Write WAV file using wave module (built-in)
            with wave.open(tmp_file.name, 'wb') as wav_file:
                wav_file.setnchannels(1)  # mono
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(sample_rate)
                # Convert float32 to int16
                samples_int16 = (samples * 32767).astype(np.int16)
                wav_file.writeframes(samples_int16.tobytes())
            
            # Create data manager for test
            dd_input_audio = {
                "src": "file:stream:audio",
                "type": "input",
                "id": INPUT_ID,
                "uri": tmp_file.name,
            }
            
            dd_output_transcript = {
                "src": "file:annotation:free",
                "type": "output", 
                "id": OUTPUT_ID,
                "uri": str(Path(tmp_file.name).parent / "test_transcript.annotation")
            }
            
            dm = DatasetManager([dd_input_audio, dd_output_transcript])
            dm.load()
            
            # Process the data
            result = processor.process_data(dm)
            output = processor.to_output(result)
            
            print(f"Transcription result: {result.get('text', 'No text generated')}")
            print("CrisperWhisper module test completed successfully!")
            
    except Exception as e:
        print(f"Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Clean up temp file
        try:
            if 'tmp_file' in locals():
                os.unlink(tmp_file.name)
                os.unlink(str(Path(tmp_file.name).parent / "test_transcript.annotation"))
        except:
            pass