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
_default_options = {"model": "mistralai/Voxtral-Mini-3B-2507", 'language': None, 'compute_type': 'bfloat16', 'verbose': False}

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
        self.length = meta_data.duration if meta_data.duration else meta_data.num_samples / meta_data.sample_rate
        input_audio = self.session_manager.input_data[INPUT_ID].data
        
        log('Loading Audio Processor..')
        processor = AutoProcessor.from_pretrained(self.options['model'])

        try:
            log(f'Loading Model in {self.options['compute_type']}..')
            model = VoxtralForConditionalGeneration.from_pretrained(self.options['model'], torch_dtype=self.options['compute_type'], device_map=self.device)
        except ValueError:  # TODO: cant test this, dont have old enough GPU
            log(f'Your device {self.device} does not support {self.options["compute_type"]} - fallback to float16')
            self.options['compute_type'] = torch.float16
            model = VoxtralForConditionalGeneration.from_pretrained(self.options['model'], torch_dtype=self.options['compute_type'], device_map=self.device)

        # Create 30-minute chunks with 15s overlap for long audio processing
        sample_rate = meta_data.sample_rate
        chunk_size = 1800 * sample_rate  # 30 minutes (1800 seconds)
        overlap_size = 15 * sample_rate  # 15 seconds overlap
        
        chunks = []
        chunk_info = []
        
        # Generate overlapping chunks
        start = 0
        chunk_idx = 0
        while start < len(input_audio):
            end = min(start + chunk_size, len(input_audio))
            
            # Check if current chunk already reaches the end
            if end >= len(input_audio):
                # Current chunk covers everything, no need for additional chunks
                chunk_audio = input_audio[start:end]
                chunks.append(chunk_audio)
                start_time = start / sample_rate
                end_time = end / sample_rate
                chunk_info.append({
                    'chunk_idx': chunk_idx,
                    'start_time': start_time,
                    'end_time': end_time,
                    'duration': (end - start) / sample_rate
                })
                if self.options['verbose']:
                    log(f'Created final chunk {chunk_idx}: {start_time:.1f}s - {end_time:.1f}s ({len(chunk_audio)/sample_rate:.1f}s)')
                break
            
            # Check if the next chunk would be too small
            next_start = start + chunk_size - overlap_size
            next_end = min(next_start + chunk_size, len(input_audio))
            next_chunk_duration = (next_end - next_start) / sample_rate
            
            if next_start < len(input_audio) and next_chunk_duration < (2 * overlap_size / sample_rate):
                # Next chunk would be too small, extend current chunk or create properly sized final chunk
                if (len(input_audio) - start) <= chunk_size:
                    # Extend current chunk to cover everything
                    end = len(input_audio)
                    chunk_audio = input_audio[start:end]
                    chunks.append(chunk_audio)
                    start_time = start / sample_rate
                    end_time = end / sample_rate
                    chunk_info.append({
                        'chunk_idx': chunk_idx,
                        'start_time': start_time,
                        'end_time': end_time,
                        'duration': (end - start) / sample_rate
                    })
                    if self.options['verbose']:
                        log(f'Created extended final chunk {chunk_idx}: {start_time:.1f}s - {end_time:.1f}s ({len(chunk_audio)/sample_rate:.1f}s)')
                    break
                else:
                    # Create current chunk normally, then create properly sized final chunk
                    chunk_audio = input_audio[start:end]
                    chunks.append(chunk_audio)
                    start_time = start / sample_rate
                    end_time = end / sample_rate
                    chunk_info.append({
                        'chunk_idx': chunk_idx,
                        'start_time': start_time,
                        'end_time': end_time,
                        'duration': (end - start) / sample_rate
                    })
                    if self.options['verbose']:
                        log(f'Created chunk {chunk_idx}: {start_time:.1f}s - {end_time:.1f}s ({len(chunk_audio)/sample_rate:.1f}s)')
                    
                    # Create final chunk starting earlier to ensure it's at least 2*overlap_size
                    chunk_idx += 1
                    min_chunk_duration = 2 * overlap_size / sample_rate
                    final_start = max(0, len(input_audio) - min_chunk_duration * sample_rate)
                    final_start = min(final_start, end - overlap_size)  # Ensure some overlap with previous chunk
                    final_start = int(final_start)  # Convert to integer for slicing
                    
                    final_chunk_audio = input_audio[final_start:len(input_audio)]
                    chunks.append(final_chunk_audio)
                    
                    final_start_time = final_start / sample_rate
                    final_end_time = len(input_audio) / sample_rate
                    chunk_info.append({
                        'chunk_idx': chunk_idx,
                        'start_time': final_start_time,
                        'end_time': final_end_time,
                        'duration': len(final_chunk_audio) / sample_rate
                    })
                    if self.options['verbose']:
                        log(f'Created final chunk {chunk_idx}: {final_start_time:.1f}s - {final_end_time:.1f}s ({len(final_chunk_audio)/sample_rate:.1f}s)')
                    break
            
            chunk_audio = input_audio[start:end]
            
            # Store chunk with timing info
            chunks.append(chunk_audio)
            start_time = start / sample_rate
            end_time = end / sample_rate
            chunk_info.append({
                'chunk_idx': chunk_idx,
                'start_time': start_time,
                'end_time': end_time,
                'duration': (end - start) / sample_rate
            })
            
            log(f'Created chunk {chunk_idx}: {start_time:.1f}s - {end_time:.1f}s ({len(chunk_audio)/sample_rate:.1f}s)')
            
            # Move to next chunk with overlap
            if end >= len(input_audio):
                break
            start += chunk_size - overlap_size
            chunk_idx += 1
        
        # Process all chunks
        log(f'Processing {len(chunks)} chunks...')
        transcriptions = []
        
        for i, chunk_audio in enumerate(chunks):
            log(f'Transcribing chunk {i+1}/{len(chunks)}...')
            
            # Create format list based on file extension
            format_ext = meta_data.file_path.suffix[1:] if meta_data.file_path.suffix else 'wav'
            
            inputs = processor.apply_transcription_request(
                language=[self.options['language']], 
                audio=[chunk_audio], 
                format=[format_ext],
                sampling_rate=meta_data.sample_rate, 
                model_id=self.options['model']
            )
            inputs = inputs.to(self.device, dtype=self.options['compute_type'])
            
            outputs = model.generate(**inputs, max_new_tokens=torch.inf)
            chunk_transcript = processor.batch_decode(outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
            
            transcriptions.append({
                'chunk_idx': i,
                'start_time': chunk_info[i]['start_time'],
                'end_time': chunk_info[i]['end_time'],
                'text': chunk_transcript.strip()
            })
            
            if self.options['verbose']:
                log(f'Chunk {i}: "{chunk_transcript}"')
        
        # Stitch transcriptions together
        if len(chunks) > 1:
            log('Stitching chunks together...')
        stitched_text = self._stitch_transcriptions(transcriptions)
        
        # For debugging: also return detailed chunk information
        debug_info = {
            'total_chunks': len(chunks),
            'chunk_size': chunk_size / sample_rate,
            'overlap_size': overlap_size / sample_rate,
            'transcriptions': transcriptions,
            'stitched_text': stitched_text
        }
        
        return stitched_text

    def _stitch_transcriptions(self, transcriptions):
        """Stitch overlapping transcriptions together using word-based overlap detection."""
        if not transcriptions:
            return ""
        
        if len(transcriptions) == 1:
            return transcriptions[0]['text']
        
        result = transcriptions[0]['text']
        
        for i in range(1, len(transcriptions)):
            prev_text = result
            curr_text = transcriptions[i]['text']
            
            # Check if this is the final merge and if last chunk is small
            is_final = (i == len(transcriptions) - 1)
            if is_final:
                # Check if last chunk duration is <= 2*overlap_size (30s for 15s overlap)
                last_chunk = transcriptions[-1]
                chunk_duration = last_chunk['end_time'] - last_chunk['start_time']
                needs_special_treatment = chunk_duration <= (2 * 15)  # 2 * overlap_size
                if self.options['verbose']:
                    log(f'Final chunk duration: {chunk_duration:.1f}s, needs special treatment: {needs_special_treatment}')
            else:
                needs_special_treatment = False
            
            # Find and merge overlap
            merged = self._merge_overlapping_texts(prev_text, curr_text, is_final_merge=needs_special_treatment)
            result = merged
            
            if self.options['verbose']:
                log(f'Merging chunk {i-1} and {i}:')
                log(f'  Previous end: "...{prev_text.split()[-10:]}"')
                log(f'  Current start: "{curr_text.split()[:10]}..."')
                log(f'  Merged: overlap found and merged')
        
        return result
    
    def _merge_overlapping_texts(self, text1, text2, is_final_merge=False):
        """Merge two texts by trimming boundary words and finding substring matches."""
        words1 = text1.split()
        words2 = text2.split()
        
        if not words1:
            return text2
        if not words2:
            return text1
        
        # Try progressively larger trim amounts until we find a bridge
        if is_final_merge:
            # For final merge, be much more aggressive since last chunk may have limited content
            max_trim = min(len(words1) - 5, len(words2) - 5, 40)  # Leave at least 5 words, go up to 40
        else:
            max_trim = min(len(words1) // 2, len(words2) // 2, 32)  # Standard limit
        
        for trim_amount in range(4, max_trim + 1, 4):  # Start with 4, then 8, 12, 16, 20, 24, 28, 32...
            if self.options['verbose']:
                log(f'    Trying trim amount: {trim_amount}')
            
            # Get trimmed portions
            if len(words1) > trim_amount:
                text1_core = words1[:-trim_amount]  # Remove last N words
                text1_boundary = words1[-trim_amount:]  # Last N words
            else:
                text1_core = words1[:len(words1)//2] if len(words1) > 2 else []
                text1_boundary = words1[len(words1)//2:] if len(words1) > 2 else words1
            
            if len(words2) > trim_amount:
                text2_core = words2[trim_amount:]  # Remove first N words  
                text2_boundary = words2[:trim_amount]  # First N words
            else:
                text2_core = words2[len(words2)//2:] if len(words2) > 2 else []
                text2_boundary = words2[:len(words2)//2] if len(words2) > 2 else words2
            
            if self.options['verbose']:
                log(f'      Text1 boundary (last {len(text1_boundary)} words): {" ".join(text1_boundary)}')
                log(f'      Text2 boundary (first {len(text2_boundary)} words): {" ".join(text2_boundary)}')
            
            # Look for substring matches between the boundary regions
            bridge_text = self._find_bridge_text(text1_boundary, text2_boundary)
            
            if bridge_text:
                # Use bridge text to connect core portions
                merged = ' '.join(text1_core + bridge_text.split() + text2_core)
                if self.options['verbose']:
                    log(f'    ✓ Found bridge with trim {trim_amount}: "{bridge_text}"')
                return merged
        
        # No bridge found with any trim amount, use simple joining with conservative trim
        trim_amount = 4
        text1_core = words1[:-trim_amount] if len(words1) > trim_amount else words1[:-2] if len(words1) > 2 else []
        text1_boundary = words1[-trim_amount:] if len(words1) > trim_amount else words1[-2:] if len(words1) > 2 else words1
        text2_core = words2[trim_amount:] if len(words2) > trim_amount else words2[2:] if len(words2) > 2 else []
        text2_boundary = words2[:trim_amount] if len(words2) > trim_amount else words2[:2] if len(words2) > 2 else words2
        
        merged = ' '.join(text1_core + text1_boundary + text2_boundary + text2_core)
        if self.options['verbose']:
            log(f'    ✗ No bridge found with any trim amount, joining boundary regions')
        
        return merged
    
    def _find_bridge_text(self, boundary1, boundary2):
        """Find connecting text between two boundary word lists."""
        text1 = ' '.join(boundary1)
        text2 = ' '.join(boundary2)
        
        # Look for common substrings of reasonable length
        min_length = 10  # Minimum 10 characters
        max_length = min(len(text1), len(text2))
        
        best_match = ""
        
        # Try substrings from text1 and see if they appear in text2
        for start in range(len(text1)):
            for end in range(start + min_length, min(start + max_length, len(text1)) + 1):
                substring = text1[start:end]
                if substring.lower() in text2.lower():
                    if len(substring) > len(best_match):
                        best_match = substring
        
        return best_match.strip() if best_match else None
    
    def _calculate_word_similarity(self, words1, words2):
        """Calculate similarity between two word sequences."""
        if len(words1) != len(words2):
            return 0
        
        if not words1:
            return 1
        
        # Simple word-by-word similarity
        matches = 0
        for w1, w2 in zip(words1, words2):
            # Exact match
            if w1.lower() == w2.lower():
                matches += 1
            # Partial match for similar words
            elif self._words_similar(w1, w2):
                matches += 0.5
        
        return matches / len(words1)
    
    def _words_similar(self, word1, word2):
        """Check if two words are similar (handles minor transcription differences)."""
        w1, w2 = word1.lower(), word2.lower()
        
        # Same length words with small differences
        if len(w1) == len(w2) and len(w1) > 3:
            diff_count = sum(c1 != c2 for c1, c2 in zip(w1, w2))
            return diff_count <= 1
        
        # Different length but similar
        if abs(len(w1) - len(w2)) == 1 and max(len(w1), len(w2)) > 3:
            shorter, longer = (w1, w2) if len(w1) < len(w2) else (w2, w1)
            return shorter in longer
        
        return False

    def to_output(self, data: str):
        # convert to milliseconds
        # from, to, text, conf
        out = self.session_manager.output_data_templates[OUTPUT_ID]
        out.data = [(0, self.length, data, 1.0)]
        return self.session_manager.output_data_templates
