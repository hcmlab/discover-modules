# CrisperWhisper Module

Advanced speech recognition module for the [DISCOVER](https://github.com/hcmlab/discover) framework, based on [CrisperWhisper](https://github.com/nyrahealth/CrisperWhisper).

## Features

- **Verbatim Transcriptions**: Captures speech including fillers like "um" and "uh"
- **Word-level Timestamps**: Provides precise timing information for each word
- **Reduced Hallucinations**: Uses advanced techniques to minimize transcription errors
- **GPU Acceleration**: Supports CUDA for faster processing
- **Flexible Configuration**: Configurable batch size, chunk length, and model parameters
- **German Language Support**: Optimized for German speech recognition
- **Tensor Dimension Fix**: Includes patch for CrisperWhisper tensor dimension issues
- **Timestamp Processing**: Advanced pause adjustment and None timestamp handling

## Usage

### Installation
```bash
cd modules/crisperwhisper
pip install -r requirements.txt
```

### Testing
```bash
python crisperwhisper.py
```

## Configuration Options

- `model_id`: CrisperWhisper model identifier (default: "nyrahealth/CrisperWhisper")
- `batch_size`: Processing batch size (default: 1, reduced for stability)
- `chunk_length_s`: Audio chunk length in seconds (default: 30)
- `device`: Processing device - auto, cpu, or cuda (default: "auto")
- `torch_dtype`: Tensor data type - auto, float16, or float32 (default: "auto")
- `language`: Target language for transcription (default: "de" for German)
- `return_timestamps`: Timestamp granularity - word, segment, or none (default: "word")
- `confidence`: Confidence score for annotations (default: 1.0)

## Integration with DISCOVER

This module integrates with the DISCOVER framework through:
- Input: Audio stream (processes file path directly for better compatibility)
- Output: Free annotation with transcript text and word-level timestamps
- Support for word-level timestamp extraction with pause adjustment

## Technical Implementation

### Fixes Applied
- **Tensor Dimension Patch**: Includes runtime patch for [CrisperWhisper tensor dimension issue](https://github.com/nyrahealth/CrisperWhisper/issues/12)
- **Timestamp Handling**: Robust handling of None timestamps using average duration approximation
- **Audio Processing**: Uses file path input instead of numpy arrays for better model compatibility

### Files
- `crisperwhisper.py`: Main processor implementation
- `utils.py`: Utility functions including pause adjustment and tensor dimension patches
- `crisperwhisper.trainer`: XML configuration for DISCOVER integration

## Known Issues

- Overlapping timestamps may occur after pause distribution (TODO: comprehensive overlap handling)
- Exception handling currently masks errors (TODO: remove for proper DISCOVER error propagation)

## Based on Research

CrisperWhisper is based on research that achieved 1st place on the OpenASR Leaderboard in verbatim datasets and was accepted at INTERSPEECH 2024.