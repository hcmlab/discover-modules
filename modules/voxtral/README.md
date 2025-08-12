# Voxtral

Mistral's first real-time speech-to-text model with native multilingual support.

* https://mistral.ai/news/voxtral
* https://arxiv.org/abs/2507.13264

## Options

- `model`: Model variant to use:
    - `voxtral-mini-3b` - Mini 3B model (~11GB VRAM required)
    - `voxtral-small-24b` - Small 24B model (~55GB VRAM required)

- `language`: Language code for transcription. Supported languages include:
    - `auto` - Automatic language detection
    - `ar`, `ca`, `cs`, `da`, `de`, `el`, `en`, `es`, `fa`, `fi`, `fr`, `he`, `hi`, `hr`, `hu`, `it`, `ja`, `ko`, `nl`, `pl`, `pt`, `ru`, `sk`, `sl`, `te`, `tr`, `uk`, `ur`, `vi`, `zh`

- `compute_type`: Precision for inference:
    - `bfloat16` - Brain floating point 16-bit (recommended)
    - `float16` - Half precision floating point

- `verbose`: Enable detailed logging:
    - `True` - Show chunking and stitching debug information
    - `False` - Minimal logging (default)

**Note**: Voxtral requires CUDA-enabled GPU. Audio files of any length are supported through intelligent chunking.

## IO

### Input
- `audio` (`Audio`): The input audio to transcribe
  
### Output
- `transcript` (`FreeAnnotation`): The transcription result

## Examples

### Request

```python
import requests
import json

payload = {
  "jobID" : "voxtral_transcript",
  "data": json.dumps([
    {"src":"file:stream:audio", "type":"input", "id":"audio", "uri":"path/to/my/file.wav"},
    {"src":"file:annotation:free", "type":"output", "id":"transcript", "uri":"path/to/my/transcript.annotation"}
  ]),
  "trainerFilePath": "modules/voxtral/voxtral.trainer",
}

url = 'http://127.0.0.1:8080/process'
headers = {'Content-type': 'application/x-www-form-urlencoded'}
x = requests.post(url, headers=headers, data=payload)
print(x.text)
```