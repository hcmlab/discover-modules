# Wav2VecBert2

Facebook open-sourced the Conformer-based W2v-BERT 2.0 speech encoder as described in Section 3.2.1 of the paper below, which is at the core of the Seamless models.
The model was pre-trained on 4.5M hours of unlabeled audio data across more than 143 languages.
It requires fine-tuning for downstream tasks such as ASR or audio classification.

* https://arxiv.org/pdf/2312.05187.pdf
* https://huggingface.co/facebook/w2v-bert-2.0#w2v-bert-20-speech-encoder

## Processing
The module extracts embeddings once per session at model-native temporal resolution (approximately 20ms at 16kHz), then applies optional iterator-style window pooling.

Default trainer settings are configured for no extra pooling:
- `frameStep`: `20ms`
- `leftContext`: `0ms`
- `rightContext`: `0ms`

If you set larger context windows, the module pools embeddings over:
- window: `leftContext + frameStep + rightContext`
- hop: `frameStep` (stride default)

## IO
Explanation of inputs and outputs as specified in the trainer file:

### Input
- `audio` (`Audio`): Audio stream to encode

### Output
- `embeddings` (`SSIStream`): 1024-dimensional frame-wise embeddings

## Options
- `batch_size` (`int`): `250`, processing batch size

## Example

### Request

```python
import requests
import json

payload = {
  "jobID": "w2vbert2",
  "data": json.dumps([
    {"src": "file:stream:audio", "type": "input", "id": "audio", "uri": "path/to/my/file.wav"},
    {"src": "file:stream:SSIStream", "type": "output", "id": "embedding", "uri": "path/to/my/embeddings.stream"}
  ]),
  "trainerFilePath": "modules\\w2v_bert_2\\w2v_bert_2.trainer",
}

url = "http://127.0.0.1:8080/process"
headers = {"Content-type": "application/x-www-form-urlencoded"}
x = requests.post(url, headers=headers, data=payload)
print(x.text)
```

## Citation

```
@article{barrault2023seamless,
  title={Seamless: Multilingual Expressive and Streaming Speech Translation},
  author={Barrault, Lo{\"\i}c and Chung, Yu-An and Meglioli, Mariano Coria and Dale, David and Dong, Ning and Duppenthaler, Mark and Duquenne, Paul-Ambroise and Ellis, Brian and Elsahar, Hady and Haaheim, Justin and others},
  journal={arXiv preprint arXiv:2312.05187},
  year={2023}
}
```

## License
MIT
