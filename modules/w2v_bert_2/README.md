# Wav2VecBert2

Facebook opensourced Conformer-based W2v-BERT 2.0 speech encoder as described in Section 3.2.1 of the [paper](https://arxiv.org/pdf/2312.05187.pdf), which is at the core of the 
Seamless models.
This model was pre-trained on 4.5M hours of unlabeled audio data covering more than 143 languages. 
It requires finetuning to be used for downstream tasks such as Automatic Speech Recognition (ASR), or Audio Classification.

* https://arxiv.org/pdf/2312.05187.pdf
* https://huggingface.co/facebook/w2v-bert-2.0#w2v-bert-20-speech-encoder

## IO
Explanation of inputs and outputs as specified in the trainer file:

### Input
- `audio` (`Audio`): The video input on which the model should detect the faces

### Output
The output of the model are three continuous annotations:
- `embeddings` (`SSIStream`): The 1024 feature embeddings of the trained model.

## Options
- `batch_size` (`int`) : `250`, batch size in which the data is processed:


## Examples

### Request

```python
import requests
import json

payload = {
  "jobID" : "w2vbert2",
  "data": json.dumps([
    {"src":"file:stream:audio", "type":"input", "id":"audio", "uri":"path/to/my/file.wav"},
    {"src":"file:stream:SSIStream", "type":"output", "id":"embedding",  "uri":"path/to/my/embeddings.stream"}
  ]),
  "trainerFilePath": "modules\\w2v_bert_2\\w2v_bert_2.trainer",
}


url = 'http://127.0.0.1:8080/process'
headers = {'Content-type': 'application/x-www-form-urlencoded'}
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

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.