# EmoW2V
The model expects a raw audio signal as input and outputs predictions for arousal, dominance and valence in a range of approximately 0...1. In addition, it also provides the pooled states of the last transformer layer.
The model was created by fine-tuning a pre-trained wav2vec 2.0 model on MSP-Podcast (v1.7). As foundation we use wav2vec2-large-robust released by Facebook under Apache.2.0, which we pruned from 24 to 12 transformer layers before fine-tuning. The model was afterwards exported to ONNX format. Further details are given in the associated paper. For an introduction how to use the model, please visit our tutorial project.
The original [Torch](https://pytorch.org/docs/stable/torch.html) model is hosted on Hugging Face.
 * https://arxiv.org/abs/2203.07378

## IO
Explanation of inputs and outputs as specified in the trainer file:

### Input
- `audio` (`Audio`): The video input on which the model should detect the faces

### Output
The output of the model are three continuous annotations:
- `arousal` (`ContinuousAnnotation`): The arousal annotations.
- `dominance` (`ContinuousAnnotation`): The dominance annotations.
- `valence` (`ContinuousAnnotation`): The valence annotations.
- `embeddings` (`SSIStream`): The 1024 feature embeddings of the trained model.

## Options
- `batch_size` (`int`) : `250`, batch size in which the data is processed:


## Examples

### Request

```python
import requests
import json

payload = {
  "jobID" : "emow2v",
  "data": json.dumps([
    {"src":"file:stream:audio", "type":"input", "id":"audio", "uri":"path/to/my/file.wav"},
    {"src":"file:annotation:continuous", "type":"output", "id":"arousal",  "uri":"path/to/my/arousal.annotation"},
    {"src":"file:annotation:continuous", "type":"output", "id":"valence",  "uri":"path/to/my/valence.annotation"},
    {"src":"file:annotation:continuous", "type":"output", "id":"dominance",  "uri":"path/to/my/dominance.annotation"},
    {"src":"file:stream:SSIStream", "type":"output", "id":"embedding",  "uri":"path/to/my/embeddings.stream"}
  ]),
  "trainerFilePath": "modules\\emow2v\\emow2v.trainer",
}


url = 'http://127.0.0.1:8080/process'
headers = {'Content-type': 'application/x-www-form-urlencoded'}
x = requests.post(url, headers=headers, data=payload)
print(x.text)

```

## Citation

```
@article{wagner2023dawn,
title={Dawn of the Transformer Era in Speech Emotion Recognition: Closing the Valence Gap},
author={Wagner, Johannes and Triantafyllopoulos, Andreas and Wierstorf, Hagen and Schmitt, Maximilian and Burkhardt, Felix and Eyben, Florian and Schuller, Bj{\"o}rn W},
journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
pages={1--13},
year={2023},
}
```

## License
The model can be used for non-commercial purposes, see CC BY-NC-SA 4.0. For commercial usage, a license for devAIce must be obtained. The source code in this GitHub repository is released under the following license.