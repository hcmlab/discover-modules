# Estimation of 8 Facial Expressions and Valence / Arousal

MobilenetV2 implementation of the paper _"Relevance-based data masking: a model-agnostic transfer learning approach for facial expression recognition"_, Dominik Schiller, Tobias Huber, Michael Dietz and Elisabeth André. 

Please find a full-text version of the paper [here](https://www.frontiersin.org/articles/10.3389/fcomp.2020.00006/full).


## IO
Explanation of inputs and outputs as specified in the trainer file:

The input of the model is a video and feature stream containing the bounding box for the face in the video:
- `input_data`: The video input on which the model should detect the emotions
- `face_bb`: The bounding box stream as created by the [blaze_face](../blazeface) model

The output of the model are three annotations
- `expression`: A discrete annotation with either 5 or 8 categorical emotions
- `valence`: A continuous annotation with valence values in the range of [-1, 1]
- `arousal`: A continuous annotation with arousal values in the range of [-1, 1]


## Class number to expression name

The mapping from class number to expression is as follows.

```
Classes:

0 - Neutral
1 - Happy
2 - Sad
3 - Surprise
4 - Fear
5 - Disgust
6 - Anger
7 - Contempt
```

## Examples

### Request

```python
import requests
import json

payload = {
  "jobID" : "rbdm",
  "data": json.dumps([
    {"src":"file:stream:video", "type":"input", "id":"input_data", "uri":"path/to/my/video.mp4"},
    {"src":"file:stream:ssifeature:blazeface", "type":"input", "id":"face:bb", "uri":"path/to/my/bounding_boxes.stream"},
    {"src":"file:annotation:discrete", "type":"output", "id":"expression",  "uri":"path/to/my/expression.annotation"},
    {"src":"file:annotation:continuous", "type":"output", "id":"valence",  "uri":"path/to/my/valence.annotation"},
    {"src":"file:annotation:continuous", "type":"output", "id":"arousal",  "uri":"path/to/my/arousal.annotation"},
  ]),
  "trainerFilePath": "modules\\rbdm\\rbdm.trainer",
}


url = 'http://127.0.0.1:8080/process'
headers = {'Content-type': 'application/x-www-form-urlencoded'}
x = requests.post(url, headers=headers, data=payload)
print(x.text)

```


## Citation

```
@article{schiller2020relevance,
  title={Relevance-based data masking: a model-agnostic transfer learning approach for facial expression recognition},
  author={Schiller, Dominik and Huber, Tobias and Dietz, Michael and Andr{\'e}, Elisabeth},
  journal={Frontiers in Computer Science},
  volume={2},
  pages={6},
  year={2020},
  publisher={Frontiers Media SA}
}

```

## License

Code available under a Creative Commons Attribution-Non Commercial-No Derivatives 4.0 International Licence (CC BY-NC-ND) license.
