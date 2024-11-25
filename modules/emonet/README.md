# Estimation of continuous valence and arousal levels from faces in naturalistic conditions, Nature Machine Intelligence 2021

Official implementation of the paper _"Estimation of continuous valence and arousal levels from faces in naturalistic conditions"_, Antoine Toisoul, Jean Kossaifi, Adrian Bulat, Georgios Tzimiropoulos and Maja Pantic, published in Nature Machine Intelligence, January 2021 [[1]](#Citation).
Work done in collaboration between Samsung AI Center Cambridge and Imperial College London.

Please find a full-text, view only, version of the paper [here](https://rdcu.be/cdnWi).

The full article is available on the [Nature Machine Intelligence website](https://www.nature.com/articles/s42256-020-00280-0).

* https://github.com/face-analysis/emonet

## IO
Explanation of inputs and outputs as specified in the trainer file

The input of the model is a video and feature stream containing the bounding box for the face in the video:
- `input_data` (`Image` | `Video`) : The image or video on which the model should detect the emotions
- `face_bb` (`SSIStream:blazeface`) : The bounding box stream as created by the [blaze_face](../blazeface) model

The output of the model are three annotations
- `expression` (`DiscreteAnnotation`): A discrete annotation with either 5 or 8 categorical emotions
  - For 8 emotions
    - 0 - Neutral
    - 1 - Happy
    - 2 - Sad
    - 3 - Surprise
    - 4 - Fear
    - 5 - Disgust
    - 6 - Anger
    - 7 - Contempt

  - For 5 emotions
    - 0 - Neutral
    - 1 - Happy
    - 2 - Sad
    - 3 - Surprise
    - 4 - Fear

- `valence` (`ContinuousAnnotation`): A continuous annotation representing the valence value of the circumplex model
- `arousal` (`ContinuousAnnotation`) : A continuous annotation representing the arousal value of the circumplex model


## Examples

### Request

```python
import requests
import json

payload = {
  "jobID" : "emonet",
  "data": json.dumps([
    {"src":"file:stream:video", "type":"input", "id":"input_data", "uri":"path/to/my/video.mp4"},
    {"src":"file:stream:ssifeature", "type":"input", "id":"face_bb", "uri":"path/to/my/bounding_boxes.stream"},
    {"src":"file:annotation:discrete", "type":"output", "id":"expression",  "uri":"path/to/my/expression.annotation"},
    {"src":"file:annotation:continuous", "type":"output", "id":"valence",  "uri":"path/to/my/valence.annotation"},
    {"src":"file:annotation:continuous", "type":"output", "id":"arousal",  "uri":"path/to/my/arousal.annotation"}
  ]),
  "trainerFilePath": "modules\\emonet\\emonet.trainer",
}


url = 'http://127.0.0.1:8080/process'
headers = {'Content-type': 'application/x-www-form-urlencoded'}
x = requests.post(url, headers=headers, data=payload)
print(x.text)

```


## Citation

```
@article{toisoul2021estimation,
  author  = {Antoine Toisoul and Jean Kossaifi and Adrian Bulat and Georgios Tzimiropoulos and Maja Pantic},
  title   = {Estimation of continuous valence and arousal levels from faces in naturalistic conditions},
  journal = {Nature Machine Intelligence},
  year    = {2021},
  url     = {https://www.nature.com/articles/s42256-020-00280-0}
}
```

[1] _"Estimation of continuous valence and arousal levels from faces in naturalistic conditions"_, Antoine Toisoul, Jean Kossaifi, Adrian Bulat, Georgios Tzimiropoulos and Maja Pantic, published in Nature Machine Intelligence, January 2021 

## License

Code available under a Creative Commons Attribution-Non Commercial-No Derivatives 4.0 International Licence (CC BY-NC-ND) license.
