# BlazeFace
This module is a Pytorch implementation of blazeface.
It will the detect the most prominent face in an image and returns a stream containing the coordinates of the bounding
box as well as the confidence values.

* https://arxiv.org/pdf/1907.05047.pdf
* https://github.com/hollance/BlazeFace-PyTorch

## Options

- `model` (`list`):  identifier of the model to choose:
  - `front`
  - `back`

- `min_suppression_thresh` (`float`) : `0.3`, The minimum non-maximum-suppression threshold for face detection to be considered overlapped.
- `min_score_thresh` (`float`) : `0.5`, The minimum confidence score for the face detection to be considered successful.
- `batch_size` (`int`) : `50`,  Number of samples to process at once, increases speed but also (V)RAM consumption.
- `repeat_last` (`bool`) : `True`,  If set to true frames with a confidence below "min_score_thresh" will be replaced with the last frame where the confidence is above the threshold. The confidence of the replaced frame is set to zero.

## IO
Explanation of inputs and outputs as specified in the trainer file:

### Input
- `video` (`Video`): The video input on which the model should detect the faces

### Output
- `bounding_box` (`stream:SSIStream:feature;face;boundingbox;blazeface`) : The output stream containing the coordinated of the bounding box. Coordinates are normalized with respect to the image resolution, with 0,0 being the top left corner. Coordinates might be negative or bigger than one. This reflects an extension of the bounding box beyond the image.  
    - `ymin`
    - `xmin`
    - `ymax`
    - `xmax`
    - `confidence_value`

    
- `landmarks` (`stream:SSIStream:feature;face;landmarks;blazeface`) : The output stream containing the coordinated of the landmarks. Coordinates are normalized with respect to the image resolution, with 0,0 being the top left corner. Coordinates might be negative or bigger than one. This reflects an extension of the bounding box beyond the image.
  - The 12 numbers are the y,x-coordinates of the 6 facial landmark keypoints:
    - `right_exe_y` 
    - `right_eye_x`
    - `left_eye_y`
    - `left_eye_x`
    - `nose_y`
    - `nose_x`
    - `mouth_y` 
    - `mouth_x`
    - `right_ear_y`
    - `right_ear_x`
    - `left_ear_y` 
    - `left_ear_x`
    - `confidence_value`
    - Tip: these labeled as seen from the perspective of the person, so their right is your left.


## Examples

### Request

```python
import requests
import json

payload = {
  "jobID" : "´blaze_face_stream",
  "data": json.dumps([
    {"src":"file:stream:video", "type":"input", "id":"input_video", "uri":"path/to/my/file.mp4"},
    {"src":"file:stream:ssistream", "type":"output", "id":"output_stream",  "uri":"path/to/my/stream.stream"}
  ]),
  "trainerFilePath": "modules\\blazeface\\blazeface.trainer",
}


url = 'http://127.0.0.1:8080/process'
headers = {'Content-type': 'application/x-www-form-urlencoded'}
x = requests.post(url, headers=headers, data=payload)
print(x.text)

```

## Citation
```
@article{bazarevsky2019blazeface,
title={Blazeface: Sub-millisecond neural face detection on mobile gpus},
author={Bazarevsky, Valentin and Kartynnik, Yury and Vakunov, Andrey and Raveendran, Karthik and Grundmann, Matthias},
journal={arXiv preprint arXiv:1907.05047},
year={2019}
}
```

## License
Integration of the Pytorch BlazeFace implementation by Matthijs Hollemans into DISCOVER.
This work is licensed under the same terms as MediaPipe (Apache License 2.0)
https://github.com/google/mediapipe/blob/master/LICENSE