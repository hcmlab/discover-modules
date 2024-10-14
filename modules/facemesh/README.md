# FaceMesh
This module is a Pytorch implementation of facemesh.
It will the detect the most prominent face in an image and retuns a stream containing the coordinates of the bounding
box as well as the confidence values.

* https://arxiv.org/pdf/1907.06724.pdf
* https://github.com/tiqq111/mediapipe_pytorch?tab=readme-ov-file

## Options
- `min_score_thresh` (`float`) : `0.5`, The minimum confidence score for the face detection to be considered successful.
- `batch_size` (`int`) : `50`,  Number of samples to process at once, increases speed but also (V)RAM consumption.
[blaze_face.py](..%2Fblazeface%2Fblaze_face.py)
## IO
Explanation of inputs and outputs as specified in the trainer file:

### Input
Explanation of inputs and outputs as specified in the trainer file

The input of the model is a video and feature stream containing the bounding box for the face in the video:
- `input_data` (`Image` | `Video`) : The image or video on which the model should detect the emotions
- `face_bb` (`SSIStream`) : The bounding box stream 

### Output
- `landmarks` (`stream:SSIStream:feature;face;landmarks;facemesh`) : The output stream containing the coordinated of the bounding box. These are normalized coordinates (between 0 and 1) with 0,0 being the top left corner.

  Each face mesh stream consisting of 937 dimensional vector per sample:
  - The first 936 values are the coordinates in alternating y,x order
  - `confidence value`

## Examples

### Request

```python
import requests
import json

payload = {
  "jobID" : "´face_mesh_stream",
  "data": json.dumps([
    {"src":"file:stream:video", "type":"input", "id":"input_data", "uri":"path/to/my/file.mp4"},
    {"src":"file:stream:ssiStream", "type":"input", "id":"face_bb", "uri":"path/to/my/bounding_boxes.stream"},
    {"src":"file:stream:ssistream", "type":"output", "id":"output_stream",  "uri":"path/to/my/stream.stream"}
  ]),
  "trainerFilePath": "modules\\facemesh\\facemesh.trainer",
}


url = 'http://127.0.0.1:8080/process'
headers = {'Content-type': 'application/x-www-form-urlencoded'}
x = requests.post(url, headers=headers, data=payload)
print(x.text)

```

## Citation
```
@article{kartynnik2019real,
  title={Real-time facial surface geometry from monocular video on mobile GPUs},
  author={Kartynnik, Yury and Ablavatski, Artsiom and Grishchenko, Ivan and Grundmann, Matthias},
  journal={arXiv preprint arXiv:1907.06724},
  year={2019}
}
```

## License
-