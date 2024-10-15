# openSMILE

Currently, three standard sets are supported. ComParE 2016 is the largest with more than 6k features. The smaller sets GeMAPS and eGeMAPS come in variants v01a, v01b and v02 (only eGeMAPS). We suggest to use the latest version unless backward compatibility with the original papers is desired.

Each feature set can be extracted on two levels:

- `Low-level descriptors (LDD)`
- `Functionals`

For ComParE 2016 a third level is available:

- `LLD deltas`

https://audeering.github.io/opensmile-python

## IO
Explanation of inputs and outputs as specified in the trainer file

The input of the model is an audio stream: 
- `input_data` (`Audio`) : The input audio signal

The output of the model are three annotations
- `output_data` (`Feature`): Calculated features  

## Examples

### Request

```python
import requests
import json

payload = {
  "jobID" : "opensmile_gemaps",
  "data": json.dumps([
    {"src":"file:stream:audio", "type":"input", "id":"input", "uri":"path/to/my/audio.wav"},
    {"src":"file:stream:SSIStream:feature", "type":"output", "id":"output",  "uri":"path/to/my/gemaps_features.stream"}
  ]),
  "trainerFilePath": "modules\\opensmile\\opensmile.trainer",
}


url = 'http://127.0.0.1:8080/process'
headers = {'Content-type': 'application/x-www-form-urlencoded'}
x = requests.post(url, headers=headers, data=payload)
print(x.text)

```


## Citation
```
@inproceedings{eyben2010opensmile,
  title={Opensmile: the munich versatile and fast open-source audio feature extractor},
  author={Eyben, Florian and W{\"o}llmer, Martin and Schuller, Bj{\"o}rn},
  booktitle={Proceedings of the 18th ACM international conference on Multimedia},
  pages={1459--1462},
  year={2010}
}
```


## License

openSMILE follows a dual-licensing model. Since the main goal of the project is a widespread use of the software to facilitate research in the field of machine learning from audio-visual signals, the source code and binaries are freely available for private, research, and educational use under an open-source license (see LICENSE). It is not allowed to use the open-source version of openSMILE for any sort of commercial product. Fundamental research in companies, for example, is permitted, but if a product is the result of the research, we require you to buy a commercial development license. Contact us at info@audeering.com (or visit us at https://www.audeering.com) for more information.

Original authors: Florian Eyben, Felix Weninger, Martin Wöllmer, Björn Schuller

Copyright © 2008-2013, Institute for Human-Machine Communication, Technische Universität München, Germany

Copyright © 2013-2015, audEERING UG (haftungsbeschränkt)

Copyright © 2016-2020, audEERING GmbH




