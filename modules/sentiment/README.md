# GermanSentiment

This model was trained for sentiment classification of German language texts. To achieve the best results all model inputs needs to be preprocessed with the same procedure, that was applied during the training. To simplify the usage of the model, we provide a Python package that bundles the code need for the preprocessing and inferencing.

The model uses the Googles Bert architecture and was trained on 1.834 million German-language samples. The training data contains texts from various domains like Twitter, Facebook and movie, app and hotel reviews. You can find more information about the dataset and the training process in the paper.

* https://github.com/oliverguhr/german-sentiment
* https://huggingface.co/oliverguhr/german-sentiment-bert

## IO
Explanation of inputs and outputs as specified in the trainer file:


### Input
- `transcript` (`FreeAnnotation`): The input text to analyze the sentiment from 
  
### Output
The output of the model are three continuous annotations:
- `sentiment` (`ContinuousAnnotation`): The expectation value of the sentiment analysis. Calculating sentiment from pos / neg / neutral: 1*pos - 1*neg + 0*neutr

## Examples

### Request

```python
import requests
import json

payload = {
  "jobID" : "germant_sentiment",
  "data": json.dumps([
    {"src":"file:annotation:free", "type":"input", "id":"transcript", "uri":"path/to/my/transcript.annotation"},
    {"src":"file:annotation:continuous", "type":"output", "id":"sentiment", "uri":"path/to/my/sentiment.annotation"},
  ]),
  "trainerFilePath": "modules\\german_sentiment\\german_sentiment.trainer",
 "frame_size": "40ms",
 "left_context": "960ms"
}


url = 'http://127.0.0.1:8080/process'
headers = {'Content-type': 'application/x-www-form-urlencoded'}
x = requests.post(url, headers=headers, data=payload)
print(x.text)

```

### License
MIT License

Copyright (c) 2019 Oliver Guhr

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.