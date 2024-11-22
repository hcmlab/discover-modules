# Nova Assistant 

This model uses the Nova Assistant project to predict data samples with the help of a large language model.

* https://github.com/hcmlab/lens

## Lens Predict

This module automatically derives a task description, examples and label scheme from the available information about the input, depending on whether the data is loaded from a database or file. 
Each sample of the input is processed using this information.  

### Options
- `ip` (`str`) : `127.0.0.1`, The ip address to reach Nova Assistant
- `port` (`str`) : `1337`, The port Nova Assistant is listening on
- `provider` (`str`) : `ollama_chat`, The model provider
- `model` (`str`) : `llama2`,  The llm model to use
- `language` (`str`) : `en`,`de`,  The language in which the instructions are written
- `turn_based_analysis` (`bool`) : `True`, If set to true speaking the main transcript and the context transcript will be aggregated into speaking turn pairs. 
If set to false only the main transcript will be processed in single segments.


### IO
Explanation of inputs and outputs as specified in the trainer file:

### Input
- `transcript` (`FreeAnnotation`): The input text to analyze
  
### Output
The output of the model are three continuous annotations:
- `sentiment` (`DiscreteAnnotation`): The prediction with respect to the description, examples, classes and naming_scheme

## Lens Free Prompt

This module iterates samplewise over the input and uses the system prompt and the user prompt to process it. 

### Options

- `ip` (`str`) : `127.0.0.1`, The ip address to reach Nova Assistant
- `port` (`str`) : `1337`, The port Nova Assistant is listening on
- `provider` (`str`) : `ollama_chat`, The model provider
- `model` (`str`) : `llama2`,  The llm model to use
- `system_prompt` (`str`) : ``,  The system prompt to pass to the llm
- `prompt` (`str`) : ``,  The user prompt to pass to the llm


### IO
Explanation of inputs and outputs as specified in the trainer file:


### Input
- `transcript` (`FreeAnnotation`): The input text to analyze

### Output
The output of the model are three continuous annotations:
- `sentiment` (`FreeAnnotation`): The prediction with respect to the system prompt and user prompt. 

## Examples

### Request

```python
import requests
import json

payload = {
  "jobID" : "lens",
  "data": json.dumps([
    {"src":"db:annotation:free", "type":"input", "id":"transcript", "role":"testrole", "name" : "transcription"},
    {"src":"db:annotation:discrete", "type":"output",  "id":"transcript", "role":"testrole", "name" : "transcription"},
  ]),
  "trainerFilePath": "modules\\lens\\lens_predictor.trainer",
 "frame_size": "40ms",
 "left_context": "960ms"
}


url = 'http://127.0.0.1:8080/process'
headers = {'Content-type': 'application/x-www-form-urlencoded'}
x = requests.post(url, headers=headers, data=payload)
print(x.text)

```

### License
