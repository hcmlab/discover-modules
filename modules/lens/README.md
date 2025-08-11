# LENS

This module uses LENS to predict data samples with the help of a large language model.

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
- `turn_based_analysis` (`bool`) : `False`, If set to false the main transcript will be processed segment by segment. If set to true the main transcript and the context transcript will be aggregated into speaking turn pairs. This can be useful to analyze interactions in a dialogue.  



### IO
Explanation of inputs and outputs as specified in the trainer file:

### Input
- `transcript` (`FreeAnnotation`): The input text to analyze
- `transcript_context` (`FreeAnnotation`): A second transcript to provide context information for the main transcript. Only used when "turn_based_analysis" is set to true.
  
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
- `prompt` (`str`) : ``,  The prompt to pass to the llm
- `group_turns` (`bool`) : `False`, If set to false the main transcript will be processed segment by segment. If set to true the main transcript and the context transcript will be aggregated into speaking turn pairs. This can be useful to analyze interactions in a dialogue.  


### IO
Explanation of inputs and outputs as specified in the trainer file:


### Input
- `transcript` (`FreeAnnotation`): The input text to analyze
- `transcript_context` (`FreeAnnotation`): A second transcript to provide context information for the main transcript. Only used when "turn_based_analysis" is set to true.

### Output
The output of the model are three continuous annotations:
- `sentiment` (`FreeAnnotation`): The prediction with respect to the system prompt and user prompt. 

## Examples

### Request


{'system_prompt': '', 'provider': 'ollama', 'model': 'mistral-nemo', 'message': 'translate the provided text to english. Respond in JSON. Only use one key called "label".. \n """therapeut:  Okay, ja Frau Hilmann, das ist unsere erste Sitzung seit den Feiertagen. Die Feiertage sind etwas ganz Besonderes und passiert meistens relativ viel. Für manche ist es auch gar nicht so einfach, die Feiertage zu überschleben. Wie war es denn so bei Ihnen? \n patient:  Also eigentlich habe ich mich ziemlich auf Weihnachten gefreut, weil Weihnachten... ...viele schöne Sachen und ja, ich freue mich meine Familie zu sehen, die... ...weil sie so weit weg wohnt, sehen wir uns ja auch nicht so häufig.""". Value:\n', 'temperature': 0, 'resp_format': 'json', 'max_new_tokens': 128, 'enforce_determinism': True, 'stream': True}
### License
