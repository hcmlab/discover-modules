# Sentiment

This module performs text-based sentiment inference and exports both:
- a continuous sentiment signal (`sentiment`)
- contextual text embeddings (`embedding`)

Default model:
- `cardiffnlp/twitter-xlm-roberta-base-sentiment-multilingual`

Useful model links:
- https://huggingface.co/cardiffnlp/twitter-xlm-roberta-base-sentiment-multilingual
- https://huggingface.co/FacebookAI/xlm-roberta-base

## Model Behavior
The module loads the configured `model_path` with `AutoModelForSequenceClassification`.

- With sentiment-finetuned checkpoints (e.g. CardiffNLP), both sentiment and embeddings are meaningful.
- With non-sentiment checkpoints (e.g. `FacebookAI/xlm-roberta-base`), embeddings are still usable, but sentiment output quality is the responsibility of the user.

## IO
### Input
- `transcript` (`FreeAnnotation`): Text transcript segments

### Output
- `sentiment` (`ContinuousAnnotation`): Expected sentiment value from class probabilities (`-1 * neg + 0 * neutral + 1 * pos`)
- `embedding` (`SSIStream`): Transformer embedding vector per processed window

## Options
- `model_path` (`STRING`): Hugging Face model id, default `cardiffnlp/twitter-xlm-roberta-base-sentiment-multilingual`

## Example Request
```python
import requests
import json

payload = {
  "jobID": "sentiment_xlmr",
  "data": json.dumps([
    {"src": "file:annotation:free", "type": "input", "id": "transcript", "uri": "path/to/my/transcript.annotation"},
    {"src": "file:annotation:continuous", "type": "output", "id": "sentiment", "uri": "path/to/my/sentiment.annotation"},
    {"src": "file:stream:ssistream", "type": "output", "id": "embedding", "uri": "path/to/my/sentiment_embedding.stream"}
  ]),
  "trainerFilePath": "modules\\sentiment\\sentiment.trainer",
  "frame_size": "40ms",
  "left_context": "0ms"
}

url = "http://127.0.0.1:8080/process"
headers = {"Content-type": "application/x-www-form-urlencoded"}
r = requests.post(url, headers=headers, data=payload)
print(r.text)
```
