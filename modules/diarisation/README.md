# diarisation of whisperx transcript

This modules diarises whisperx transcript to the given roles. Do this either by clustering agglomeratively and assigning speakers chronologically, or by specifying timestamps for speaker samples and assign segments according to smallest distance.

## Options

- `anno`: select the annotation to diarise, usually `transcript`
  
- `metric`: metric to use for distances
    - `manhattan`: block-distance, L1-norm
    - `euclidean`: distance, L2-norm
    - `cosine`: 1 - cosine similarity

- `roles`: specify the speakers (roles) in their chronological first occurence

- `intervals`:
    - specify speaker samples in the same order as `roles` using format `(start, stop), (start, stop), ...` in seconds (floating point number) to apply speaker similarity matching
    - use `None` or empty string to do unsupervised clustering instead
 
- `speaker_embedding`: which model to use for embeddings. If segments are too short, they will be enlargened, thereby word alignment may get inaccurate, especially in `pyannote`'s case.
    - `speechbrain`: 192-dim, minimum duration 40ms
    - `pyannote`: 512-dim, minimum duration 300ms (!)
    - `wespeak`: 256-dim, minimum duration 105ms
    - `titanet`: Not implemented yet, since `nemo-toolkit` uses outdated packages, raising a `ModuleNotFoundError` in `pytorch_lightning`

- `method`: which clustering method to apply
    - `finch`: [finch](https://github.com/ssarfraz/FINCH-Clustering/tree/master)
    - `agglomerative`: [sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html)

## Example payload

```python
payload = {
    'trainerFilePath': 'models\\trainer\\free\\transcript\\audio{audio}\\diarisation\\pyannote.trainer',
    'server': '127.0.0.1',
    'username': 'nova-user',
    'password': 'nova-password',
    'database': 'test-db',
    'sessions': 'session_1;session_2',
    'scheme': 'transcript',
    'roles': 'session',
    'annotator': 'whisperx',
    'streamName': 'audio',
    'schemeType': 'FREE',
    'optStr': 'annotation=transcript;roles=role1,role2;samples=;speaker_embedding=speechbrain;method=finch;metric=cosine'
}

import requests

url = 'http://127.0.0.1:53770/predict'
headers = {'Content-type': 'application/x-www-form-urlencoded'}
requests.post(url, headers=headers, data=payload)
```


