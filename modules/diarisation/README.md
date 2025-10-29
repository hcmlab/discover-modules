# diarisation of whisperx transcript

This modules diarises whisperx transcript to the given roles. Do this either by clustering agglomeratively and assigning speakers chronologically, or by specifying timestamps for speaker samples and assign segments according to smallest distance.

## Options

- `anno`: select the annotation to diarise, usually `transcript`
  
- `metric`: metric to use for distances
    - `manhattan`: block-distance, L1-norm
    - `euclidean`: distance, L2-norm
    - `cosine`: 1 - cosine similarity

- `roles`: specify the speakers (roles) in their chronological first occurence

- `role_samples`:
    - specify speaker samples in the same order as `roles` in seconds (floating point number) to apply speaker similarity matching
    - **Multiple samples per role (recommended)**: `[(start,stop),(start,stop)],[(start,stop),(start,stop)]`
        - Use `[]` to group samples for each role
        - Use `()` for individual time intervals (start, stop) in seconds
        - Multiple samples per role are aggregated using mean averaging to create more robust centroids
        - Example: `[(4.3,10),(5.2,8)],[(10.5,18),(20,25),(30,35)]` - role 1 has 2 samples, role 2 has 3 samples
    - **Legacy single sample per role**: `(start,stop),(start,stop)` - still supported for backward compatibility
    - use `None` or empty string to do unsupervised clustering instead
 
- `speaker_embedding`: which model to use for embeddings. If segments are too short, they will be enlargened, thereby word alignment may get inaccurate, especially in `pyannote`'s case.
    - `speechbrain`: 192-dim, minimum duration 40ms
    - `pyannote`: 512-dim, minimum duration 300ms (!)
    - `wespeak`: 256-dim, minimum duration 105ms
    - `titanet`: Not implemented yet, since `nemo-toolkit` uses outdated packages, raising a `ModuleNotFoundError` in `pytorch_lightning`

- `method`: which clustering method to apply
    - `finch`: [finch](https://github.com/ssarfraz/FINCH-Clustering/tree/master)
    - `agglomerative`: [sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html)

- `split_audio`: check to split the source audio into the specified role-specific audios using respective timestamps from transcription

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
    'optStr': 'annotation=transcript;roles=role1,role2;role_samples=[(4.3,10),(5.2,8)],[(10.5,18),(20,25)];speaker_embedding=speechbrain;method=finch;metric=cosine'
}

import requests

url = 'http://127.0.0.1:53770/predict'
headers = {'Content-type': 'application/x-www-form-urlencoded'}
requests.post(url, headers=headers, data=payload)
```


