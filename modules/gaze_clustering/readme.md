# Gaze clustering

This modules clusters gaze directions into main direction and other directions.

## Options

- `metric`: distance metric to use for clustering. Default is l1.
    - `l1`
    - `l2`
    - `cosine`

- `window`: integer describing window size used for clustering. Default is 300 ms.

- `cluster`: integer of initial clusters before agglomerating. Default is 20.
 
- `threshold`: stop agglomerating once above this percentage of points are in the main direction cluster. Default is 0.7.

- `greedy`: greedy clustering method to use for agglomeration. Default is distance.
    - `distance`
    - `mass`