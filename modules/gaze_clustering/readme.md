# Gaze clustering

This modules clusters gaze directions into main direction and other directions.

## Options

- `metric`: distance metric to use for clustering. Default is l2.
    - `l2`
    - `l1`
    - `cosine`
    - `medium`

- `window`: integer value of the window size used for clustering. Default is 10 frames.

- `cluster`: number of initial clusters before agglomerating. Default is 10.
 
- `threshold`: stop agglomerating once above this percentage of points are in the main direction cluster. Default is 0.7.

- `greedy`: greedy clustering method to use for agglomeration. Default is distance.
    - `distance`
    - `mass`