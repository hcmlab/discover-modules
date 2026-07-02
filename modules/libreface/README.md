# LibreFace

Facial action units (12 binary BP4D detections + 12 DISFA intensities),
facial expression (AffectNet-8) and FER embeddings via
[LibreFace](https://github.com/ihp-lab/LibreFace) (pip `libreface`, pure
PyTorch).

## I/O

Inputs:

- `video`: the input video
- `facemesh`: 468-landmark stream from the `facemesh` module (used for the
  FFHQ-style face alignment LibreFace models expect)

Outputs:

- `libreface_stream`: SSIStream, 24 dims: `AU{1,2,4,6,7,10,12,14,15,17,23,24}_c`
  (binary detection) + `AU{1,2,4,5,6,9,12,15,17,20,25,26}_r` (intensity 0-5)
- `libreface_emotion`: discrete annotation, AffectNet-8 order (Neutral,
  Happiness, Sadness, Surprise, Fear, Disgust, Anger, Contempt) + rest class
  (id 8) for frames without a valid face; confidence = FER softmax max
- `aligned`: 256x256 aligned face video (debug/downstream reuse)
- `embedding`: SSIStream with the FER student model's penultimate features

## Options

- `batch_size` (default 256)
- `min_conf` (default 0.001): minimum facemesh confidence; below, the frame
  gets the rest class and zeroed streams. NB the facemesh stream's
  confidence is the raw landmark-model score / 100, NOT a probability:
  real faces typically score 0.05-0.35 and no-face rows are exactly 0, so
  the default only skips true no-face frames.

## Implementation notes (v2 rewrite)

- Alignment runs batched on the GPU (`affine_grid`/`grid_sample`): the FFHQ
  crop quad is a parallelogram, so PIL's QUAD warp is an exact affine map.
  Supersampled at 512, antialias-downscaled to 256, mirroring upstream
  (whose aligned images are effectively the 512 transform, resized by the
  model transforms). The previous per-frame PIL/scipy warp in a multiprocess
  pool made the module CPU-bound while the GPU idled: measured ~3 ms/frame
  end-to-end (align + AU + FER, RTX 4090) vs ~255 ms/frame for the old
  alignment alone on one core.
- The AU (combined detection+intensity) and FER models are loaded once and
  reused; the upstream `*_video()` helpers re-download/reload weights on
  every call.
- Verified against the upstream pipeline on the same image: identical AU
  detections (12/12), AU intensity MAE 0.03 (0-5 scale), identical FER label.
- Emotion confidence is the real softmax (the old module wrote constant 1.0).
- Gaze / head pose are not computed: upstream derives them from iris-refined
  MediaPipe landmarks (with z) on the aligned crop, which the facemesh stream
  does not provide; running MediaPipe per frame would reintroduce the CPU
  bottleneck.
- Weights are downloaded once via gdown into `CACHE_DIR/downloads/libreface`.

## License

LibreFace: USC Research License — research/education only, no commercial use.
