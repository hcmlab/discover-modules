# OpenFace 3.0

Multitask facial analysis via
[OpenFace 3.0](https://github.com/CMU-MultiComp-Lab/OpenFace-3.0)
(pip `openface-test`): one EfficientNet-B0 forward pass yields 8-class
facial expression (AffectNet-8), gaze direction and 8 action units.
Successor to the OpenFace 2 toolchain (which is Windows-bound in our
setups); note OpenFace 3 predicts FEWER outputs than OpenFace 2
(8 AUs instead of 17/18, no head pose in the multitask head).

## I/O

Inputs:

- `video`: the input video
- `face_bb`: face bounding box stream from the `blazeface` module

Outputs:

- `expression`: discrete annotation, AffectNet-8 order (Neutral, Happy, Sad,
  Surprise, Fear, Disgust, Anger, Contempt) + rest class (id 8) for frames
  without a detected face; confidence = softmax max
- `action_units`: SSIStream, 8 dims in upstream demo order
  AU1, AU2, AU4, AU6, AU9, AU12, AU25, AU26. Values are ME-GraphAU-style
  relevance scores (no sigmoid); upstream treats >= 0.5 as "present".
- `gaze`: SSIStream, 2 dims (gaze_yaw, gaze_pitch)
- `embedding`: SSIStream, shared 1280-dim backbone features

## Options

- `batch_size` (default 64)
- `face_detection_min_conf` (default 0.5): matches blazeface's default
  `min_score_thresh`
- `margin` (default 0.2): relative bbox margin before cropping

## Notes

- The bundled RetinaFace detector is not used. In the childaffect audit it
  returned tight partial boxes on high-resolution multi-face frames that
  collapsed emotion accuracy; the blazeface bbox + margin crop is consistent
  with the other affect modules and was validated there.
- The expression head is fragile on small faces: on a real webcam session
  (~100 px face) it over-predicted Fear (65% of frames, rising with larger
  margins) where emonet/emotiefflib/libreface agreed on Neutral/Happy.
  Verified not to be a porting bug (bit-identical to the audit wrapper) and
  not fixable by upsampling: LANCZOS/BICUBIC/NEAREST pre-upscaling to 224
  leaves the skew intact, and a controlled test (one AffectNet face,
  LANCZOS-downscaled) flips Happy -> Surprise at 150 px -> Disgust at
  <= 100 px, i.e. the information is gone, not badly resampled. Treat
  expression as unreliable below roughly 200 px face height; the AUs, gaze
  and embeddings degrade more gracefully, and emotiefflib stays stable on
  the same 100 px crops.
- Weights (MTL_backbone.pth) come from the huggingface mirror
  nutPace/openface_weights via the nova-server cache (hash-pinned).

## License

OpenFace 3.0 code is research-oriented; the pip package `openface-test`
declares no license and the model was trained on research datasets
(AffectNet among others). TODO-verify the upstream license before any
non-research use; treat as research-only until confirmed.
