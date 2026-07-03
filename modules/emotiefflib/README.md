# EmotiEffLib

Facial expression recognition (AffectNet-8), continuous valence/arousal and
face embeddings from the [EmotiEffLib](https://github.com/sb-ai-lab/EmotiEffLib)
package (ex-HSEmotion, Savchenko). Default model `enet_b0_8_va_mtl`: an
EfficientNet-B0 pretrained on VGGFace2 and trained multi-task on AffectNet
(8-class expression AND valence/arousal), so all outputs come from a single
forward pass.

## I/O

Inputs:

- `video`: the input video
- `bounding_box`: face bounding box stream from the `blazeface` module

Outputs:

- `expression`: discrete annotation, AffectNet-8 order (Neutral, Happy, Sad,
  Surprise, Fear, Disgust, Anger, Contempt) + rest class (id 8) for frames
  without a detected face
- `valence`, `arousal`: continuous annotations in [-1, 1] (0 where no face)
- `embedding`: SSIStream with the backbone features (1280-dim for b0)

## Options

- `model_name` (default `enet_b0_8_va_mtl`): any 8-class EmotiEffLib torch
  model. Non-`_mtl` models emit no valence/arousal (outputs stay 0).
- `batch_size` (default 64)
- `face_detection_min_conf` (default 0.5): matches blazeface's
  `min_score_thresh` default so accepted frames are also scored here
- `margin` (default 0.1): relative bbox margin before cropping

## Notes

- The package's native class order is alphabetical; the module remaps to
  AffectNet-8 order (consistent with the emonet / libreface / openface3
  modules) and asserts the mapping at load time.
- Weights are auto-downloaded by the package to `~/.emotiefflib` on first run
  (not via the nova-server cache).
- The face crop comes from the blazeface bounding box plus margin; the
  recognizer resizes and normalizes internally.

## License

EmotiEffLib is Apache-2.0. Model weights were trained on VGGFace2 and
AffectNet; AffectNet itself is research-only, so check your use case against
the upstream model card before commercial deployment of the weights.
