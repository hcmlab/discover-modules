"""Vendored subset of the `openface-test` pip package (v0.1.26, OpenFace 3.0,
https://github.com/CMU-MultiComp-Lab/OpenFace-3.0).

Only the multitask predictor (EfficientNet-B0 MTL head: expression, gaze, AUs)
is included. The pip package pins Pillow==9.4.0, which has no Python >= 3.12
wheels and fails to build from source; its face/landmark detectors
(RetinaFace, STAR) are unused here since detection comes from the blazeface
stream.
"""
