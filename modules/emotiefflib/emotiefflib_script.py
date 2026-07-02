"""
EmotiEffLib Module
Author: Tobias Hallmen <tobias.hallmen@uni-a.de>
Date: 02.07.2026
cf. https://github.com/sb-ai-lab/EmotiEffLib

Facial expression (AffectNet-8) + valence/arousal + embeddings from a
pre-cropped face. Uses the torch engine of the pip package `emotiefflib`
(ex-HSEmotion, Savchenko). Default model `enet_b0_8_va_mtl` is multi-task:
8 emotion logits AND continuous valence/arousal from one forward pass.

Weights are downloaded by the package itself to ~/.emotiefflib on first use.

NOTE: the package's native emotion class order is ALPHABETICAL
(Anger, Contempt, Disgust, Fear, Happiness, Neutral, Sadness, Surprise).
Outputs are remapped to AffectNet-8 order (Neutral, Happy, Sad, Surprise,
Fear, Disgust, Anger, Contempt) to match the other affect modules (emonet,
libreface, openface3). The remap is built dynamically from the package's
own class dict and asserted, so a silent upstream reordering fails loudly
instead of corrupting predictions.
"""

import sys
import os

# Add local dir to path for relative imports
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch

from discover_utils.data.annotation import (
    DiscreteAnnotationScheme,
    ContinuousAnnotationScheme,
)
from discover_utils.data.static import Image
from discover_utils.data.stream import SSIStream
from discover_utils.interfaces.server_module import Processor
from discover_utils.utils.anno_utils import resample
from discover_utils.utils.log_utils import log
from discover_utils.utils.type_definitions import SSINPDataType

INPUT_ID = "video"
INPUT_ID_BB = "face_bb"
OUTPUT_ID_EXPRESSION = "expression"
OUTPUT_ID_VALENCE = "valence"
OUTPUT_ID_AROUSAL = "arousal"
OUTPUT_ID_EMBEDDING = "embedding"

_default_options = {
    "model_name": "enet_b0_8_va_mtl",
    "batch_size": 64,
    # Matches BlazeFace's _default_options["min_score_thresh"] (0.5) so a frame
    # BlazeFace accepted as a face is also scored here.
    "face_detection_min_conf": 0.5,
    # Relative bbox margin before cropping; the recognizer resizes internally.
    "margin": 0.1,
}

# AffectNet-8 order, shared across the affect modules. Index 8 = rest class
# (no face detected).
_label_map = {
    "0": "Neutral",
    "1": "Happy",
    "2": "Sad",
    "3": "Surprise",
    "4": "Fear",
    "5": "Disgust",
    "6": "Anger",
    "7": "Contempt",
}

# Package class names for the AffectNet-8 targets above.
_AFFECTNET8_NATIVE_NAMES = [
    "Neutral", "Happiness", "Sadness", "Surprise",
    "Fear", "Disgust", "Anger", "Contempt",
]


def _blaze_face_converter(detection: np.ndarray, img_shape: tuple):
    ymin = detection[:, 0] * img_shape[0]
    xmin = detection[:, 1] * img_shape[1]
    ymax = detection[:, 2] * img_shape[0]
    xmax = detection[:, 3] * img_shape[1]
    return np.swapaxes(np.vstack([xmin, ymin, xmax, ymax]), 0, 1)


def _get_low_confidence_detections(detection: np.ndarray, thresh: float):
    return np.where(detection[:, -1] < thresh)[0]


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def _patch_timm_compat():
    """The pip package ships a model pickled with an older timm; newer timm's
    forward reads attributes (conv_s2d/bn_s2d/aa) the unpickled blocks lack.
    Backfill them as class defaults so any recent timm works."""
    import torch.nn as nn
    try:
        from timm.models import _efficientnet_blocks as B
    except ImportError:
        return
    for cls in ("DepthwiseSeparableConv", "InvertedResidual", "EdgeResidual", "ConvBnAct"):
        c = getattr(B, cls, None)
        if c is None:
            continue
        # conv_s2d/bn_s2d are guarded by `is not None` in forward -> None is fine.
        for attr in ("conv_s2d", "bn_s2d"):
            if not hasattr(c, attr):
                setattr(c, attr, None)
        # aa (anti-alias) is CALLED unconditionally in forward -> must be a module.
        if not hasattr(c, "aa"):
            c.aa = nn.Identity()


class EmotiEffLib(Processor):
    chainable = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.options = _default_options | self.options

        self._device = (
            "cuda"
            if torch.cuda.is_available()
            else ("mps" if torch.backends.mps.is_available() else "cpu")
        )
        log(f"Using device {self._device}")

        _patch_timm_compat()
        from emotiefflib.facial_analysis import EmotiEffLibRecognizer

        self.model_name = self.options["model_name"]
        self.rec = EmotiEffLibRecognizer(
            engine="torch", model_name=self.model_name, device=self._device
        )
        self.is_mtl = "_mtl" in self.model_name

        # Build the native -> AffectNet-8 remap from the package's own dict;
        # fail loud if upstream ever changes names or order.
        native = self.rec.idx_to_emotion_class
        assert len(native) == 8, (
            f"Expected an 8-class model, got {len(native)} classes for "
            f"{self.model_name}: {native}"
        )
        name_to_idx = {v: k for k, v in native.items()}
        self._remap = [name_to_idx[n] for n in _AFFECTNET8_NATIVE_NAMES]

        self.batch_size = int(self.options["batch_size"])
        self.face_detection_min_conf = float(self.options["face_detection_min_conf"])
        self.margin = float(self.options["margin"])
        img_size = self.rec.img_size
        self.zero_img = np.zeros((img_size, img_size, 3), dtype=np.uint8)

    def _crop(self, frame: np.ndarray, bb: np.ndarray) -> np.ndarray:
        x1, y1, x2, y2 = bb
        w, h = x2 - x1, y2 - y1
        if w < 2 or h < 2:
            return self.zero_img
        mx, my = self.margin * w, self.margin * h
        H, W = frame.shape[:2]
        x1 = max(0, int(round(x1 - mx)))
        y1 = max(0, int(round(y1 - my)))
        x2 = min(W, int(round(x2 + mx)))
        y2 = min(H, int(round(y2 + my)))
        if x2 - x1 < 2 or y2 - y1 < 2:
            return self.zero_img
        return np.ascontiguousarray(frame[y1:y2, x1:x2, :3])

    def process_data(self, ds_manager) -> dict:
        predictions = {"expression": [], "valence": [], "arousal": [], "embedding": []}
        self.session_manager = self.get_session_manager(ds_manager)
        data_object = self.session_manager.input_data[INPUT_ID]
        data = data_object.data
        face_bb = self.session_manager.input_data[INPUT_ID_BB].data

        # Append batch dimension
        if isinstance(data_object, Image):
            data = np.expand_dims(data, 0)

        orig_height, orig_width = data.shape[1:3]

        iter_len = min(len(data), len(face_bb))
        for sample in range(0, iter_len, self.batch_size):
            idx_start = sample
            idx_end = min(idx_start + self.batch_size, iter_len)
            idxs = list(range(idx_start, idx_end))
            if not idxs:
                continue
            log(f"Batch {sample / self.batch_size} : {idx_start} - {idx_end}")

            frame = data[idxs]
            bb_frame = face_bb[idxs]
            bb_low_conf = _get_low_confidence_detections(
                bb_frame, self.face_detection_min_conf
            )
            bb_abs = _blaze_face_converter(bb_frame, (orig_height, orig_width))

            crops = [
                self._crop(img, bb) if i not in bb_low_conf else self.zero_img
                for i, img, bb in zip(range(len(bb_abs)), frame, bb_abs)
            ]

            # One backbone pass; emotion + V/A come from the package's linear
            # head applied to the same features.
            with torch.no_grad():
                feats = np.asarray(self.rec.extract_features(crops), dtype=np.float32)
            scores = feats @ self.rec.classifier_weights.T + self.rec.classifier_bias

            probs = _softmax(scores[:, :8])[:, self._remap]
            # Add rest class; force it for frames without a face
            probs = np.concatenate([probs, np.zeros((len(idxs), 1), np.float32)], axis=-1)
            rest = np.zeros(9, dtype=np.float32)
            rest[-1] = 1.0
            probs[bb_low_conf] = rest
            predictions["expression"].append(probs)

            if self.is_mtl and scores.shape[1] >= 10:
                valence = np.clip(scores[:, 8], -1, 1)
                arousal = np.clip(scores[:, 9], -1, 1)
            else:
                valence = np.zeros(len(idxs), np.float32)
                arousal = np.zeros(len(idxs), np.float32)
            valence[bb_low_conf] = 0
            arousal[bb_low_conf] = 0
            predictions["valence"].append(valence.astype(np.float32))
            predictions["arousal"].append(arousal.astype(np.float32))

            feats[bb_low_conf] = 0
            predictions["embedding"].append(feats)

        for k in predictions.keys():
            predictions[k] = np.concatenate(predictions[k])

        return predictions

    def to_output(self, data: dict) -> dict:
        sr_hz = self.session_manager.input_data[INPUT_ID].meta_data.sample_rate
        for l, d in data.items():
            if l == OUTPUT_ID_EMBEDDING:
                continue
            anno = self.session_manager.output_data_templates[l]
            if isinstance(anno.annotation_scheme, DiscreteAnnotationScheme):
                data_ = np.asarray(
                    list(
                        zip(
                            np.array(range(len(d))) * (1 / sr_hz) * 1000,
                            np.array(range(1, len(d) + 1)) * (1 / sr_hz) * 1000,
                            np.argmax(d, axis=-1),
                            np.amax(d, axis=-1),
                        )
                    ),
                    dtype=anno.annotation_scheme.label_dtype,
                )
            elif isinstance(anno.annotation_scheme, ContinuousAnnotationScheme):
                data_ = d.astype(dtype=anno.annotation_scheme.label_dtype)
                data_ = resample(data_, sr_hz, anno.annotation_scheme.sample_rate)
            else:
                data_ = None
            anno.data = data_

        self.session_manager.output_data_templates[OUTPUT_ID_EMBEDDING] = SSIStream(
            data=np.array(data["embedding"], SSINPDataType.FLOAT.value),
            sample_rate=sr_hz,
            media_type="feature;face;emotiefflib",
        )

        return self.session_manager.output_data_templates


if __name__ == "__main__":
    from pathlib import Path
    from discover_utils.utils.ssi_xml_utils import Trainer
    from discover_utils.data.provider.data_manager import DatasetManager

    ee_trainer = Trainer()
    ee_trainer.load_from_file("emotiefflib.trainer")
    ee = EmotiEffLib(model_io=None, opts={}, trainer=ee_trainer)

    data_dir = Path(os.getenv("TEST_DATA_DIR", "./test_files"))
    for img in sorted(data_dir.glob("*.jpg")):
        bb_stream_file = img.with_name(img.stem + "_bb").with_suffix(".stream")
        if not bb_stream_file.is_file():
            print(f"No bounding box found for file {img}")
            continue

        dd_input_image = {
            "src": "file:image",
            "type": "input",
            "id": INPUT_ID,
            "uri": str(img),
        }
        dd_input_bb = {
            "src": "file:stream",
            "type": "input",
            "id": INPUT_ID_BB,
            "uri": str(bb_stream_file),
        }
        dd_output_expression = {
            "src": "file:annotation:discrete",
            "type": "output",
            "id": OUTPUT_ID_EXPRESSION,
            "uri": str(img.parent / "emotiefflib_expression.annotation"),
        }
        dd_output_valence = {
            "src": "file:annotation:continuous",
            "type": "output",
            "id": OUTPUT_ID_VALENCE,
            "uri": str(img.parent / "emotiefflib_valence.annotation"),
        }
        dd_output_arousal = {
            "src": "file:annotation:continuous",
            "type": "output",
            "id": OUTPUT_ID_AROUSAL,
            "uri": str(img.parent / "emotiefflib_arousal.annotation"),
        }

        dm_image = DatasetManager(
            [dd_input_image, dd_input_bb, dd_output_expression,
             dd_output_valence, dd_output_arousal]
        )
        dm_image.load()
        preds = ee.process_data(dm_image)
        expr = preds["expression"][0]
        print(
            f"{img.name}: {_label_map.get(str(int(np.argmax(expr))), 'rest_class')} "
            f"({np.max(expr):.2f}) | V {preds['valence'][0]:+.2f} "
            f"| A {preds['arousal'][0]:+.2f}"
        )
