"""
OpenFace 3.0 Module
Author: Tobias Hallmen <tobias.hallmen@uni-a.de>
Date: 02.07.2026
cf. https://github.com/CMU-MultiComp-Lab/OpenFace-3.0

Multitask facial analysis (predictor vendored in ./openface from pip
`openface-test`, see its __init__.py): one EfficientNet-B0 forward
yields 8-class expression (AffectNet-8), gaze direction (yaw, pitch) and
8 action-unit scores. Successor to the OpenFace 2 toolchain.

The bundled RetinaFace detector is NOT used: in our audit it returned tight
partial boxes on high-resolution frames that collapsed accuracy, and DISCOVER
pipelines already carry a blazeface bounding-box stream. The face crop is the
blazeface bbox plus a relative margin; the multitask model resizes/normalizes
internally-equivalent transforms here.

AU output order (upstream demo): AU1, AU2, AU4, AU6, AU9, AU12, AU25, AU26.
Values are ME-GraphAU-style relevance scores (no sigmoid); upstream treats
>= 0.5 as "present".
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
from discover_utils.utils.cache_utils import get_file
from discover_utils.utils.log_utils import log
from discover_utils.utils.type_definitions import SSINPDataType

INPUT_ID = "video"
INPUT_ID_BB = "bounding_box"
OUTPUT_ID_EXPRESSION = "expression"
OUTPUT_ID_AU = "action_units"
OUTPUT_ID_GAZE = "gaze"
OUTPUT_ID_EMBEDDING = "embedding"

_default_options = {
    "batch_size": 64,
    # Matches BlazeFace's _default_options["min_score_thresh"] (0.5).
    "face_detection_min_conf": 0.5,
    # Relative bbox margin before cropping.
    "margin": 0.2,
}

# Emotion head is native AffectNet-8 order (verified by per-class recall on
# posed faces in the childaffect audit).
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

_AU_ORDER = [1, 2, 4, 6, 9, 12, 25, 26]
AU_DIM_LABELS = [{"id": i, "name": f"AU{n}"} for i, n in enumerate(_AU_ORDER)]
GAZE_DIM_LABELS = [{"id": 0, "name": "gaze_yaw"}, {"id": 1, "name": "gaze_pitch"}]
MEDIA_TYPES = {
    "action_units": "feature;face;actionunits;openface3",
    "gaze": "feature;face;gaze;openface3",
    "embedding": "feature;face;embedding;openface3",
}


def _blaze_face_converter(detection: np.ndarray, img_shape: tuple):
    ymin = detection[:, 0] * img_shape[0]
    xmin = detection[:, 1] * img_shape[1]
    ymax = detection[:, 2] * img_shape[0]
    xmax = detection[:, 3] * img_shape[1]
    return np.swapaxes(np.vstack([xmin, ymin, xmax, ymax]), 0, 1)


def _get_low_confidence_detections(detection: np.ndarray, thresh: float):
    return np.where(detection[:, -1] < thresh)[0]


class OpenFace3(Processor):
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

        weights_uri = next(
            filter(lambda x: x.uri_id == "mtl_backbone", self.trainer.meta_uri), None
        )
        weight_file = get_file(
            fname=weights_uri.uri_url.split("/")[-1],
            origin=weights_uri.uri_url,
            file_hash=weights_uri.uri_hash,
            cache_dir=os.getenv("CACHE_DIR"),
            tmp_dir=os.getenv("TMP_DIR"),
        )
        log(f"Loading the model from {weight_file}.")

        from openface.multitask_model import MultitaskPredictor

        self.mt = MultitaskPredictor(str(weight_file), device=self._device)
        self.mt.model.eval()

        self.batch_size = int(self.options["batch_size"])
        self.face_detection_min_conf = float(self.options["face_detection_min_conf"])
        self.margin = float(self.options["margin"])
        self.zero_img = np.zeros((224, 224, 3), dtype=np.uint8)

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
        predictions = {"expression": [], "action_units": [], "gaze": [], "embedding": []}
        self.session_manager = self.get_session_manager(ds_manager)
        data_object = self.session_manager.input_data[INPUT_ID]
        data = data_object.data
        face_bb = self.session_manager.input_data[INPUT_ID_BB].data

        # Append batch dimension
        if isinstance(data_object, Image):
            data = np.expand_dims(data, 0)

        orig_height, orig_width = data.shape[1:3]

        # capture the shared backbone features as embedding
        feats_cap = {}
        hook = self.mt.model.base_model.register_forward_hook(
            lambda m, inp, out: feats_cap.__setitem__("f", out.detach())
        )
        try:
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
                # the predictor's own transform: ToTensor -> Resize(224) -> ImageNet norm
                batch = torch.stack([self.mt.transform(c) for c in crops]).to(self._device)

                with torch.no_grad():
                    emo, gaze, au = self.mt.model(batch)
                probs = torch.softmax(emo, dim=-1).cpu().numpy()
                gaze = gaze.cpu().numpy().astype(np.float32)
                au = au.cpu().numpy().astype(np.float32)
                feats = feats_cap["f"].cpu().numpy().astype(np.float32)

                # Add rest class; force it for frames without a face
                probs = np.concatenate(
                    [probs, np.zeros((len(idxs), 1), np.float32)], axis=-1
                )
                rest = np.zeros(9, dtype=np.float32)
                rest[-1] = 1.0
                probs[bb_low_conf] = rest
                gaze[bb_low_conf] = 0
                au[bb_low_conf] = 0
                feats[bb_low_conf] = 0

                predictions["expression"].append(probs)
                predictions["gaze"].append(gaze)
                predictions["action_units"].append(au)
                predictions["embedding"].append(feats)
        finally:
            hook.remove()

        for k in predictions.keys():
            predictions[k] = np.concatenate(predictions[k])

        return predictions

    def to_output(self, data: dict) -> dict:
        sr_hz = self.session_manager.input_data[INPUT_ID].meta_data.sample_rate

        d = data["expression"]
        anno = self.session_manager.output_data_templates[OUTPUT_ID_EXPRESSION]
        if isinstance(anno.annotation_scheme, DiscreteAnnotationScheme):
            anno.data = np.asarray(
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

        for out_id, dim_labels in (
            (OUTPUT_ID_AU, AU_DIM_LABELS),
            (OUTPUT_ID_GAZE, GAZE_DIM_LABELS),
            (OUTPUT_ID_EMBEDDING, None),
        ):
            self.session_manager.output_data_templates[out_id] = SSIStream(
                data=np.array(data[out_id if out_id != OUTPUT_ID_AU else "action_units"],
                              SSINPDataType.FLOAT.value),
                sample_rate=sr_hz,
                dim_labels=dim_labels,
                media_type=MEDIA_TYPES[out_id],
            )

        return self.session_manager.output_data_templates
