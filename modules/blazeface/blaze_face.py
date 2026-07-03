"""Blaze face Module
Author: Dominik Schiller <dominik.schiller@uni-a.de>
Date: 04.10.2023
This code relies heavily on the work of Matthijs Hollemans published in the following repository https://github.com/hollance/BlazeFace-PyTorch

"""
# Todo enable the detecion of more than one face
# Todo enable optional video stream output

import sys
import os

import discover_utils.data.stream

# Add local dir to path for relative imports
sys.path.insert(0, os.path.dirname(__file__))

import torch
import numpy as np
import torchvision
import math

torchvision.disable_beta_transforms_warning()
from torchvision.transforms import v2
from blaze_face_source import _BlazeFace
from discover_utils.interfaces.server_module import Processor
from discover_utils.utils.cache_utils import get_file
from discover_utils.utils.log_utils import log
from discover_utils.data.static import Image
from discover_utils.utils.string_utils import string_to_bool
from discover_utils.data.stream import SSIStream

INPUT_ID = "video"
OUTPUT_ID_BB = "bounding_box"
OUTPUT_ID_LM = "landmarks"

MEDIA_TYPE_ID_BB = "feature;face;boundingbox;blazeface"
MEDIA_TYPE_ID_LM = "feature;face;landmarks;blazeface"

_default_options = {
    "min_score_thresh": 0.5,
    "min_suppression_thresh": 0.3,
    "model": "back",
    "batch_size": 250,
    "repeat_last": True,
}

# Full-resolution frames are uploaded and resized on the GPU in sub-chunks of
# this size, so peak GPU memory is bounded by the chunk instead of
# batch_size x video resolution (the antialias resize works in float32:
# 250 x 1080p was ~3 GiB before the detector even ran).
UPLOAD_CHUNK = 32

_dl_bb = ["ymin", "xmin", "ymax", "xmax"]

_dl_lm = [
    "right_eye_y",
    "right_eye_x",
    "left_eye_y",
    "left_eye_x",
    "nose_y",
    "nose_x",
    "mouth_y",
    "mouth_x",
    "right_ear_y",
    "right_ear_x",
    "left_ear_y",
    "left_ear_x",
]

_dl_bb = [{"id": i, "name": x} for i, x in enumerate(_dl_bb)]
_dl_lm = [{"id": i, "name": x} for i, x in enumerate(_dl_lm)]


class BlazeFace(Processor):
    chainable = False

    def __init__(self, *args, **kwargs):
        """

        Args:
            *args: Optional args for parent classes
            min_score_thres: Minimum confidence score for a bounding box to be returned
            min_supression_thresh: Minimum confidence score for a bounding box candidate not be suppressed
            **kwargs: *args: Optional kwargs for parent classes
        """

        # Setting options
        super().__init__(*args, **kwargs)
        self.options = _default_options | self.options

        # Build model
        self._device = (
            # "cuda" if torch.cuda.is_available() else ("mps" if torch.has_mps else "cpu")
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )
        log(f"Using device {self._device}")
        if self.options["model"] == "front":

            # Download weights and Anchors
            weights_uri = [
                x for x in self.trainer.meta_uri if x.uri_id == "front_weights"
            ][0]
            anchors_uri = [
                x for x in self.trainer.meta_uri if x.uri_id == "front_anchors"
            ][0]
            back_model = False
            input_size = (128, 128)
        else:
            weights_uri = [
                x for x in self.trainer.meta_uri if x.uri_id == "back_weights"
            ][0]
            anchors_uri = [
                x for x in self.trainer.meta_uri if x.uri_id == "back_anchors"
            ][0]
            back_model = True
            input_size = (256, 256)

        weights = get_file(
            fname=weights_uri.uri_url.split("/")[-1],
            origin=weights_uri.uri_url,
            file_hash=weights_uri.uri_hash,
            cache_dir=os.getenv("CACHE_DIR"),
            tmp_dir=os.getenv("TMP_DIR"),
        )
        anchors = get_file(
            fname=anchors_uri.uri_url.split("/")[-1],
            origin=anchors_uri.uri_url,
            file_hash=anchors_uri.uri_hash,
            cache_dir=os.getenv("CACHE_DIR"),
            tmp_dir=os.getenv("TMP_DIR"),
        )

        self.model = _BlazeFace(back_model=back_model).to(self._device)
        self.model.load_weights(weights)
        self.model.load_anchors(anchors)

        self.input_size = input_size[0]

        # Optionally change the thresholds:
        self.model.min_score_thresh = float(self.options["min_score_thresh"])
        self.model.min_suppression_thresh = float(
            self.options["min_suppression_thresh"]
        )
        self.batch_size = int(self.options["batch_size"])
        self.repeat_last = string_to_bool(self.options["repeat_last"])

        # TODO allow detection of more than one face
        self.num_faces = 1

    def _remove_letterbox(self, d, pad_t, pad_l, new_h, new_w):
        """Map detections from padded-square coords back to original-frame
        normalized coords (MediaPipe's DetectionLetterboxRemoval)."""
        if len(d) == 0:
            return d
        S = self.input_size
        y_cols = [0, 2, 5, 7, 9, 11, 13, 15]   # ymin, ymax + keypoint y
        x_cols = [1, 3, 4, 6, 8, 10, 12, 14]   # xmin, xmax + keypoint x
        d = d.clone()
        d[:, y_cols] = (d[:, y_cols] * S - pad_t) / new_h
        d[:, x_cols] = (d[:, x_cols] * S - pad_l) / new_w
        return d

    def _post_process_sample(self, x):
        # Move to cpu
        x = x.cpu().numpy()

        # No faces
        if x.shape[0] == 0:
            return np.zeros((self.num_faces, 17))

        # Too few faces
        elif x.shape[0] < self.num_faces:
            pred = np.concatenate(
                (x, np.zeros((self.num_faces - x.shape[0], 17)))
            )

        else:
            # Too many faces
            # sort by confidence and get num_faces
            idx = np.argsort(x[:, 16])[-self.num_faces:]
            pred = x[idx]

        # Letterboxed input preserves the aspect ratio, so boxes are already
        # square in pixel space after letterbox removal. Sanity-check only.
        meta_data = self.session_manager.input_data[INPUT_ID].meta_data
        orig_height = meta_data.sample_shape[-3]
        orig_width = meta_data.sample_shape[-2]

        h_bb = abs(pred[:, 2] - pred[:, 0]) * orig_height
        w_bb = abs(pred[:, 3] - pred[:, 1]) * orig_width

        # Maximum difference between length and width in pixels to still be considered square.
        # Compensates for rounding errors (letterbox resize rounds the scaled side to whole pixels).
        max_ar_diff = 2
        if np.any(np.abs(h_bb - w_bb) > max_ar_diff):
            raise ValueError(f'Assertion Error: Bounding box aspect ratio is forced to be 1:1 but {h_bb / w_bb} ')

        return pred

    def process_data(self, ds_manager) -> tuple:
        self.session_manager = self.get_session_manager(ds_manager)
        data_object = self.session_manager.input_data[INPUT_ID]
        data = data_object.data

        # Append batch dimension
        if isinstance(data_object, discover_utils.data.static.Image):
            data = np.expand_dims(data, 0)

        predictions = []
        # self.orig_height, self.orig_width, self.channels = data.shape[1:]
        for i in range(0, len(data), self.batch_size):
            idx_start = i
            idx_end = (
                idx_start + self.batch_size
                if idx_start + self.batch_size <= len(data)
                else len(data)
            )
            idxs = list(range(idx_start, idx_end))
            log(f"Batch {i / self.batch_size} : {idx_start} - {idx_end}")
            if not idxs:
                continue

            frame = np.asarray(data[idxs])
            # MediaPipe-style FIT: aspect-preserving resize + symmetric
            # zero padding to the square model input
            H, W = frame.shape[1:3]
            S = self.input_size
            scale = S / max(H, W)
            new_h = min(S, max(1, round(H * scale)))
            new_w = min(S, max(1, round(W * scale)))
            pad_t, pad_l = (S - new_h) // 2, (S - new_w) // 2
            tf = v2.Compose([
                v2.Resize(size=(new_h, new_w), antialias=True),
                v2.Pad(padding=[pad_l, pad_t, S - new_w - pad_l, S - new_h - pad_t],
                       fill=0),
            ])
            # upload + resize in sub-chunks; the resized frames (128/256 px)
            # are small, so the detector still sees the whole batch at once
            frame_t = torch.cat([
                tf(
                    torch.from_numpy(frame[j:j + UPLOAD_CHUNK])
                    .to(self._device)
                    .permute((0, 3, 1, 2))
                )
                for j in range(0, len(frame), UPLOAD_CHUNK)
            ])
            detections = self.model.predict_on_batch(frame_t)
            detections = [
                self._remove_letterbox(d, pad_t, pad_l, new_h, new_w)
                for d in detections
            ]
            predictions.extend(detections)

        # Removing additional faces and adjust aspect ratio
        predictions = np.concatenate(
            [self._post_process_sample(x) for x in predictions]
        )

        if self.repeat_last:
            # Init bounding box with full image
            last_p = np.zeros((self.num_faces, 17))
            last_p[:, 2:4] = 1
            for i, p in enumerate(predictions):
                if p[-1] == 0:
                    predictions[i] = last_p
                    predictions[i][-1] = 0
                else:
                    last_p = p

        # Bounding Box and confidence
        bb = predictions[:, [0, 1, 2, 3, -1]]

        # Landmarks flip x,y order to y,x order
        lm = predictions[:, 4:]
        lm = lm[:, [x - 2 if x % 2 == 0 else x for x in range(1, lm.shape[1])] + [-1]]
        return bb, lm

    def to_output(self, data: tuple) -> dict:
        def create_stream(stream_data, template, input_stream, dim_labels, media_type):
            return SSIStream(
                data=stream_data.astype(template.meta_data.dtype),
                sample_rate=1
                if isinstance(input_stream, Image)
                else input_stream.meta_data.sample_rate,
                dim_labels=dim_labels,
                media_type=media_type,
                custom_meta={
                    "size": f"{input_stream.meta_data.sample_shape[-2]}:{input_stream.meta_data.sample_shape[-3]}"
                },
                role=template.meta_data.role,
                dataset=template.meta_data.dataset,
                name=template.meta_data.name,
                session=template.meta_data.session,
            )

        # Bounding Box
        self.session_manager.output_data_templates[OUTPUT_ID_BB] = create_stream(
            data[0],
            self.session_manager.output_data_templates[OUTPUT_ID_BB],
            self.session_manager.input_data[INPUT_ID],
            _dl_bb,
            MEDIA_TYPE_ID_BB,
        )

        # Landmarks
        self.session_manager.output_data_templates[OUTPUT_ID_LM] = create_stream(
            data[1],
            self.session_manager.output_data_templates[OUTPUT_ID_LM],
            self.session_manager.input_data[INPUT_ID],
            _dl_lm,
            MEDIA_TYPE_ID_LM,
        )
        return self.session_manager.output_data_templates


if __name__ == "__main__":

    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from pathlib import Path


    def plot_detections(img, bb, lm, with_keypoints=True):
        fig, ax = plt.subplots(1, figsize=(10, 10))
        ax.grid(False)
        ax.imshow(img)

        if isinstance(bb, torch.Tensor):
            bb = bb.cpu().numpy()

        if bb.ndim == 1:
            bb = np.expand_dims(bb, axis=0)

        if isinstance(lm, torch.Tensor):
            lm = lm.cpu().numpy()

        if lm.ndim == 1:
            lm = np.expand_dims(lm, axis=0)

        print("Found %d faces" % bb.shape[0])

        for i in range(bb.shape[0]):
            ymin = bb[i, 0] * img.shape[0]
            xmin = bb[i, 1] * img.shape[1]
            ymax = bb[i, 2] * img.shape[0]
            xmax = bb[i, 3] * img.shape[1]

            rect = patches.Rectangle(
                (xmin, ymin),
                xmax - xmin,
                ymax - ymin,
                linewidth=1,
                edgecolor="r",
                facecolor="none",
                )
            ax.add_patch(rect)

            if with_keypoints:
                for k in range(6):
                    kp_y = lm[i, k * 2] * img.shape[0]
                    kp_x = lm[i, k * 2 + 1] * img.shape[1]
                    circle = patches.Circle(
                        (kp_x, kp_y),
                        radius=0.5,
                        linewidth=1,
                        edgecolor="lightskyblue",
                        facecolor="none",
                    )
                    ax.add_patch(circle)

        plt.show()


    PYTORCH_ENABLE_MPS_FALLBACK = 1
    from discover_utils.utils.ssi_xml_utils import Trainer
    from discover_utils.data.provider.data_manager import DatasetManager

    bf_trainer = Trainer()
    bf_trainer.load_from_file("blazeface.trainer")
    bf = BlazeFace(
        model_io=None,
        trainer=bf_trainer,
        opts={"model": "front"},
    )

    video = True
    if video:
        dd_input_video = {
            "src": "file:stream",
            "type": "input",
            "id": INPUT_ID,
            "uri": "/Users/dominikschiller/Work/local_nova_dir/test_files/test_video.mp4",
        }

        dd_output_bb = {
            "src": "file:stream",
            "type": "output",
            "id": OUTPUT_ID_BB,
            "uri": "/Users/dominikschiller/Work/local_nova_dir/test_files/blazeface_bb_test_video.stream",
        }

        dd_output_lm = {
            "src": "file:stream",
            "type": "output",
            "id": OUTPUT_ID_LM,
            "uri": "/Users/dominikschiller/Work/local_nova_dir/test_files/blazeface_lm_test_video.stream",
        }

        dm_video = DatasetManager([dd_input_video, dd_output_bb, dd_output_lm])
        sm_video = bf.get_session_manager(dm_video)
        dm_video.load()
        output = bf.process_data(dm_video)
        video = sm_video.input_data[INPUT_ID].data
        plot_detections(video[100], output[0][100], output[1][100])

        for k, v in bf.to_output(output).items():
            sm_video.output_data_templates[k] = v
        sm_video.save()

    else:
        data_dir = Path(r"/Users/dominikschiller/Work/local_nova_dir/test_files/emonet")

        for img in data_dir.glob("*.jpg"):
            dd_input_image = {
                "src": "file:image",
                "type": "input",
                "id": INPUT_ID,
                "uri": str(img),
            }

            dd_output_bb = {
                "src": "file:stream",
                "type": "output",
                "id": OUTPUT_ID_BB,
                "uri": str(img.with_name(img.stem + "_bb").with_suffix(".stream")),
            }

            dd_output_lm = {
                "src": "file:stream",
                "type": "output",
                "id": OUTPUT_ID_LM,
                "uri": str(img.with_name(img.stem + "_lm").with_suffix(".stream")),
            }

            dm_image = DatasetManager([dd_input_image, dd_output_bb, dd_output_lm])
            sm_image = bf.get_session_manager(dm_image)
            dm_image.load()
            output = bf.process_data(dm_image)
            img_data = sm_image.input_data[INPUT_ID].data
            plot_detections(img_data, output[0], output[1])

            for k, v in bf.to_output(output).items():
                sm_image.output_data_templates[k] = v
            sm_image.save()
