"""FaceMesh Module
Author: Dominik Schiller <dominik.schiller@uni-a.de>
Date: 30.01.2024
This code relies heavily on the work of following repository https://github.com/tiqq111/mediapipe_pytorch/tree/main

"""


import sys
import os
import torch
import numpy as np
import torchvision
import discover_utils.data.stream

# Add local dir to path for relative imports
sys.path.insert(0, os.path.dirname(__file__))

torchvision.disable_beta_transforms_warning()
from torchvision.transforms import v2
from facemesh_source import _FaceMesh
from utils import select_bounding_box, get_low_confidence_detections, crop_and_resize, rel_to_abs, abs_to_rel
from discover_utils.interfaces.server_module import Processor
from discover_utils.utils.cache_utils import get_file
from discover_utils.utils.log_utils import log
from discover_utils.data.static import Image
from discover_utils.data.stream import SSIStream

PYTORCH_ENABLE_MPS_FALLBACK = 1
INPUT_ID = "data"
INPUT_ID_BB = "bounding_box"
OUTPUT_ID = "landmarks"
MEDIA_TYPE_ID = "feature;face;landmarks;facemesh"

_default_options = {
    #"min_score_thresh": 0.5,
    "batch_size": 100,
}

# _dl = {}
#
#
# _dim_labels = [{"id": i, "name": x} for i, x in enumerate(_dl)]


class FaceMesh(Processor):
    chainable = False

    def __init__(self, *args, **kwargs):
        """

        Args:
            *args: Optional args for parent classes
            min_supression_thresh: Minimum confidence score for a bounding box candidate not be suppressed
            **kwargs: *args: Optional kwargs for parent classes
        """

        # Setting options
        super().__init__(*args, **kwargs)
        self.options = _default_options | self.options

        # Model
        weights_uri = [x for x in self.trainer.meta_uri if x.uri_id == "weights"][0]
        weights_uri = get_file(
            fname=weights_uri.uri_url.split("=")[-1],
            origin=weights_uri.uri_url,
            file_hash=weights_uri.uri_hash,
            cache_dir=os.getenv("CACHE_DIR"),
            tmp_dir=os.getenv("TMP_DIR"),
        )
        weights = torch.load(weights_uri)

        self._device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )
        self.model = _FaceMesh()
        self.model.load_state_dict(weights)
        self.model = self.model.eval().to(self._device)
        log(f"Using device {self._device}")

        # Fix parameters
        self.input_size = 192
        self.zero_img = np.zeros((self.input_size, self.input_size, 3), dtype=np.float32)

        # Options
        #self.min_score_thresh = float(self.options["min_score_thresh"])
        self.batch_size = int(self.options["batch_size"])

        # TODO allow detection of more than one face
        self.num_faces = 1

    def process_data(self, ds_manager) -> np.ndarray:
        self.session_manager = self.get_session_manager(ds_manager)
        data_object = self.session_manager.input_data[INPUT_ID]
        data = data_object.data
        face_bb_stream = self.session_manager.input_data[INPUT_ID_BB].data

        # Append batch dimension
        if isinstance(data_object, discover_utils.data.static.Image):
            data = np.expand_dims(data, 0)

        #bounding_boxes = []
        predictions = []
        confidence = []

        # Append batch dimension
        if isinstance(data_object, Image):
            data = np.expand_dims(data, 0)

        orig_height, orig_width, channels = data.shape[1:]

        iter_len = min(len(data), len(face_bb_stream))
        last_frame = self.zero_img

        for sample in range(0, iter_len, self.batch_size):
            idx_start = sample
            idx_end = (
                idx_start + self.batch_size
                if idx_start + self.batch_size <= iter_len
                else idx_start + (iter_len - idx_start)
            )
            log(f"Batch {sample / self.batch_size} : {idx_start} - {idx_end}")
            idxs = list(range(idx_start, idx_end))
            if not idxs:
                continue

            # Preprocess frames
            frame = data[idxs]


            bb_frame = face_bb_stream[idxs]
            #bb_low_conf = get_low_confidence_detections(
            #    bb_frame, self.min_score_thresh
            #)
            bb_coord_rel = select_bounding_box(bb_frame)
            bb_coord_abs = rel_to_abs(bb_coord_rel, (orig_height, orig_width))

            # Cropping images to face
            frame_c = [
                crop_and_resize(img, bb, self.input_size)
                #if i not in bb_low_conf
                #else self.zero_img
                for i, img, bb in zip(range(len(bb_coord_abs)), frame, bb_coord_abs)
            ]
            # frame_c = []
            #
            # for i, img, bb in zip(range(len(bb_coord_abs)), frame, bb_coord_abs):
            #     if i not in bb_low_conf:
            #         image_c = crop_and_resize(img, bb, self.input_size)
            #         frame_c.append(image_c)
            #
            #         # Repeat the last known frame if set to true
            #         if self.repeat_last:
            #             last_frame = image_c
            #     else:
            #         frame_c.append(last_frame)

            # Convert to input tensor in correct form
            frame_c = np.asarray(frame_c, dtype=np.float32)
            frame_c = (frame_c / 127.0) - 1.0
            frame_t = torch.from_numpy(frame_c).to(self._device)
            frame_t = frame_t.permute([0, 3, 1, 2])


            # image_tensor = self.transform(
            #    torch.from_numpy(np.asarray(frame)).permute([0, 3, 1, 2])
            #).to(self._device)

            with torch.no_grad():
                detections = self.model.batch_predict(frame_t)

            landmarks = detections[0].cpu().numpy().squeeze((-1,-2))  # keep batch size dimension in case it is 1!
            # Scale relative to inputsize
            landmarks = abs_to_rel(landmarks, (self.input_size, self.input_size))

            # Scale to absolute values relative to bounding box
            landmarks *= np.stack([np.tile([c[3]-c[1], c[2]-c[0], 1], landmarks.shape[1] // 3) for c in bb_coord_abs.astype(int)])

            # Add bounding box offset bb_xmin, bb_ymin, 0
            landmarks += np.stack([np.tile([c[1], c[0], 0], landmarks.shape[1] // 3) for c in bb_coord_abs.astype(int)])

            predictions.extend(landmarks)
            confidence.extend(detections[1].cpu().numpy())

        predictions = np.stack(predictions)

        # Removing z coordinates from faces and convert to numpy-array
        predictions = np.squeeze(
            predictions[:, np.mod(np.arange(predictions[1].size), 3) - 2 != 0]
        )

        #  x,y -> x,y order
        predictions = predictions[
                      :, [x - 2 if x % 2 == 0 else x for x in range(1, len(predictions[1]) + 1)]
                      ]

        # Scale
        predictions = abs_to_rel(predictions, (orig_height, orig_width))


        # TODO diff between image and video
        # Combine bounding box, landmarks, confidence
        confidence = np.expand_dims(np.squeeze(np.stack(confidence)), -1)

        # Remove negative values and scale between 0 and 1
        confidence = np.clip(confidence / 100,0,1)
        return np.hstack([predictions, confidence])

    def to_output(self, data: np.ndarray) -> dict:
        input_stream = self.session_manager.input_data[INPUT_ID]

        # Bounding Box
        mesh_template = self.session_manager.output_data_templates[OUTPUT_ID]
        self.session_manager.output_data_templates[OUTPUT_ID] = SSIStream(
            data=data.astype(mesh_template.meta_data.dtype),
            sample_rate=1
            if isinstance(input_stream, Image)
            else input_stream.meta_data.sample_rate,
            #dim_labels=_dim_labels,
            media_type=MEDIA_TYPE_ID,
            custom_meta={
                "size": f"{input_stream.meta_data.sample_shape[-2]}:{input_stream.meta_data.sample_shape[-3]}"
            },
            role=mesh_template.meta_data.role,
            dataset=mesh_template.meta_data.dataset,
            name=mesh_template.meta_data.name,
            session=mesh_template.meta_data.session,
        )

        return self.session_manager.output_data_templates


if __name__ == "__main__":

    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from pathlib import Path

    def plot_detections(img, detection):
        fig, ax = plt.subplots(1, figsize=(10, 10))
        ax.grid(False)
        ax.imshow(img)

        if detection.ndim == 1:
            detections = np.expand_dims(detection, axis=0)

        lm = detection[:-1]
        print("Found %d faces" % lm.shape[0])

        def _plot_range(lm, f, t, c):
            lm = lm.reshape(-1)
            for k in range(f, t):
                kp_y = int(lm[k * 2] * img.shape[0])
                kp_x = int(lm[k * 2 + 1] * img.shape[1])

                circle = patches.Circle(
                    (kp_x, kp_y),
                    radius=1,
                    linewidth=1,
                    edgecolor="lightskyblue",
                    facecolor="none",
                    color=c,
                )
                ax.add_patch(circle)

        _plot_range(lm, 0, int(len(lm) / 2), "red")

        plt.show()

    PYTORCH_ENABLE_MPS_FALLBACK = 1
    from discover_utils.utils.ssi_xml_utils import Trainer
    from discover_utils.data.provider.data_manager import DatasetManager

    bf_trainer = Trainer()
    bf_trainer.load_from_file("facemesh.trainer")
    fm = FaceMesh(model_io=None, opts={}, trainer=bf_trainer)

    data_dir = Path(r"/Users/dominikschiller/Work/local_nova_dir/test_files/facemesh")

    test_video = True

    if test_video:

        dd_input_video = {
            "src": "file:stream",
            "type": "input",
            "id": INPUT_ID,
            "uri": "/Users/dominikschiller/Work/local_nova_dir/test_files/test_video.mp4",
        }

        dd_input_bb = {
            "src": "file:stream",
            "type": "input",
            "id": INPUT_ID_BB,
            "uri": "/Users/dominikschiller/Work/local_nova_dir/test_files/blazeface_test_video.stream",
        }

        dd_output = {
            "src": "file:stream",
            "type": "output",
            "id": OUTPUT_ID,
            "uri": "/Users/dominikschiller/Work/local_nova_dir/test_files/face_mesh_test_video.stream",
        }
        dm_video = DatasetManager([dd_input_video, dd_input_bb, dd_output])
        dm_video.load()
        sm_video = fm.get_session_manager(dm_video)
        video_bb = fm.process_data(dm_video)
        video = sm_video.input_data[INPUT_ID].data

        plot_detections(video[94], video_bb[94])
        sm_video.output_data_templates[OUTPUT_ID] = fm.to_output(video_bb)[OUTPUT_ID]
        sm_video.save()


    else:
        for img in data_dir.glob("*.jpg"):
            dd_input_image = {
                "src": "file:image",
                "type": "input",
                "id": INPUT_ID,
                "uri": str(img),
            }

            dd_output = {
                "src": "file:stream",
                "type": "output",
                "id": OUTPUT_ID,
                "uri": str(img.with_name(img.stem + "_bb").with_suffix(".stream")),
            }

            dm_image = DatasetManager([dd_input_image, dd_output])
            sm_image = fm.get_session_manager(dm_image)
            dm_image.load()
            image_bb = fm.process_data(dm_image)
            img_data = sm_image.input_data[INPUT_ID].data
            plot_detections(img_data, image_bb)

            sm_image.output_data_templates[OUTPUT_ID] = fm.to_output(image_bb)[
                OUTPUT_ID
            ]
            sm_image.save()