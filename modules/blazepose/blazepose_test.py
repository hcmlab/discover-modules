"""BlaePose Module
Author: Dominik Schiller <dominik.schiller@uni-a.de>
Date: 30.10.2024
"""
# Todo enable the detecion of more than one face
# Todo enable optional video stream output

import sys
import os
import cv2
import time
import discover_utils.data.stream
from mediapipe.tasks.python.components.containers import NormalizedLandmark

# Add local dir to path for relative imports
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from discover_utils.interfaces.server_module import Processor
from discover_utils.utils.cache_utils import get_file
from discover_utils.utils.log_utils import log
from discover_utils.data.static import Image
from discover_utils.utils.string_utils import string_to_bool
from discover_utils.data.stream import SSIStream

INPUT_ID = "video"
OUTPUT_ID = "pose"

MEDIA_TYPE_ID_BB = "stream:SSIStream:feature;body;pose;blazepose"

_default_options = {
    "batch_size" : 1,
    "repeat_last": True
}

# TODO: Fix output labels
_dl = [
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

_dl= [{"id": i, "name": x} for i, x in enumerate(_dl)]


class BlazePose(Processor):
    chainable = False

    def __init__(self, *args, **kwargs):
        """

        Args:
            *args: Optional args for parent classes
            **kwargs: *args: Optional kwargs for parent classes
        """

        # Setting options
        super().__init__(*args, **kwargs)
        self.options = _default_options | self.options
        self.batch_size = int(self.options["batch_size"])
        self.repeat_last = string_to_bool(self.options["repeat_last"])
        # TODO allow detection of more than one face
        self.num_poses = 1


        # Download weights and Anchors
        task_uri = [
            x for x in self.trainer.meta_uri if x.uri_id == "task_file"
        ][0]

        def print_result(result, output_image: mp.Image, timestamp_ms: int):
            #print('gesture recognition result: {}'.format(result, timestamp_ms))
            ...


    # Build model
        task = get_file(
            fname=task_uri.uri_url.split("/")[-1],
            origin=task_uri.uri_url,
            file_hash=task_uri.uri_hash,
            cache_dir=os.getenv("CACHE_DIR"),
            tmp_dir=os.getenv("TMP_DIR"),
        )
        base_options = python.BaseOptions(model_asset_path=task, delegate=python.BaseOptions.Delegate.CPU)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            output_segmentation_masks=False,
            running_mode=mp.tasks.vision.RunningMode.LIVE_STREAM,
            result_callback=print_result
        )
        self.detector = vision.PoseLandmarker.create_from_options(options)

        # # Optionally change the thresholds:
        # self.model.min_score_thresh = float(self.options["min_score_thresh"])
        # self.model.min_suppression_thresh = float(
        #     self.options["min_suppression_thresh"]
        # )
        # self.batch_size = int(self.options["batch_size"])
        # self.force_square_ar = string_to_bool(self.options["force_square_ar"])
        # self.repeat_last = string_to_bool(self.options["repeat_last"])
        #
        # # TODO allow detection of more than one face
        # self.num_faces = 1

    def _post_process_sample(self, x):
        return x
        # Convert to numpy and remove z axis, presence and visability
        out = []
        for frame in x:
            frame_np = []
            for landmark in frame:
                frame_np.extend([landmark.y, landmark.x])


        # # Move to cpu
        # x = x.cpu().numpy()
        #
        # # No faces
        # if x.shape[0] == 0:
        #     return np.zeros((self.num_faces, 17))
        #
        # # Too few faces
        # elif x.shape[0] < self.num_faces:
        #     pred = np.concatenate(
        #         (x, np.zeros((self.num_faces - x.shape[0], 17)))
        #     )
        #
        # else:
        #     # Too many faces
        #     # sort by confidence and get num_faces
        #     idx = np.argsort(x[:, 16])[-self.num_faces:]
        #     pred = x[idx]
        #
        # # Adapt aspect ratio to original image
        # if self.force_square_ar:
        #
        #     meta_data = self.session_manager.input_data[INPUT_ID].meta_data
        #     orig_height = meta_data.sample_shape[-3]
        #     orig_width = meta_data.sample_shape[-2]
        #
        #     h_gt_w = orig_height > orig_width
        #
        #     # Stretch width
        #     if h_gt_w:
        #         # Ratio height to width
        #         ratio = orig_height / orig_width
        #
        #         # Difference of the current bounding box width to the scaled bounding box width
        #         bb_width = abs(pred[:, 3] - pred[:, 1])
        #         diff_x_scaled = bb_width * ratio - bb_width
        #
        #         # Adding half the distance to the end and abstract half distance from the beginning
        #         pred[:, 1] = pred[:, 1] - diff_x_scaled / 2
        #         pred[:, 3] = pred[:, 3] + diff_x_scaled / 2
        #
        #     # Stretch height
        #     else:
        #         # Ratio width to height
        #         ratio = orig_width / orig_height
        #
        #         # Difference of the current bounding box height to the scaled bounding box height
        #         bb_height = abs(pred[:, 2] - pred[:, 0])
        #         diff_y_scaled = bb_height * ratio - bb_height
        #
        #         # Adding half the distance to the end and abstract half distance from the beginning
        #         pred[:, 0] = pred[:, 0] - diff_y_scaled / 2
        #         pred[:, 2] = pred[:, 2] + diff_y_scaled / 2
        #
        #     h_bb = abs(pred[:, 2] - pred[:, 0]) * orig_height
        #     w_bb = abs(pred[:, 3] - pred[:, 1]) * orig_width
        #
        #     # Maximum difference between length and width in pixels to still be considered square. Compensates for rounding errors.
        #     max_ar_diff = 1
        #     if abs(int(h_bb) - int(w_bb)) > max_ar_diff:
        #         raise ValueError(f'Assertion Error: Bounding box aspect ratio is forced to be 1:1 but {h_bb / w_bb} ')
        #
        # return pred

    def process_data(self, ds_manager) -> tuple:
        self.session_manager = self.get_session_manager(ds_manager)
        data_object = self.session_manager.input_data[INPUT_ID]
        data = data_object.data

        # Append batch dimension
        if isinstance(data_object, discover_utils.data.static.Image):
            data = np.expand_dims(data, 0)

        predictions = []
        # self.orig_height, self.orig_width, self.channels = data.shape[1:]
        start = time.perf_counter()
        for i in range(0, len(data), self.batch_size):
            idx_start = i
            idx_end = (
                idx_start + self.batch_size
                if idx_start + self.batch_size <= len(data)
                else len(data)
            )
            idxs = list(range(idx_start, idx_end))
            if idx_start % 100 == 0 and idx_start > 0:
                log(f"Processed: {idx_start} frames / {(idx_start + 1) / (time.perf_counter() - start) } FPS ")
            #log(f"Batch {i / self.batch_size} : {idx_start} - {idx_end}")
            if not idxs:
                continue

            frame = np.asarray(data[idxs])
            # TODO: Add batch support
            frame = frame[0]
            #image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)#data=cv2.cvtColor(frame, cv2.COLOR_RGB2RGB))
            #detections = self.detector.detect(int((idx_start + 1) * 1000 / data_object.meta_data.sample_rate), image)
            detections = self.detector.detect_async(image, int((idx_start) * 1000 / data_object.meta_data.sample_rate))
            #predictions.append(detections.pose_landmarks)

            ### TODO: DEBUG ONLY ###
            if idx_end >= 500:
                break


        # if self.repeat_last:
        #     # Init pose box with 33 empty landmarks
        #     last_p = [NormalizedLandmark()] * 33
        #     for i in predictions:
        #         if True:
        #             ...

        #last_p[:, 2:4] = 1
        # for i, p in enumerate(predictions):
        #     if p[-1] == 0:
        #         predictions[i] = last_p
        #         predictions[i][-1] = 0
        #     else:
        #         last_p = p

        # Adjust format
        #predictions = np.concatenate(
        #    [self._post_process_sample(x) for x in predictions]
        #)

        # Landmarks flip x,y order to y,x order
        #lm = predictions[:, 4:]
        #lm = lm[:, [x - 2 if x % 2 == 0 else x for x in range(1, lm.shape[1])] + [-1]]

        return predictions

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

        self.session_manager.output_data_templates[OUTPUT_ID] = create_stream(
            data[0],
            self.session_manager.output_data_templates[OUTPUT_ID],
            self.session_manager.input_data[INPUT_ID],
            _dl,
            MEDIA_TYPE_ID_BB,
        )

        return self.session_manager.output_data_templates


if __name__ == "__main__":
    import dotenv
    from pathlib import Path
    from discover_utils.utils.ssi_xml_utils import Trainer
    from discover_utils.data.provider.data_manager import DatasetManager
    from mediapipe.framework.formats import landmark_pb2
    from mediapipe import solutions
    from matplotlib import pyplot as plt


    def draw_landmarks_on_image(rgb_image, detection_result):
        pose_landmarks_list = detection_result
        annotated_image = np.copy(rgb_image)

        # Loop through the detected poses to visualize.
        for idx in range(len(pose_landmarks_list)):
            pose_landmarks = pose_landmarks_list[idx]

            # Draw the pose landmarks.
            pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            pose_landmarks_proto.landmark.extend([
                landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
            ])
            solutions.drawing_utils.draw_landmarks(
                annotated_image,
                pose_landmarks_proto,
                solutions.pose.POSE_CONNECTIONS,
                solutions.drawing_styles.get_default_pose_landmarks_style())
        return annotated_image

    dotenv.load_dotenv()
    base_dir = Path(os.getenv("DISCOVER_DATA_DIR"))
    out_dir = Path(os.getenv("DISCOVER_TEST_DIR"))
    stream_out = Path(out_dir /"blaze_pose_out.stream")

    image = False
    video = True

    bp_trainer = Trainer()
    bp_trainer.load_from_file("blazepose.trainer")
    bp = BlazePose(
        model_io=None,
        trainer=bp_trainer,
        opts={},
    )

    if image:
        img_in = Path(base_dir / "test_files" / "test_pose.jpg")

        dd_input_image = {
            "src": "file:image",
            "type": "input",
            "id": INPUT_ID,
            "uri": str(img_in),
        }

        dd_output = {
            "src": "file:stream",
            "type": "output",
            "id": OUTPUT_ID,
            "uri": str(stream_out),
        }

        # Create dataset from image
        dm_image = DatasetManager([dd_input_image, dd_output])
        dm_image.load()

        # Predict
        detection_result = bp.process_data(dm_image)


        # Get original input image for plotting
        input_image = bp.get_session_manager(dm_image).input_data[INPUT_ID].data
        annotated_image = draw_landmarks_on_image(input_image, detection_result[0])


        plt.imshow(annotated_image)
        plt.show()
        exit()

        # Save to Disk
        # for k, v in bf.to_output(output).items():
        #     sm_image.output_data_templates[k] = v
        # sm_image.save()

    if video:
        video_in = Path(base_dir / "test_files" / "patient.video.mp4")

        dd_input_video = {
            "src": "file:stream:video",
            "type": "input",
            "id": INPUT_ID,
            "uri": str(video_in),
        }

        dd_output = {
            "src": "file:stream",
            "type": "output",
            "id": OUTPUT_ID,
            "uri": str(stream_out),
        }

        # Create dataset from image
        dm_video = DatasetManager([dd_input_video, dd_output])
        dm_video.load()

        # Predict
        detection_result = bp.process_data(dm_video)


        # Get original input image for plotting
        input_ = bp.get_session_manager(dm_video).input_data[INPUT_ID].data

        for i in range(0,500,30):
            annotated_image = draw_landmarks_on_image(input_[i], detection_result[i])
            plt.imshow(annotated_image)
            plt.show()
        exit()



        dm_video = DatasetManager([dd_input_video, dd_output_bb, dd_output_lm])
        sm_video = bf.get_session_manager(dm_video)
        dm_video.load()
        output = bf.process_data(dm_video)
        video = sm_video.input_data[INPUT_ID].data
        plot_detections(video[100], output[0][100], output[1][100])

        for k, v in bf.to_output(output).items():
            sm_video.output_data_templates[k] = v
        sm_video.save()

