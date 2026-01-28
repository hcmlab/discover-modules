"""BlazePose Module
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
from mediapipe.tasks.python.vision import PoseLandmarker

# Add local dir to path for relative imports
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import mediapipe as mp

from ssi_skeleton import SSISkeleton, get_dim_labels
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import PoseLandmarkerResult

from discover_utils.interfaces.server_module import Processor
from discover_utils.utils.cache_utils import get_file
from discover_utils.utils.log_utils import log
from discover_utils.data.static import Image
from discover_utils.utils.string_utils import string_to_bool
from discover_utils.data.stream import SSIStream

INPUT_ID = "video"
OUTPUT_ID_SSI = "pose_ssi"
OUTPUT_ID_MP = "pose_mp"

MEDIA_TYPE_ID_POSE_SSI = "stream:SSIStream:feature;body;skeleton;blazepose_ssi"
MEDIA_TYPE_ID_POSE_MP = "stream:SSIStream:feature;body;skeleton;blazepose_mp"

_default_options = {
    "repeat_last": True,
    "running_mode": "image",
    "model": "full",
    "ssi_format": True,
    "mp_format": True
}

_dl = get_dim_labels()

_dl = [{"id": i, "name": x} for i, x in enumerate(_dl)]


class BlazePose(Processor):
    chainable = False

    def get_running_mode(self, string):
        if string == "video":
            return mp.tasks.vision.RunningMode.VIDEO
        elif string == "live_stream":
            return mp.tasks.vision.RunningMode.LIVE_STREAM
        return mp.tasks.vision.RunningMode.IMAGE

    def __init__(self, *args, **kwargs):
        """

        Args:
            *args: Optional args for parent classes
            **kwargs: *args: Optional kwargs for parent classes
        """

        # Setting options
        super().__init__(*args, **kwargs)
        self.options = _default_options | self.options
        self.repeat_last = string_to_bool(self.options["repeat_last"])
        self.num_poses = 1
        self.running_mode = self.get_running_mode(self.options["running_mode"])
        self.model = self.options["model"]
        self.output_ssi = self.options["ssi_format"]
        self.output_mp = self.options["mp_format"]
        print(f"Running mode: {self.running_mode}")

        # Only relevant if used in live_stream mode
        self.detections = []

        def live_callback(result, output_image: mp.Image, timestamp_ms: int):
            self.detections.append((result, timestamp_ms))

        # Download weights and Anchors
        task_uri = next(filter(lambda x: x.uri_id == f"task_file_{self.model}", self.trainer.meta_uri), None)

        # Build model
        task = get_file(
            fname=task_uri.uri_url.split("/")[-1],
            origin=task_uri.uri_url,
            file_hash=task_uri.uri_hash,
            cache_dir=os.getenv("CACHE_DIR"),
            tmp_dir=os.getenv("TMP_DIR"),
        )
        base_options = python.BaseOptions(model_asset_path=str(task), delegate=python.BaseOptions.Delegate.CPU)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            output_segmentation_masks=False,
            running_mode=self.running_mode,
            result_callback=live_callback if self.running_mode == mp.tasks.vision.RunningMode.LIVE_STREAM else None
        )
        self.detector = vision.PoseLandmarker.create_from_options(options)


    def process_data(self, ds_manager) -> list:
        self.session_manager = self.get_session_manager(ds_manager)
        data_object = self.session_manager.input_data[INPUT_ID]
        data = data_object.data

        # Append batch dimension
        if isinstance(data_object, discover_utils.data.static.Image):
            data = np.expand_dims(data, 0)

        self.detections = []
        start = time.perf_counter()
        print_interval = 100
        for i, frame in enumerate(data):
            if i % print_interval == 0 and i > 0:
                log(f"Processed: {i} frames / {print_interval / (time.perf_counter() - start)} FPS ")
                start = time.perf_counter()

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            if self.running_mode == mp.tasks.vision.RunningMode.IMAGE:
                self.detections.append((self.detector.detect(mp_image), i))
            else:
                #time_stamp_ms = int((i) * 1000 / data_object.meta_data.sample_rate)
                time_stamp_ms = i
                if self.running_mode == mp.tasks.vision.RunningMode.VIDEO:
                    self.detections.append((self.detector.detect_for_video(mp_image, time_stamp_ms), time_stamp_ms))
                if self.running_mode == mp.tasks.vision.RunningMode.LIVE_STREAM:
                    self.detector.detect_async(mp_image, time_stamp_ms)

            # ### TODO: DEBUG ONLY ###
            # if i == 500:
            #     break

        if self.running_mode == mp.tasks.vision.RunningMode.LIVE_STREAM:
            # Wait for async tasks to finish
            timeout = 5
            for t in range(timeout):
                if len(self.detections) == i+1:
                    break
                print(f"Wait for last predictions to finish...")
                time.sleep(1)

        # Sort results according to time stamps
        self.detections.sort(key=lambda tup: tup[1])

        # Add missing frames
        frame_idxs = [x[1] for x in self.detections]
        detections_ = []
        filler_landmarks = PoseLandmarkerResult([],[])
        for idx in range(i):
            if idx not in frame_idxs:
                detections_.append(filler_landmarks)
            else:
                detections_.append(self.detections[idx])
                if self.repeat_last:
                    filler_landmarks = self.detections[idx]

        self.detections = detections_


            # for idx in range(i):
            #             if idx >= len(self.detections) or self.detections[idx+1][1] - self.detections[idx][1] != 1:
            #                 self.detections.insert(idx, (None, None))



            # if self.repeat_last:
            #     # Init pose box with 33 empty landmarks
            #     last_p = [NormalizedLandmark()] * 33
            #     for i in predictions:
            #         if True:
            #             ...

        # last_p[:, 2:4] = 1
        # for i, p in enumerate(predictions):
        #     if p[-1] == 0:
        #         predictions[i] = last_p
        #         predictions[i][-1] = 0
        #     else:
        #         last_p = p

        # Adjust format
        # predictions = np.concatenate(
        #    [self._post_process_sample(x) for x in predictions]
        # )

        # Landmarks flip x,y order to y,x order
        # lm = predictions[:, 4:]
        # lm = lm[:, [x - 2 if x % 2 == 0 else x for x in range(1, lm.shape[1])] + [-1]]

        return [x for x, _ in self.detections]

    def convert_to_ssi_skeleton(self, frame: PoseLandmarkerResult) -> list:
        # Media pipe mapping: https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker
        # SSI Mapping: https://hcai.eu/svn/Johannes/openssi/trunk/core/include/SSI_SkeletonCons.h


        def unpack_landmarks(pl: NormalizedLandmark):
            return pl.x, pl.y, pl.z, pl.visibility, pl.presence

        def normalize_landmarks(x,y,z,visibility,presence):
            return (x*2)-1, ((1-y)*2)-1, z, visibility, presence

        def average_landmarks(landmarks: list[NormalizedLandmark]):
            x, y, z, visibility, presence = 0, 0, 0, 0, 0
            for l in landmarks:
                x += l.x
                y += l.y
                z += l.z
                visibility += l.visibility
                presence += l.presence

            x = x / len(landmarks)
            y = y / len(landmarks)
            z = z / len(landmarks)
            visibility = visibility / len(landmarks)
            presence = presence / len(landmarks)

            return x, y, z, visibility, presence

        # Assign to SSISkeleton and calculate missing values
        # Skeletons
        skeletons = []
        for person_id, pose_landmark_list in enumerate(frame.pose_landmarks):
            ssi_skel = SSISkeleton()
            # 0 - HEAD
            ssi_skel.HEAD.POS_X, ssi_skel.HEAD.POS_Y, ssi_skel.HEAD.POS_Z, _, ssi_skel.HEAD.POS_CONF = normalize_landmarks( *unpack_landmarks(pose_landmark_list[0]) )

            # 1 - NECK
            ssi_skel.NECK.POS_X, ssi_skel.NECK.POS_Y, ssi_skel.NECK.POS_Z, _, ssi_skel.NECK.POS_CONF  = normalize_landmarks( *average_landmarks([pose_landmark_list[11], pose_landmark_list[12]]) )

            # 2 - TORSO
            ssi_skel.TORSO.POS_X, ssi_skel.TORSO.POS_Y, ssi_skel.TORSO.POS_Z, _, ssi_skel.TORSO.POS_CONF  = normalize_landmarks( *average_landmarks([pose_landmark_list[23], pose_landmark_list[24], pose_landmark_list[11], pose_landmark_list[12]]) )

            # 3 - WAIST
            ssi_skel.WAIST.POS_X, ssi_skel.WAIST.POS_Y, ssi_skel.WAIST.POS_Z, _, ssi_skel.WAIST.POS_CONF  = normalize_landmarks( *average_landmarks([pose_landmark_list[23], pose_landmark_list[24]]) )

            # 4 - LEFT_SHOULDER
            ssi_skel.LEFT_SHOULDER.POS_X, ssi_skel.LEFT_SHOULDER.POS_Y, ssi_skel.LEFT_SHOULDER.POS_Z, _, ssi_skel.LEFT_SHOULDER.POS_CONF = normalize_landmarks(*unpack_landmarks(pose_landmark_list[11]))

            # 5 - LEFT_ELBOW
            ssi_skel.LEFT_ELBOW.POS_X, ssi_skel.LEFT_ELBOW.POS_Y, ssi_skel.LEFT_ELBOW.POS_Z, _, ssi_skel.LEFT_ELBOW.POS_CONF = normalize_landmarks(*unpack_landmarks(pose_landmark_list[13]))

            # 6 - LEFT_WRIST
            ssi_skel.LEFT_WRIST.POS_X, ssi_skel.LEFT_WRIST.POS_Y, ssi_skel.LEFT_WRIST.POS_Z, _, ssi_skel.LEFT_WRIST.POS_CONF = normalize_landmarks(*unpack_landmarks(pose_landmark_list[15]))

            # 7 - LEFT_HAND
            ssi_skel.LEFT_HAND.POS_X, ssi_skel.LEFT_HAND.POS_Y, ssi_skel.LEFT_HAND.POS_Z, _, ssi_skel.LEFT_HAND.POS_CONF = normalize_landmarks(*unpack_landmarks(pose_landmark_list[19]))

            # 8 - RIGHT_SHOULDER
            ssi_skel.RIGHT_SHOULDER.POS_X, ssi_skel.RIGHT_SHOULDER.POS_Y, ssi_skel.RIGHT_SHOULDER.POS_Z, _, ssi_skel.RIGHT_SHOULDER.POS_CONF = normalize_landmarks(*unpack_landmarks(pose_landmark_list[12]))

            # 9 - RIGHT_ELBOW
            ssi_skel.RIGHT_ELBOW.POS_X, ssi_skel.RIGHT_ELBOW.POS_Y, ssi_skel.RIGHT_ELBOW.POS_Z, _, ssi_skel.RIGHT_ELBOW.POS_CONF = normalize_landmarks(*unpack_landmarks(pose_landmark_list[14]))

            # 10 - RIGHT_WRIST
            ssi_skel.RIGHT_WRIST.POS_X, ssi_skel.RIGHT_WRIST.POS_Y, ssi_skel.RIGHT_WRIST.POS_Z, _, ssi_skel.RIGHT_WRIST.POS_CONF = normalize_landmarks(*unpack_landmarks(pose_landmark_list[16]))

            # 11 - RIGHT_HAND
            ssi_skel.RIGHT_HAND.POS_X, ssi_skel.RIGHT_HAND.POS_Y, ssi_skel.RIGHT_HAND.POS_Z, _, ssi_skel.RIGHT_HAND.POS_CONF = normalize_landmarks(*unpack_landmarks(pose_landmark_list[20]))

            # 12 - LEFT_HIP
            ssi_skel.LEFT_HIP.POS_X, ssi_skel.LEFT_HIP.POS_Y, ssi_skel.LEFT_HIP.POS_Z, _, ssi_skel.LEFT_HIP.POS_CONF = normalize_landmarks(*unpack_landmarks(pose_landmark_list[23]))

            # 13 - LEFT_KNEE
            ssi_skel.LEFT_KNEE.POS_X, ssi_skel.LEFT_KNEE.POS_Y, ssi_skel.LEFT_KNEE.POS_Z, _, ssi_skel.LEFT_KNEE.POS_CONF = normalize_landmarks(*unpack_landmarks(pose_landmark_list[25]))

            # 14 - LEFT_ANKLE
            ssi_skel.LEFT_ANKLE.POS_X, ssi_skel.LEFT_ANKLE.POS_Y, ssi_skel.LEFT_ANKLE.POS_Z, _, ssi_skel.LEFT_ANKLE.POS_CONF = normalize_landmarks(*unpack_landmarks(pose_landmark_list[27]))

            # 15 - LEFT_FOOT
            ssi_skel.LEFT_FOOT.POS_X, ssi_skel.LEFT_FOOT.POS_Y, ssi_skel.LEFT_FOOT.POS_Z, _, ssi_skel.LEFT_FOOT.POS_CONF = normalize_landmarks(*unpack_landmarks(pose_landmark_list[31]))

            # 16 - RIGHT_HIP
            ssi_skel.RIGHT_HIP.POS_X, ssi_skel.RIGHT_HIP.POS_Y, ssi_skel.RIGHT_HIP.POS_Z, _, ssi_skel.RIGHT_HIP.POS_CONF = normalize_landmarks(*unpack_landmarks(pose_landmark_list[24]))

            # 17 - RIGHT_KNEE
            ssi_skel.RIGHT_KNEE.POS_X, ssi_skel.RIGHT_KNEE.POS_Y, ssi_skel.RIGHT_KNEE.POS_Z, _, ssi_skel.RIGHT_KNEE.POS_CONF = normalize_landmarks(*unpack_landmarks(pose_landmark_list[26]))

            # 18 - RIGHT_ANKLE
            ssi_skel.RIGHT_ANKLE.POS_X, ssi_skel.RIGHT_ANKLE.POS_Y, ssi_skel.RIGHT_ANKLE.POS_Z, _, ssi_skel.RIGHT_ANKLE.POS_CONF = normalize_landmarks(*unpack_landmarks(pose_landmark_list[28]))

            # 19 - RIGHT_FOOT
            ssi_skel.RIGHT_FOOT.POS_X, ssi_skel.RIGHT_FOOT.POS_Y, ssi_skel.RIGHT_FOOT.POS_Z, _, ssi_skel.RIGHT_FOOT.POS_CONF = normalize_landmarks(*unpack_landmarks(pose_landmark_list[32]))

            # 20 - FACE_NOSE
            ssi_skel.FACE_NOSE.POS_X, ssi_skel.FACE_NOSE.POS_Y, ssi_skel.FACE_NOSE.POS_Z, _, ssi_skel.FACE_NOSE.POS_CONF = normalize_landmarks(*unpack_landmarks(pose_landmark_list[0]))

            # 21 - FACE_LEFT_EAR
            ssi_skel.FACE_LEFT_EAR.POS_X, ssi_skel.FACE_LEFT_EAR.POS_Y, ssi_skel.FACE_LEFT_EAR.POS_Z, _, ssi_skel.FACE_LEFT_EAR.POS_CONF = normalize_landmarks(*unpack_landmarks(pose_landmark_list[7]))

            # 21 - FACE_RIGHT_EAR
            ssi_skel.FACE_RIGHT_EAR.POS_X, ssi_skel.FACE_RIGHT_EAR.POS_Y, ssi_skel.FACE_RIGHT_EAR.POS_Z, _, ssi_skel.FACE_RIGHT_EAR.POS_CONF = normalize_landmarks(*unpack_landmarks(pose_landmark_list[8]))

            # 22 - FACE_FOREHEAD
            eyes_avg_x, eyes_avg_y, eyes_avg_z, eyes_avg_visibility, eyes_avg_presence = normalize_landmarks( *average_landmarks([pose_landmark_list[1], pose_landmark_list[2], pose_landmark_list[3], pose_landmark_list[4], pose_landmark_list[5], pose_landmark_list[6] ]))
            ssi_skel.FACE_FOREHEAD.POS_X = ssi_skel.FACE_NOSE.POS_X + 2 * abs(eyes_avg_x-ssi_skel.FACE_NOSE.POS_X)
            ssi_skel.FACE_FOREHEAD.POS_Y = ssi_skel.FACE_NOSE.POS_Y + 2 * abs(eyes_avg_y-ssi_skel.FACE_NOSE.POS_Y)
            ssi_skel.FACE_FOREHEAD.POS_Z = ssi_skel.FACE_NOSE.POS_Z + 2 * abs(eyes_avg_z-ssi_skel.FACE_NOSE.POS_Z)
            ssi_skel.FACE_FOREHEAD.POS_CONF = (ssi_skel.FACE_NOSE.POS_CONF + eyes_avg_presence)  / 2

            # 23 - FACE_CHIN
            mouth_avg_x, mouth_avg_y, mouth_avg_z, mouth_avg_visibility, mouth_avg_presence = normalize_landmarks( *average_landmarks([pose_landmark_list[9], pose_landmark_list[10] ]))
            ssi_skel.FACE_CHIN.POS_X = ssi_skel.FACE_NOSE.POS_X + 2 * abs(mouth_avg_x-ssi_skel.FACE_NOSE.POS_X)
            ssi_skel.FACE_CHIN.POS_Y = ssi_skel.FACE_NOSE.POS_Y - 2 * abs(mouth_avg_y-ssi_skel.FACE_NOSE.POS_Y)
            ssi_skel.FACE_CHIN.POS_Z = ssi_skel.FACE_NOSE.POS_Z + 2 * abs(mouth_avg_z-ssi_skel.FACE_NOSE.POS_Z)
            ssi_skel.FACE_CHIN.POS_CONF = (ssi_skel.FACE_NOSE.POS_CONF + mouth_avg_presence)  / 2

            skeletons.append(ssi_skel)

        # Append empty skeleton in case of no detections. Alle values are 0.
        if not skeletons:
            skeletons.append(SSISkeleton())

        return skeletons

    def create_stream(self, stream_data, template, input_stream, dim_labels, media_type):
        return SSIStream(
            data=np.asarray(stream_data).astype(template.meta_data.dtype),
            sample_rate=1
            if isinstance(input_stream, Image)
            else input_stream.meta_data.sample_rate,
            dim_labels=dim_labels,
            media_type=media_type,
            custom_meta={
                "size": f"{input_stream.meta_data.sample_shape[-2]}:{input_stream.meta_data.sample_shape[-3]}",
                "num" : str(self.num_poses),
                "normalized" : "true"
            },
            role=template.meta_data.role,
            dataset=template.meta_data.dataset,
            name=template.meta_data.name,
            session=template.meta_data.session,
        )

    def to_output(self, data: list) -> dict:

        if True:
            ssi_skeleton_stream_data = []
            for frame in data:
                ssi_skels = self.convert_to_ssi_skeleton(frame)
                tmp_skels_ = []
                for skel in ssi_skels:
                    tmp_skels_.append(skel.to_numpy())

                ssi_skeleton_stream_data.append(np.hstack(tmp_skels_))

            self.session_manager.output_data_templates[OUTPUT_ID_SSI] = self.create_stream(
                ssi_skeleton_stream_data,
                self.session_manager.output_data_templates[OUTPUT_ID_SSI],
                self.session_manager.input_data[INPUT_ID],
                _dl,
                MEDIA_TYPE_ID_POSE_SSI
            )

        if True:
            mp_skeleton_stream_data = []
            for frame in data:
                mp_skels = []
                for skel in frame.pose_landmarks:
                    tmp_skel_ = []
                    for landmark in skel:
                        tmp_skel_.append(landmark.x)
                        tmp_skel_.append(landmark.y)
                        tmp_skel_.append(landmark.z)
                        tmp_skel_.append(landmark.visibility)
                        tmp_skel_.append(landmark.presence)
                    mp_skels.append(np.asarray(tmp_skel_))
                if not mp_skels:
                    mp_skels = np.zeros(shape=(165,))
                mp_skeleton_stream_data.append(np.hstack(mp_skels))
            self.session_manager.output_data_templates[OUTPUT_ID_MP] = self.create_stream(
            mp_skeleton_stream_data,
            self.session_manager.output_data_templates[OUTPUT_ID_MP],
            self.session_manager.input_data[INPUT_ID],
            _dl,
            MEDIA_TYPE_ID_POSE_MP
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
        pose_landmarks_list = detection_result.pose_landmarks
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

    def draw_ssi_landmarks_on_image(rgb_image, detection_result):
        annotated_image = np.copy(rgb_image)
        # Radius of circle
        radius = 10
        # Red color in BGR
        color = (0, 0, 255)
        # Line thickness of -1 px
        thickness = -1
        for skeleton in detection_result:
            for name, joint in skeleton.__dict__.items():
                annotated_image = cv2.circle(annotated_image, (int((joint.POS_X+1)/2*rgb_image.shape[1]), int((joint.POS_Y+1)/2* rgb_image.shape[0])), radius, color, thickness)
            return annotated_image


    dotenv.load_dotenv()
    base_dir = Path(os.getenv("DISCOVER_DATA_DIR"))
    out_dir = Path(os.getenv("DISCOVER_TEST_DIR"))
    stream_out_ssi = Path(out_dir / "blaze_pose_out_ssi.stream")
    stream_out_mp = Path(out_dir / "blaze_pose_out_mp.stream")

    running_mode = "live_stream"  # "video", "live_stream", "image"

    bp_trainer = Trainer()
    bp_trainer.load_from_file("blazepose.trainer")
    bp = BlazePose(
        model_io=None,
        trainer=bp_trainer,
        opts={"running_mode": running_mode, "model" : "light"},
    )

    if running_mode == "image":
        img_in = Path(base_dir / "test_files" / "test_pose.jpg")

        dd_input_image = {
            "src": "file:image",
            "type": "input",
            "id": INPUT_ID,
            "uri": str(img_in),
        }

        dd_output_ssi = {
            "src": "file:stream",
            "type": "output",
            "id": OUTPUT_ID_SSI,
            "uri": str(stream_out_ssi),
        }

        dd_output_mp = {
            "src": "file:stream",
            "type": "output",
            "id": OUTPUT_ID_SSI,
            "uri": str(stream_out_mp),
        }

        # Create dataset from image
        dm_image = DatasetManager([dd_input_image, dd_output_ssi, dd_output_mp])
        dm_image.load()

        # Predict
        detection_result = bp.process_data(dm_image)

        # To SSI-Stream
        detection_results_stream = bp.to_output(detection_result)

        # Get original input image for plotting
        input_image = bp.get_session_manager(dm_image).input_data[INPUT_ID].data

        #annotated_image = draw_landmarks_on_image(input_image, detection_result[0])
        annotated_image = draw_ssi_landmarks_on_image(input_image, detection_results_stream[0])

        plt.imshow(annotated_image)
        plt.show()

        # Save to Disk
        # for k, v in bf.to_output(output).items():
        #     sm_image.output_data_templates[k] = v
        # sm_image.save()

    if running_mode == "video" or running_mode == "live_stream":
        video_in = Path(base_dir / "test_files" / "test_video.mp4")

        dd_input_video = {
            "src": "file:stream:video",
            "type": "input",
            "id": INPUT_ID,
            "uri": str(video_in),
        }

        dd_output_ssi = {
            "src": "file:stream",
            "type": "output",
            "id": OUTPUT_ID_SSI,
            "uri": str(stream_out_ssi),
        }

        dd_output_mp = {
            "src": "file:stream",
            "type": "output",
            "id": OUTPUT_ID_MP,
            "uri": str(stream_out_mp),
        }

        # Create dataset from image
        dm_video = DatasetManager([dd_input_video, dd_output_ssi, dd_output_mp])
        dm_video.load()

        # Predict
        detection_result = bp.process_data(dm_video)

        # Get original input image for plotting
        input_ = bp.get_session_manager(dm_video).input_data[INPUT_ID].data

        # for i in range(0, 499, 30):
        #     annotated_image = draw_ssi_landmarks_on_image(input_[i], detection_result)
        #     plt.imshow(annotated_image)
        #     plt.show()

        # Save output
        sm_video = bp.get_session_manager(dm_video)
        for k, v in bp.to_output(detection_result).items():
            sm_video.output_data_templates[k] = v
        sm_video.save()

