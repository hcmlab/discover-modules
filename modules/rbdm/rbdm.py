import cv2
import sys
import os
import numpy as np

# Add local dir to path for relative imports
sys.path.insert(0, os.path.dirname(__file__))

from rbdm_source import MultitaskMobileNetV2Model
from discover_utils.data.annotation import (
     DiscreteAnnotationScheme, ContinuousAnnotationScheme
)
from discover_utils.data.static import Image
from discover_utils.interfaces.server_module import Processor
from discover_utils.utils.anno_utils import resample
from discover_utils.utils.cache_utils import get_file
from discover_utils.utils.log_utils import log


INPUT_ID = "input_data"
INPUT_ID_BB = "face_bb"
OUTPUT_ID_FE = "expression"
OUTPUT_ID_AROUSAL = "valence"
OUTPUT_ID_VALENCE = "arousal"

_default_options = {
    "batch_size": 16
}

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

def _blaze_face_converter(detection: np.ndarray, img_shape: tuple):
    ymin = detection[:, 0] * img_shape[0]
    xmin = detection[:, 1] * img_shape[1]
    ymax = detection[:, 2] * img_shape[0]
    xmax = detection[:, 3] * img_shape[1]
    return np.swapaxes(np.vstack([xmin, ymin, xmax, ymax]), 0, 1)

def _get_low_confidence_detections(detection: np.ndarray, thresh=0.5):
    return np.where(detection[:,-1] < thresh)[0]


class RBDM(Processor):
    chainable = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.options = _default_options | self.options

        # Download weights
        weights_uri = next(
            filter(lambda x: x.uri_id == 'weights', self.trainer.meta_uri), None
        )
        weight_file = get_file(
            fname=weights_uri.uri_url.split("=")[-1],
            origin=weights_uri.uri_url,
            file_hash=weights_uri.uri_hash,
            cache_dir=os.getenv("CACHE_DIR"),
            tmp_dir=os.getenv("TMP_DIR"),
        )


        # Set Module Variables
        self._conf_thresh = 0.5
        self.img_width = 224
        self.img_height = 224
        self.zero_img = np.zeros((self.img_width, self.img_height, 3))
        self.batch_size = self.options['batch_size']
        self.no_face_value = 0


        # Build model
        mobile_net = MultitaskMobileNetV2Model(weights=weight_file, input_width=self.img_width, input_height=self.img_height)
        mobile_net.model.load_weights(weight_file)
        self.model = mobile_net.model

    def process_data(self, ds_manager) -> dict:
        predictions = {OUTPUT_ID_VALENCE: [], OUTPUT_ID_AROUSAL: [], OUTPUT_ID_FE: []}


        self.session_manager = self.get_session_manager(ds_manager)
        data_object = self.session_manager.input_data[INPUT_ID]
        data = data_object.data
        face_bb = self.session_manager.input_data[INPUT_ID_BB].data

        # # Append batch dimension
        if isinstance(data_object, Image):
             data = np.expand_dims(data, 0)
        orig_height, orig_width, channels = data.shape[-3:]

        iter_len = min(len(data), len(face_bb))
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

            # shape: (num_samples, height, width, num_channels)
            frame = data[idxs]
            bb_frame = face_bb[idxs]
            bb_low_conf = _get_low_confidence_detections(bb_frame)
            bb_frame = _blaze_face_converter(bb_frame, (orig_height, orig_width))
            bb_frame = bb_frame.astype(int)

            image_cropped = [
                    img[
                    bbf[1]:bbf[3],
                    bbf[0]:bbf[2],
                    :
                    ] if not all(bbf == 0) else img
                for img, bbf in
                zip(frame, bb_frame)
            ]

            image_resized = np.asarray([cv2.resize(x,  (self.img_width, self.img_height)) for x in image_cropped])

            if len(image_resized.shape) == 3:
                image_resized = np.expand_dims(image_resized, 0)

            pred = self.model(image_resized.astype(np.float32))

            # Expression
            expression = pred[1].cpu().numpy()

            # Handling rest class
            expression = np.concatenate((expression, np.zeros((len(expression),1))), axis=-1)
            rest_class_tensor = np.zeros(expression.shape[-1], dtype=expression.dtype)
            rest_class_tensor[-1] = 1
            expression[bb_low_conf] = rest_class_tensor
            predictions[OUTPUT_ID_FE].append(expression)

            # Valence Arousal
            va = pred[0].cpu().numpy()

            valence = va[:,0]
            valence[bb_low_conf] = self.no_face_value
            predictions[OUTPUT_ID_VALENCE].append(valence)

            arousal = va[:,1]
            arousal[bb_low_conf] = self.no_face_value
            predictions[OUTPUT_ID_AROUSAL].append(arousal)

        for k in predictions.keys():
            predictions[k] = np.concatenate(predictions[k])

        return predictions

    def to_output(self, data: dict) -> dict:
        input_stream = self.session_manager.input_data[INPUT_ID]
        sr_hz = 1 if isinstance(input_stream, Image) else input_stream.meta_data.sample_rate
        for l, d in data.items():
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

        return self.session_manager.output_data_templates



if __name__ == "__main__":

    import matplotlib.pyplot as plt
    from pathlib import Path
    def plot_detections(img, detections,):
        fig, ax = plt.subplots(1, figsize=(10, 10))
        ax.grid(False)
        ax.imshow(img.data)

        valence = detections[OUTPUT_ID_VALENCE][0]
        arousal = detections[OUTPUT_ID_AROUSAL][0]
        id = np.argmax(detections[OUTPUT_ID_FE])
        expression = _label_map.get(str(id), 'REST')
        fp = img.meta_data.file_path
        fig.suptitle(f'EX: {expression}, V: {valence}, A: {arousal}')
        plt.title(fp)

        plt.show()

    PYTORCH_ENABLE_MPS_FALLBACK = 1
    from discover_utils.utils.ssi_xml_utils import Trainer
    from discover_utils.data.provider.data_manager import DatasetManager
    import dotenv
    env = dotenv.load_dotenv(r'../.env')
    TEST_DIR = Path(os.getenv("TEST_DIR")) / 'emonet'


    en_trainer = Trainer()
    en_trainer.load_from_file("rbdm_mt.trainer")
    rbdm = RBDM(model_io=None, opts={}, trainer=en_trainer)

    for img in TEST_DIR.glob('*.jpg'):

        bb_stream_file = img.with_name(img.stem + '_bb').with_suffix('.stream')
        if not bb_stream_file.is_file():
            print('No bounding box found for file {img}')
            continue

        # Inputs
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

        # Outputs
        dd_output_fe = {
            "src": "file:annotation:discrete",
            "type": "output",
            "id": OUTPUT_ID_FE,
            "uri": str(img.parent / 'emonet_facial_expression.annotation')
        }

        # Outputs
        dd_output_arousal = {
            "src": "file:annotation:continuous",
            "type": "output",
            "id": OUTPUT_ID_AROUSAL,
            "uri": str(img.parent / 'arousal.annotation')
        }

        # Outputs
        dd_output_valence = {
            "src": "file:annotation:continuous",
            "type": "output",
            "id": OUTPUT_ID_VALENCE,
            "uri": str(img.parent / 'valence.annotation')
        }

        dm_image = DatasetManager([dd_input_image, dd_input_bb, dd_output_fe, dd_output_valence, dd_output_arousal])
        dm_image.load()
        data = rbdm.process_data(dm_image)

        output = rbdm.to_output(data)

        image = rbdm.get_session_manager(dm_image).input_data[INPUT_ID]
        plot_detections(image, data)
