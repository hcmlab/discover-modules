import sys
import os

# Add local dir to path for relative imports
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch
import torchvision

torchvision.disable_beta_transforms_warning()
from torchvision.transforms import v2
from discover_utils.data.annotation import (
    DiscreteAnnotationScheme,
    ContinuousAnnotationScheme,
)
from discover_utils.data.static import Image
from discover_utils.interfaces.server_module import Processor
from discover_utils.utils.anno_utils import resample
from discover_utils.utils.log_utils import log
from emonet_source import _EmoNet
from discover_utils.utils.cache_utils import get_file
from data_augmentor import DataAugmentor
from discover_utils.utils.type_definitions import SSINPDataType
from discover_utils.data.stream import SSIStream

INPUT_ID = "video"
INPUT_ID_BB = "face_bb"
OUTPUT_ID_EXPRESSION = "expression"
OUTPUT_ID_AROUSAL = "arousal"
OUTPUT_ID_VALENCE = "valence"
OUTPUT_ID_EMBEDDING = "embedding"

_default_options = {
    "num_expressions": 8,
    "batch_size": 16,
    "face_detection_min_conf" : 0.8,
    "temporal_smoothing": False
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

transform_image = v2.Compose([torchvision.transforms.v2.ConvertImageDtype()])


def _blaze_face_converter(detection: np.ndarray, img_shape: tuple):
    ymin = detection[:, 0] * img_shape[0]
    xmin = detection[:, 1] * img_shape[1]
    ymax = detection[:, 2] * img_shape[0]
    xmax = detection[:, 3] * img_shape[1]
    return np.swapaxes(np.vstack([xmin, ymin, xmax, ymax]), 0, 1)

def _get_low_confidence_detections(detection: np.ndarray, thresh=0.8):
    return np.where(detection[:,-1] < thresh)[0]


class EmoNet(Processor):
    chainable = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.options = _default_options | self.options

        # Build model
        self._device = (
            "cuda" if torch.cuda.is_available() else ("mps" if torch.has_mps else "cpu")
        )
        log(f"Using device {self._device}")

        # Load model
        self.n_classes = self.options["num_expressions"]
        model_id = f"emonet_{self.n_classes}"
        weights_uri = next(
            filter(lambda x: x.uri_id == model_id, self.trainer.meta_uri), None
        )

        weight_file = get_file(
            fname=weights_uri.uri_url.split("=")[-1],
            origin=weights_uri.uri_url,
            file_hash=weights_uri.uri_hash,
            cache_dir=os.getenv("CACHE_DIR"),
            tmp_dir=os.getenv("TMP_DIR"),
        )

        log(f"Loading the model from {weight_file}.")
        _net = _EmoNet(n_expression=self.n_classes, temporal_smoothing=self.options["temporal_smoothing"], device=self._device).to(self._device)
        state_dict = torch.load(str(weight_file), map_location=self._device)
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        _net.load_state_dict(state_dict, strict=False)
        _net.eval()
        torch.no_grad()
        self.model = _net
        self._conf_thresh = 0.5
        self.img_width = 256
        self.img_height = 256
        self.zero_img = np.zeros((self.img_width, self.img_height, 3))
        self.transform_image_shape = DataAugmentor(self.img_width, self.img_height)
        self.softmax = torch.nn.Softmax(dim=-1)
        self.batch_size = self.options['batch_size']
        self.face_detection_min_conf = self.options['face_detection_min_conf']
        self.no_face_value = 0


    def process_data(self, ds_manager) -> dict:

        predictions = {"valence": [], "arousal": [], "expression": [], "embedding": []}
        self.session_manager = self.get_session_manager(ds_manager)
        data_object = self.session_manager.input_data[INPUT_ID]
        data = data_object.data
        face_bb = self.session_manager.input_data[INPUT_ID_BB].data

        # Append batch dimension
        if isinstance(data_object, Image):
            data = np.expand_dims(data, 0)

        orig_height, orig_width, channels = data.shape[1:]

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
            frame = data[idxs]
            bb_frame = face_bb[idxs]
            bb_low_conf = _get_low_confidence_detections(bb_frame, self.face_detection_min_conf)
            bb_frame = _blaze_face_converter(bb_frame, (orig_height, orig_width))

            # Predict zero if no face is found to keep predictions consistent
            images_pp = [
                self.transform_image_shape(img, bb)[0] if i not in bb_low_conf else self.zero_img
                for i, img, bb in zip(range(len(bb_frame)), frame, bb_frame)
            ]
            image_tensor = transform_image(
                torch.from_numpy(np.asarray(images_pp, dtype=np.float32) / 255.).permute([0, 3, 1, 2])
            ).to(self._device)

            with torch.no_grad():
                pred = self.model.forward(image_tensor)

                for k in predictions.keys():
                    p = pred[k]
                    if k == "expression":
                        p = self.softmax(p)

                        # Add rest class
                        p = torch.cat((p, torch.zeros(len(idxs), 1).to(self._device)), dim=-1)

                        # Set rest class for all predictions where no face has been detected
                        rest_class_tensor = torch.zeros(p.shape[-1], dtype=p.dtype)
                        rest_class_tensor[-1] = 1
                        p[bb_low_conf] = rest_class_tensor.to(self._device)

                    elif k == "embedding":
                        # Don't clip embeddings, just set to zero for low confidence faces
                        p[bb_low_conf] = self.no_face_value
                    else:
                        # Clip valence/arousal to [-1, 1] range
                        p[bb_low_conf] = self.no_face_value
                        p = p.clip(-1,1)
                    predictions[k].append(p)

        for k in predictions.keys():
            predictions[k] = torch.cat(predictions[k]).cpu().numpy()

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

            # embeddings
            self.session_manager.output_data_templates[OUTPUT_ID_EMBEDDING] = SSIStream(
                data=np.array(data["embedding"], SSINPDataType.FLOAT.value),
                sample_rate=sr_hz
            )

        return self.session_manager.output_data_templates



if __name__ == "__main__":

    import matplotlib.pyplot as plt
    from pathlib import Path
    def plot_detections(img, detections):
        fig, ax = plt.subplots(1, figsize=(10, 10))
        ax.grid(False)
        ax.imshow(img.data)

        valence = detections['valence'][0]
        arousal = detections['arousal'][0]
        class_id = str(np.argmax(detections['expression']))
        expression = _label_map.get(class_id, 'rest_class')
        fp = img.meta_data.file_path
        fig.suptitle(f'V: {valence} | A: {arousal} | EX: {expression}')
        plt.title(fp)

        plt.show()

    PYTORCH_ENABLE_MPS_FALLBACK = 1
    from discover_utils.utils.ssi_xml_utils import Trainer
    from discover_utils.data.provider.data_manager import DatasetManager


    en_trainer = Trainer()
    en_trainer.load_from_file("emonet.trainer")
    en = EmoNet(model_io=None, opts={'num_expressions' : 5}, trainer=en_trainer)

    data_dir = Path(r'/Users/dominikschiller/Work/local_nova_dir/test_files/emonet')
    for img in data_dir.glob('*.jpg'):
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
            "id": OUTPUT_ID_EXPRESSION,
            "uri": str(img.parent / 'emonet_facial_expression.annotation')
        }

        dd_output_arousal = {
            "src": "file:annotation:discrete",
            "type": "output",
            "id": OUTPUT_ID_AROUSAL,
            "uri": str(img.parent / 'emonet_arousal.annotation')
        }

        dd_output_valence = {
            "src": "file:annotation:discrete",
            "type": "output",
            "id": OUTPUT_ID_VALENCE,
            "uri": str(img.parent / 'emonet_valence.annotation'),
        }

        dm_image = DatasetManager([dd_input_image, dd_input_bb, dd_output_valence, dd_output_arousal, dd_output_fe])
        dm_image.load()
        emotions = en.process_data(dm_image)
        image = en.get_session_manager(dm_image).input_data[INPUT_ID]
        plot_detections(image, emotions)
