"""
LibreFace Module
Author: Tobias Hallmen <tobias.hallmen@uni-a.de>
Date: 20.02.2025
cf. https://github.com/ihp-lab/LibreFace
"""

import os
import numpy as np
import scipy.ndimage
from PIL import Image
from multiprocess import Pool

from discover_utils.interfaces.server_module import Processor
from discover_utils.utils.log_utils import log
from time import perf_counter

from discover_utils.data.stream import SSIStream, Video
from discover_utils.data.annotation import DiscreteAnnotation
from discover_utils.utils.cache_utils import get_file, validate_file
#import onnxruntime as ort

import sys
#sys.path.insert(0, __file__[:-3])  # <path>/libreface.py -> <path>/libreface for imports from libreface folder
sys.path.insert(0, __file__[:-1-len(__file__.split('/')[-1])])  # <path>/libreface_script.py -> <path> for imports from <libreface> folder
from libreface.AU_Recognition.inference import get_au_intensities_and_detect_aus, get_au_intensities_and_detect_aus_video
from libreface.Facial_Expression_Recognition.inference import get_facial_expression, get_facial_expression_video
import torch
import pandas as pd
pd.set_option('future.no_silent_downcasting', True)


def image_align(img, face_landmarks, output_size=256,  # their shared onnx models expect 224 instead of 256
                transform_size=512, enable_padding=True, x_scale=1,
                y_scale=1, em_scale=0.1, alpha=False, pad_mode='const'):
    # img = my_draw_image_by_points(img, face_landmarks[36:60], 1, (0,255,0))
    # Align function from FFHQ dataset pre-processing step
    # https://github.com/NVlabs/ffhq-dataset/blob/master/download_ffhq.py

    # import PIL.Image
    # import scipy.ndimage
    # print(img.size)
    lm = np.array(face_landmarks)
    lm[:, 0] *= img.size[0]
    lm[:, 1] *= img.size[1]
    # lm_chin          = lm[0  : 17]  # left-right
    # lm_eyebrow_left  = lm[17 : 22]  # left-right
    # lm_eyebrow_right = lm[22 : 27]  # left-right
    # lm_nose          = lm[27 : 31]  # top-down
    # lm_nostrils      = lm[31 : 36]  # top-down
    lm_eye_right = lm[0:16]
    lm_eye_left = lm[16:32]
    lm_mouth_outer = lm[32:]
    # lm_mouth_inner   = lm[60 : 68]  # left-clockwise
    lm_mouth_outer_x = lm_mouth_outer[:, 0].tolist()
    left_index = lm_mouth_outer_x.index(min(lm_mouth_outer_x))
    right_index = lm_mouth_outer_x.index(max(lm_mouth_outer_x))
    # print(left_index,right_index)
    # Calculate auxiliary vectors.
    eye_left = np.mean(lm_eye_left, axis=0)
    # eye_left[[0,1]] = eye_left[[1,0]]
    eye_right = np.mean(lm_eye_right, axis=0)
    # eye_right[[0,1]] = eye_right[[1,0]]
    eye_avg = (eye_left + eye_right) * 0.5
    eye_to_eye = eye_right - eye_left
    # print(lm_mouth_outer)s
    mouth_avg = (lm_mouth_outer[left_index, :] + lm_mouth_outer[right_index, :]) / 2.0
    # mouth_avg[[0,1]] = mouth_avg[[1,0]]

    eye_to_mouth = mouth_avg - eye_avg
    # Choose oriented crop rectangle.
    x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
    x /= np.hypot(*x)
    x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
    x *= x_scale
    y = np.flipud(x) * [-y_scale, y_scale]
    c = eye_avg + eye_to_mouth * em_scale
    quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
    qsize = np.hypot(*x) * 2

    # Shrink.
    shrink = int(np.floor(qsize / output_size * 0.5))
    if shrink > 1:
        rsize = (int(np.rint(float(img.size[0]) / shrink)), int(np.rint(float(img.size[1]) / shrink)))
        img = img.resize(rsize, Image.LANCZOS)
        quad /= shrink
        qsize /= shrink

    # Crop.
    border = max(int(np.rint(qsize * 0.1)), 3)
    crop = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))), int(np.ceil(max(quad[:, 0]))),
            int(np.ceil(max(quad[:, 1]))))
    crop = (max(crop[0] - border, 0), max(crop[1] - border, 0), min(crop[2] + border, img.size[0]),
            min(crop[3] + border, img.size[1]))
    if crop[2] - crop[0] < img.size[0] or crop[3] - crop[1] < img.size[1]:
        img = img.crop(crop)
        quad -= crop[0:2]

    # Pad.
    pad = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))), int(np.ceil(max(quad[:, 0]))),
           int(np.ceil(max(quad[:, 1]))))
    pad = (max(-pad[0] + border, 0), max(-pad[1] + border, 0), max(pad[2] - img.size[0] + border, 0),
           max(pad[3] - img.size[1] + border, 0))
    if enable_padding and max(pad) > border - 4:
        pad = np.maximum(pad, int(np.rint(qsize * 0.3)))
        if pad_mode == 'const':
            img = np.pad(np.float32(img), ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), 'constant', constant_values=0)
        else:
            img = np.pad(np.float32(img), ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), 'reflect')
        h, w, _ = img.shape
        y, x, _ = np.ogrid[:h, :w, :1]
        mask = np.maximum(1.0 - np.minimum(np.float32(x) / pad[0], np.float32(w - 1 - x) / pad[2]),
                          1.0 - np.minimum(np.float32(y) / pad[1], np.float32(h - 1 - y) / pad[3]))
        blur = qsize * 0.02
        img += (scipy.ndimage.gaussian_filter(img, [blur, blur, 0]) - img) * np.clip(mask * 3.0 + 1.0, 0.0, 1.0)
        img += (np.median(img, axis=(0, 1)) - img) * np.clip(mask, 0.0, 1.0)
        img = np.uint8(np.clip(np.rint(img), 0, 255))
        if alpha:
            mask = 1 - np.clip(3.0 * mask, 0.0, 1.0)
            mask = np.uint8(np.clip(np.rint(mask * 255), 0, 255))
            img = np.concatenate((img, mask), axis=2)
            img = Image.fromarray(img, 'RGBA')
        else:
            img = Image.fromarray(img, 'RGB')
        quad += pad[:2]

    img = img.transform((transform_size, transform_size), Image.Transform.QUAD,
                        (quad + 0.5).flatten(), Image.Resampling.BILINEAR)
    out_image = img.resize((output_size, output_size), Image.Resampling.LANCZOS)

    return np.array(out_image)


_default_options = {'batch_size': 256}
INP_VID = "video"
INP_FM = 'facemesh'
OUT_LF = "libreface_stream"
OUT_LF_EMO = "libreface_emotion"
OUT_ALIGN = 'aligned'

detection = [1, 2, 4, 6, 7, 10, 12, 14, 15, 17, 23, 24]
regression = [1, 2, 4, 5, 6, 9, 12, 15, 17, 20, 25, 26]
emotions = ["Neutral", "Happiness", "Sadness", "Surprise", "Fear", "Disgust", "Anger", "Contempt"]
emo2id = {x: i for i, x in enumerate(emotions)}
DIM_LABELS = [f'AU{str(x)}_c' for x in detection]
DIM_LABELS.extend([f'AU{str(x)}_r' for x in regression])
#DIM_LABELS.extend(emotions)
DIM_LABELS = [{"id": i, "name": x} for i, x in enumerate(DIM_LABELS)]
MEDIA_TYPE = 'feature;face;libreface'
SUBDIR = 'libreface'


class LibreFace(Processor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.options = _default_options | self.options
        self.batch_size = int(self.options['batch_size'])
        for f in self.trainer.meta_uri:
            path = os.getenv("CACHE_DIR") + '/' + SUBDIR + '/' + f.uri_id + '.onnx'
            if not os.path.exists(path) or not validate_file(path, f.uri_hash):
                get_file(fname=f.uri_id + '.onnx', origin=f.uri_url, md5_hash=f.uri_hash,
                         cache_dir=os.getenv("CACHE_DIR"), tmp_dir=os.getenv("TMP_DIR"), cache_subdir=SUBDIR)
        self.au_enc = None#ort.InferenceSession(os.getenv("CACHE_DIR") + '/' + SUBDIR + '/au_enc.onnx',
                          #                 providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.au_det = None#ort.InferenceSession(os.getenv("CACHE_DIR") + '/' + SUBDIR + '/au_det.onnx',
                          #                 providers=['CUDAExecutionProvider','CPUExecutionProvider'])
        self.au_reg = None#ort.InferenceSession(os.getenv("CACHE_DIR") + '/' + SUBDIR + '/au_reg.onnx',
                          #                 providers=['CUDAExecutionProvider','CPUExecutionProvider'])
        self.fer = None#ort.InferenceSession(os.getenv("CACHE_DIR") + '/' + SUBDIR + '/fer.onnx',
                       #                 providers=['CUDAExecutionProvider','CPUExecutionProvider'])
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        log(f'Initialised LibreFace processor using device {self.device}')

    def process_sample(self, sample):
        images, face_landmarks = sample
        # facemesh has 468 y,x-landmarks + confidence
        face_landmarks, conf = face_landmarks[:, :936], face_landmarks[:, 936]
        face_landmarks = face_landmarks.reshape((-1, 468, 2))

        # debug
        # images = images[-1]
        # face_landmarks = face_landmarks[-1]

        Left_eye = [263, 249, 390, 373, 374, 380, 381, 382, 362, 466, 388, 387, 386, 385, 384, 398]
        Right_eye = [33, 7, 163, 144, 145, 153, 154, 155, 133, 246, 161, 160, 159, 158, 157, 173]
        Lips = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 185, 40, 39, 37, 0, 267, 269, 270, 409, 78, 95, 88,
                178, 87, 14, 317, 402, 318, 324, 308, 191, 80, 81, 82, 13, 312, 311, 310, 415]

        
        images_aligned = []
        #images_preprocessed = []
        #for image, face_landmark in zip(images, face_landmarks):
        def f(x):
            image, face_landmark = x
            lm_left_eye_x = []
            lm_left_eye_y = []
            lm_right_eye_x = []
            lm_right_eye_y = []
            lm_lips_x = []
            lm_lips_y = []
            for i in Left_eye:
                lm_left_eye_x.append(face_landmark[i][1])
                lm_left_eye_y.append(face_landmark[i][0])
            for i in Right_eye:
                lm_right_eye_x.append(face_landmark[i][1])
                lm_right_eye_y.append(face_landmark[i][0])
            for i in Lips:
                lm_lips_x.append(face_landmark[i][1])
                lm_lips_y.append(face_landmark[i][0])
            lm_x = lm_left_eye_x + lm_right_eye_x + lm_lips_x
            lm_y = lm_left_eye_y + lm_right_eye_y + lm_lips_y
            landmark = np.array([lm_x, lm_y]).T

            try:
                image = image_align(Image.fromarray(image), landmark)
            except ValueError:
                image = np.ones((256, 256, 3), dtype=np.uint8) * image.mean()
            # debug
            # aligned_image.save(f'{os.getenv("CACHE_DIR")}/{idx:05d}.jpg')
            '''
            image_preprocessed = image.astype(float)
            image_preprocessed = image_preprocessed / 255.0
            # supposedly RGB
            image_preprocessed = image_preprocessed - [[[0.485, 0.456, 0.406]]]
            image_preprocessed = image_preprocessed / [[[0.229, 0.224, 0.225]]]
            image_preprocessed = image_preprocessed.astype(np.float32)
            image_preprocessed = image_preprocessed.transpose((2, 0, 1))
            
            images_preprocessed.append(image_preprocessed)
            '''
            #images_aligned.append(image)
            return image

            #test = get_au_intensities_and_detect_aus(image, device=self.device, weights_download_dir=os.getenv("CACHE_DIR") + '/' + SUBDIR)
            #test2 = get_facial_expression(image, device=self.device, weights_download_dir=os.getenv("CACHE_DIR") + '/' + SUBDIR)

        with Pool() as p:
            images_aligned = list(p.imap(f, zip(images, face_landmarks)))

        test = get_au_intensities_and_detect_aus_video(images_aligned, device=self.device, weights_download_dir=os.getenv("CACHE_DIR") + '/' + SUBDIR)
        test2 = get_facial_expression_video(images_aligned, device=self.device, weights_download_dir=os.getenv("CACHE_DIR") + '/' + SUBDIR)
        '''
        images_preprocessed_encoded = [self.au_enc.run(['feature'], {'image': np.expand_dims(x, axis=0)}) for x in
                                       images_preprocessed]

        preds = [[self.au_det.run(['au_presence'], {'feature': np.expand_dims(np.squeeze(x), axis=0)})
                  for x in images_preprocessed_encoded],
                 [self.au_reg.run(['au_intensity'], {'feature': np.expand_dims(np.squeeze(x), axis=0)})
                  for x in images_preprocessed_encoded],
                 [self.fer.run(['FEs'], {'image': np.expand_dims(x, axis=0)})
                  for x in images_preprocessed]]

        return np.concatenate(preds, axis=-1).squeeze(), np.array(images_aligned, dtype=np.uint8)
        '''
        test2 = test2.replace(emo2id).infer_objects(copy=False)
        return pd.concat([test[0], test[1], test2], axis=1).values, np.array(images_aligned, dtype=np.uint8)

    def process_data(self, ds_iterator) -> dict[str, np.ndarray]:
        """Returning a dictionary that contains the original keys from the dataset iterator and a list of processed
        samples as value. Can be overwritten to customize the processing"""

        self.session_manager = self.get_session_manager(ds_iterator)
        data_object = self.session_manager.input_data[INP_VID]
        data = data_object.data
        fm_obj = self.session_manager.input_data[INP_FM]
        fm_data = fm_obj.data

        predictions, alignments = [], []
        tot_batch = int(len(data) / self.batch_size)
        for i in range(0, len(data), self.batch_size):
            idx_start = i
            idx_end = (
                idx_start + self.batch_size
                if idx_start + self.batch_size <= len(data)
                else len(data)
            )
            idxs = list(range(idx_start, idx_end))
            log(f"Batch {i / self.batch_size:.0f} of {tot_batch} : {idx_start} - {idx_end}")
            if not idxs:
                continue

            frame = data[idxs], fm_data[idxs]
            s = perf_counter()
            preds, aligns = self.process_sample(frame)
            predictions.extend(preds)
            alignments.extend(aligns)
            e = perf_counter()
            log(f'{e - s:.3f}s/batch, {(e - s) / self.batch_size:.3f}s/image,'
                  f' eta {(tot_batch - i / self.batch_size) * (e - s):.3f}s')
            # debug
            # break

        return {'predictions': np.array(predictions), 'alignments': np.array(alignments)}

    def to_output(self, data: np.ndarray) -> dict:
        output_templates = self.session_manager.output_data_templates
        frame = int(1000 / self.session_manager.input_data[INP_VID].meta_data.sample_rate)
        l = len(d := data['predictions'][:, -1])
        emo_data = np.array([[i*frame for i in range(l)], [i*frame for i in range(1,l+1)], d, [1]*l]).T  # from, to, id, conf
        output_templates[OUT_LF_EMO].data = [tuple(x) for x in emo_data]
        output_templates[OUT_LF] = SSIStream(
            data=data['predictions'][:, :-1],
            sample_rate=self.session_manager.input_data[INP_VID].meta_data.sample_rate,
            dim_labels=DIM_LABELS,
            media_type=MEDIA_TYPE,
            role=output_templates[OUT_LF].meta_data.role,
            dataset=output_templates[OUT_LF].meta_data.dataset,
            name=output_templates[OUT_LF].meta_data.name,
            session=output_templates[OUT_LF].meta_data.session,
        )
        output_templates[OUT_ALIGN] = Video(
            data=data['alignments'],
            sample_rate=self.session_manager.input_data[INP_VID].meta_data.sample_rate,
            dim_labels=DIM_LABELS,
            media_type=MEDIA_TYPE,
            role=output_templates[OUT_ALIGN].meta_data.role,
            dataset=output_templates[OUT_ALIGN].meta_data.dataset,
            name=output_templates[OUT_ALIGN].meta_data.name,
            session=output_templates[OUT_ALIGN].meta_data.session,
        )
        return output_templates