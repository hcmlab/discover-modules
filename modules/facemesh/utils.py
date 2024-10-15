import numpy as np
import cv2


def select_bounding_box(detection: np.ndarray):
    return detection[:, :4]


def scale_bounding_box(detection: np.ndarray, img_shape: tuple):
    x_min = detection[:, 0] * img_shape[0]
    y_min = detection[:, 1] * img_shape[1]
    x_max = detection[:, 2] * img_shape[0]
    y_max = detection[:, 3] * img_shape[1]
    return np.swapaxes(np.vstack([x_min, y_min, x_max, y_max]), 0, 1)


def get_low_confidence_detections(detection: np.ndarray, thresh=0.8):
    return np.where(detection[:, -1] < thresh)[0]


def crop_and_resize(img: np.ndarray, face: np.ndarray, target_size: int):
    face = face.astype(np.int32)

    # Remove crop areas beyond the image
    y_min_pad = 0
    x_min_pad = 0
    y_max_pad = 0
    x_max_pad = 0

    # Y-Axis Min
    if face[0] < 0:
        y_min_pad = face[0] * -1
        face[0] = 0

     # X-Axis Min
    if face[1] < 0:
        x_min_pad = face[1] * -1
        face[1] = 0

    # Y-Axis Max
    if face[2] > img.shape[0]:
        y_max_pad = face[2] - img.shape[0]
        face[2] = img.shape[0]

    # X-Axis Max
    if face[3] > img.shape[1]:
        y_max_pad = face[3] - img.shape[1]
        face[3] = img.shape[1]

    # Crop
    crop = img[face[0] : face[2], face[1] : face[3], :]

    # Pad with 0 to ensure aspect ratio
    crop = np.pad(crop, ((y_min_pad, y_max_pad), (x_min_pad, x_max_pad), (0,0)))

    # Resize
    try:
        resized = cv2.resize(crop, (target_size, target_size))
    except:
        breakpoint()
    return resized

def rel_to_abs(coords: np.ndarray, img_shape: tuple):
    # Expecting y,x format
    scaler = np.tile(list(img_shape), coords.shape[1] // 2)
    return coords * scaler

def abs_to_rel(coords: np.ndarray, img_shape: tuple):
    # Expecting y,x format
    scaler = np.tile(list(img_shape), coords.shape[1] // 2)
    return coords / scaler
