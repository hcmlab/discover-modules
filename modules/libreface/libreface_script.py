"""
LibreFace Module
Author: Tobias Hallmen <tobias.hallmen@uni-a.de>
Date: 02.07.2026
cf. https://github.com/ihp-lab/LibreFace

Rewrite of the original module (pure PyTorch since pip libreface, no ONNX):

- Face alignment (FFHQ-style, from facemesh landmarks) now runs BATCHED ON THE
  GPU via torch.affine_grid/grid_sample instead of the per-frame PIL/scipy warp
  in a multiprocess pool. The FFHQ quad is a parallelogram, so the quad-to-square
  mapping is exactly affine; we sample at 512 and antialias-downscale to 256,
  mirroring FFHQ's transform_size=512 -> LANCZOS 256. The old CPU path made the
  whole module CPU-bound while the GPU idled.
- The three torch models (combined AU detection+intensity, FER, none reloaded
  per batch anymore: the upstream *_video() helpers construct the solver and
  re-download/load weights on EVERY call) are built once in __init__ and reused.
- FER now outputs a real softmax confidence instead of a constant 1.0, plus the
  penultimate features as an embedding stream.
- Frames whose facemesh confidence is below min_conf get the rest class /
  zeroed outputs instead of a mean-gray warp.

Gaze / head pose are NOT computed: upstream derives them from iris-refined
MediaPipe landmarks (incl. z) on the aligned crop, which the facemesh stream
does not carry; running MediaPipe per frame would reintroduce the CPU
bottleneck this rewrite removes.
"""

import os
import sys

# Add local dir to path for relative imports
if (p := os.path.dirname(__file__)) not in sys.path:
    sys.path.insert(0, p)

import numpy as np
import torch
import torch.nn.functional as F

from discover_utils.data.annotation import DiscreteAnnotationScheme
from discover_utils.data.stream import SSIStream, Video
from discover_utils.data.static import Image
from discover_utils.interfaces.server_module import Processor
from discover_utils.utils.log_utils import log
from discover_utils.utils.type_definitions import SSINPDataType

INP_VID = "video"
INP_FM = "facemesh"
OUT_LF = "libreface_stream"
OUT_LF_EMO = "libreface_emotion"
OUT_ALIGN = "aligned"
OUT_EMBEDDING = "embedding"

_default_options = {
    "batch_size": 256,
    # Minimum facemesh landmark confidence; below -> rest class / zero outputs.
    "min_conf": 0.5,
}

# Upstream AU sets (solver_inference_combine: au_detection_aus / au_recognition_aus)
detection = [1, 2, 4, 6, 7, 10, 12, 14, 15, 17, 23, 24]
regression = [1, 2, 4, 5, 6, 9, 12, 15, 17, 20, 25, 26]
# LibreFace FER order == AffectNet-8 order (libreface/__init__.py)
emotions = ["Neutral", "Happiness", "Sadness", "Surprise", "Fear", "Disgust", "Anger", "Contempt"]
DIM_LABELS = [f"AU{x}_c" for x in detection] + [f"AU{x}_r" for x in regression]
DIM_LABELS = [{"id": i, "name": x} for i, x in enumerate(DIM_LABELS)]
MEDIA_TYPE = "feature;face;libreface"
SUBDIR = "downloads/libreface"

ALIGN_SIZE = 256      # FFHQ output_size (what the models were trained on)
SAMPLE_SIZE = 512     # FFHQ transform_size (supersample, then antialias down)
CROP_SIZE = 224       # AU branch: CenterCrop(224) of the 256 aligned image

# MediaPipe landmark subsets used by upstream LibreFace alignment. NB the
# first list (mediapipe "left eye" = subject's left = image RIGHT side) plays
# the role of eye_right in the FFHQ math below, matching the original module's
# ordering exactly.
_EYE_IMG_RIGHT = [263, 249, 390, 373, 374, 380, 381, 382, 362,
                  466, 388, 387, 386, 385, 384, 398]
_EYE_IMG_LEFT = [33, 7, 163, 144, 145, 153, 154, 155, 133,
                 246, 161, 160, 159, 158, 157, 173]
_LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 185, 40, 39, 37,
         0, 267, 269, 270, 409, 78, 95, 88, 178, 87, 14, 317, 402, 318, 324,
         308, 191, 80, 81, 82, 13, 312, 311, 310, 415]

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def _ffhq_quads(landmarks: np.ndarray, img_wh: tuple) -> np.ndarray:
    """Batched FFHQ crop quads from facemesh landmarks.

    Args:
        landmarks: [B, 468, 2] normalized (y, x) as written by the facemesh module.
        img_wh: (width, height) in pixels.

    Returns:
        [B, 4, 2] pixel-space quads (UL, LL, LR, UR corners of the source
        parallelogram; the same order PIL's QUAD transform expects). Rows with
        degenerate geometry come out non-finite and must be masked by the caller.
    """
    W, H = img_wh
    xy = np.stack([landmarks[..., 1] * W, landmarks[..., 0] * H], axis=-1)  # [B,468,2] (x,y)

    eye_right = xy[:, _EYE_IMG_RIGHT].mean(axis=1)
    eye_left = xy[:, _EYE_IMG_LEFT].mean(axis=1)
    eye_avg = (eye_left + eye_right) * 0.5
    eye_to_eye = eye_right - eye_left

    lips = xy[:, _LIPS]                                   # [B,40,2]
    left_i = np.argmin(lips[..., 0], axis=1)
    right_i = np.argmax(lips[..., 0], axis=1)
    rows = np.arange(len(lips))
    mouth_avg = (lips[rows, left_i] + lips[rows, right_i]) / 2.0
    eye_to_mouth = mouth_avg - eye_avg

    # FFHQ: x = eye_to_eye - flipud(eye_to_mouth) * [-1, 1]
    x = eye_to_eye - eye_to_mouth[:, ::-1] * np.array([-1.0, 1.0])
    with np.errstate(divide="ignore", invalid="ignore"):
        x = x / np.hypot(x[:, 0], x[:, 1])[:, None]
    scale = np.maximum(np.hypot(eye_to_eye[:, 0], eye_to_eye[:, 1]) * 2.0,
                       np.hypot(eye_to_mouth[:, 0], eye_to_mouth[:, 1]) * 1.8)
    x = x * scale[:, None]
    y = x[:, ::-1] * np.array([-1.0, 1.0])                # flipud(x) * [-1, 1]
    c = eye_avg + eye_to_mouth * 0.1

    return np.stack([c - x - y, c - x + y, c + x + y, c + x - y], axis=1)


class LibreFace(Processor):
    chainable = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.options = _default_options | self.options
        self.batch_size = int(self.options["batch_size"])
        self.min_conf = float(self.options["min_conf"])
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        log(f"Initialised LibreFace processor using device {self.device}")

        weights_dir = os.path.join(os.getenv("CACHE_DIR") or ".", SUBDIR)

        # Build the solvers ONCE (their __init__ downloads + loads the weights).
        # Configs replicate the upstream *_video() helpers verbatim.
        from libreface.AU_Recognition.inference import ConfigObject
        from libreface.AU_Recognition.solver_inference_combine import (
            solver_inference_image_task_combine,
        )
        from libreface.Facial_Expression_Recognition.solver_inference_image import (
            solver_inference_image,
        )

        au_opts = ConfigObject({
            "seed": 0,
            "ckpt_path": f"{weights_dir}/AU_Recognition/weights/combined_repvgg.pt",
            "weights_download_id": "1CbnBr8OBt8Wb73sL1ENcrtrWAFWSSRv0",
            "image_inference": False,
            "au_recognition_data_root": "", "au_recognition_data": "DISFA",
            "au_detection_data_root": "", "au_detection_data": "BP4D",
            "fer_train_csv": "training_filtered.csv",
            "fer_test_csv": "validation_filtered.csv",
            "fer_data_root": "", "fer_data": "AffectNet",
            "fold": "all", "num_workers": 0,
            "image_size": 256, "crop_size": CROP_SIZE,
            "au_recognition_num_labels": 12, "au_detection_num_labels": 12,
            "fer_num_labels": 8, "sigma": 10.0, "jitter": False,
            "copy_classifier": False, "model_name": "resnet", "dropout": 0.1,
            "ffhq_pretrain": "", "hidden_dim": 128, "fm_distillation": False,
            "num_epochs": 30, "interval": 500, "threshold": 0,
            "batch_size": self.batch_size, "learning_rate": 3e-5,
            "weight_decay": 1e-4, "clip": 1.0, "when": 10, "patience": 5,
            "device": self.device,
        })
        self.au_solver = solver_inference_image_task_combine(au_opts).to(self.device)
        self.au_solver.eval()

        fer_opts = ConfigObject({
            "seed": 0,
            "train_csv": "training_filtered.csv", "test_csv": "validation_filtered.csv",
            "data_root": "",
            "ckpt_path": f"{weights_dir}/Facial_Expression_Recognition/weights/repvgg.pt",
            "weights_download_id": "1yPBUjPhkwcIkRLt47-JJRLRD7rCwPIVU",
            "data": "AffectNet", "num_workers": 0,
            "image_size": 224, "num_labels": 8, "dropout": 0.1,
            "hidden_dim": 128, "sigma": 10.0,
            "student_model_name": "repvgg", "alpha": 1.0, "T": 1.0,
            "fm_distillation": True, "grad": True, "interval": 500,
            "threshold": 0.0, "loss": "unweighted", "num_epochs": 50,
            "batch_size": self.batch_size, "learning_rate": "3e-5",
            "weight_decay": "1e-4", "clip": 1.0, "when": 10, "patience": 10,
            "device": self.device,
        })
        self.fer_solver = solver_inference_image(fer_opts).to(self.device)
        self.fer_solver.student_model.eval()

        mean = torch.tensor(_IMAGENET_MEAN, device=self.device).view(1, 3, 1, 1)
        std = torch.tensor(_IMAGENET_STD, device=self.device).view(1, 3, 1, 1)
        self._norm = lambda t: (t - mean) / std

    def _align_gpu(self, frames: np.ndarray, quads: np.ndarray) -> torch.Tensor:
        """Batched FFHQ warp: frames [B,H,W,3] uint8 + quads [B,4,2] -> [B,3,256,256] float 0..1.

        The FFHQ quad (c±x±y) is a parallelogram, so PIL's QUAD transform is an
        affine map; grid_sample reproduces it. Supersample at 512, antialias down
        to 256 (FFHQ: transform at 512, LANCZOS resize to 256).
        """
        B, H, W = frames.shape[:3]
        t = torch.from_numpy(np.ascontiguousarray(frames[..., :3])).to(self.device)
        t = t.permute(0, 3, 1, 2).float() / 255.0

        q = torch.from_numpy(quads.astype(np.float32)).to(self.device)  # [B,4,2]
        u = (torch.arange(SAMPLE_SIZE, device=self.device, dtype=torch.float32) + 0.5) / SAMPLE_SIZE
        # src(row v, col u) = q0 + u*(q3-q0) + v*(q1-q0)
        q0 = q[:, 0][:, None, None]
        du = (q[:, 3] - q[:, 0])[:, None, None]
        dv = (q[:, 1] - q[:, 0])[:, None, None]
        src = q0 + u[None, None, :, None] * du + u[None, :, None, None] * dv  # [B,S,S,2] (x,y)
        grid = torch.empty_like(src)
        grid[..., 0] = 2.0 * src[..., 0] / W - 1.0
        grid[..., 1] = 2.0 * src[..., 1] / H - 1.0

        warped = F.grid_sample(t, grid, mode="bilinear", padding_mode="border",
                               align_corners=False)
        return F.interpolate(warped, size=(ALIGN_SIZE, ALIGN_SIZE), mode="bicubic",
                             antialias=True).clamp(0, 1)

    def process_sample(self, images: np.ndarray, face_landmarks: np.ndarray):
        """One batch: frames [B,H,W,3] uint8 + facemesh rows [B,937] -> per-frame outputs."""
        B = len(images)
        H, W = images.shape[1:3]
        conf = face_landmarks[:, 936]
        lms = face_landmarks[:, :936].reshape((-1, 468, 2))

        quads = _ffhq_quads(lms, (W, H))
        valid = (conf >= self.min_conf) & np.isfinite(quads).all(axis=(1, 2))

        aus = np.zeros((B, 24), dtype=np.float32)
        probs = np.zeros((B, 9), dtype=np.float32)
        probs[:, 8] = 1.0                        # default: rest class (no face)
        aligned = np.zeros((B, ALIGN_SIZE, ALIGN_SIZE, 3), dtype=np.uint8)
        embedding = None

        idx = np.where(valid)[0]
        if len(idx):
            with torch.no_grad():
                al = self._align_gpu(images[idx], quads[idx])   # [n,3,256,256] 0..1

                # AU branch: CenterCrop(224) of the 256 aligned image + ImageNet norm
                off = (ALIGN_SIZE - CROP_SIZE) // 2
                au_in = self._norm(al[:, :, off:off + CROP_SIZE, off:off + CROP_SIZE])
                au_det = (self.au_solver.model(au_in, task_name="au_detection") >= 0.5).int()
                au_int = self.au_solver.model(au_in, task_name="au_recognition") * 5.0

                # FER branch: straight resize 256 -> 224 + ImageNet norm
                fer_in = self._norm(F.interpolate(al, size=(224, 224), mode="bicubic",
                                                  antialias=True).clamp(0, 1))
                logits, feat = self.fer_solver.student_model(fer_in)
                fer_probs = torch.softmax(logits, dim=-1)

            aus[idx, :12] = au_det.cpu().numpy()
            aus[idx, 12:] = au_int.cpu().numpy()
            probs[idx, 8] = 0.0
            probs[idx, :8] = fer_probs.cpu().numpy()
            aligned[idx] = (al.permute(0, 2, 3, 1) * 255).round().byte().cpu().numpy()
            embedding = np.zeros((B, feat.shape[-1]), dtype=np.float32)
            embedding[idx] = feat.cpu().numpy()

        return aus, probs, aligned, embedding

    def process_data(self, ds_manager) -> dict:
        self.session_manager = self.get_session_manager(ds_manager)
        data_object = self.session_manager.input_data[INP_VID]
        data = data_object.data
        fm_data = self.session_manager.input_data[INP_FM].data

        # Append batch dimension
        if isinstance(data_object, Image):
            data = np.expand_dims(data, 0)

        from time import perf_counter
        aus, probs, aligned, embeddings = [], [], [], []
        emb_dim = None
        iter_len = min(len(data), len(fm_data))
        tot_batch = -(-iter_len // self.batch_size)
        for i in range(0, iter_len, self.batch_size):
            idxs = list(range(i, min(i + self.batch_size, iter_len)))
            if not idxs:
                continue
            s = perf_counter()
            a, p, al, emb = self.process_sample(np.asarray(data[idxs]), fm_data[idxs])
            aus.append(a)
            probs.append(p)
            aligned.append(al)
            if emb is not None:
                emb_dim = emb.shape[-1]
            embeddings.append(emb)
            e = perf_counter()
            log(f"Batch {i // self.batch_size + 1} of {tot_batch}: {e - s:.3f}s/batch,"
                f" {(e - s) / len(idxs) * 1000:.1f}ms/frame")

        # batches without any valid face returned emb=None; backfill zeros
        if emb_dim is None:
            emb_dim = 1
        embeddings = [
            e if e is not None else np.zeros((len(a), emb_dim), np.float32)
            for e, a in zip(embeddings, aus)
        ]

        torch.cuda.empty_cache()
        return {
            "aus": np.concatenate(aus) if aus else np.zeros((0, 24), np.float32),
            "probs": np.concatenate(probs) if probs else np.zeros((0, 9), np.float32),
            "aligned": np.concatenate(aligned) if aligned else
            np.zeros((0, ALIGN_SIZE, ALIGN_SIZE, 3), np.uint8),
            "embedding": np.concatenate(embeddings) if embeddings else
            np.zeros((0, emb_dim), np.float32),
        }

    def to_output(self, data: dict) -> dict:
        output_templates = self.session_manager.output_data_templates
        sr_hz = self.session_manager.input_data[INP_VID].meta_data.sample_rate
        frame_ms = 1000 / sr_hz

        probs = data["probs"]
        n = len(probs)
        emo_data = np.array([
            [i * frame_ms for i in range(n)],
            [i * frame_ms for i in range(1, n + 1)],
            np.argmax(probs, axis=-1),
            np.amax(probs, axis=-1),
        ]).T
        anno = output_templates[OUT_LF_EMO]
        if isinstance(anno.annotation_scheme, DiscreteAnnotationScheme):
            emo_data = np.asarray(
                [tuple(x) for x in emo_data], dtype=anno.annotation_scheme.label_dtype
            )
        anno.data = emo_data

        output_templates[OUT_LF] = SSIStream(
            data=data["aus"].astype(np.dtype("<f4")),
            sample_rate=sr_hz,
            dim_labels=DIM_LABELS,
            media_type=MEDIA_TYPE,
            role=output_templates[OUT_LF].meta_data.role,
            dataset=output_templates[OUT_LF].meta_data.dataset,
            name=output_templates[OUT_LF].meta_data.name,
            session=output_templates[OUT_LF].meta_data.session,
        )
        output_templates[OUT_ALIGN] = Video(
            data=data["aligned"],
            sample_rate=sr_hz,
            media_type=MEDIA_TYPE,
            role=output_templates[OUT_ALIGN].meta_data.role,
            dataset=output_templates[OUT_ALIGN].meta_data.dataset,
            name=output_templates[OUT_ALIGN].meta_data.name,
            session=output_templates[OUT_ALIGN].meta_data.session,
        )
        output_templates[OUT_EMBEDDING] = SSIStream(
            data=np.array(data["embedding"], SSINPDataType.FLOAT.value),
            sample_rate=sr_hz,
            media_type=MEDIA_TYPE,
        )
        return output_templates
