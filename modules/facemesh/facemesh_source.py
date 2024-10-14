import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
import cv2
import torch.nn.functional as F


class FacialLMBasicBlock(nn.Module):
    """Building block for mediapipe facial landmark model

    DepthwiseConv + Conv + PRelu
    downsampling + channel padding for few blocks(when stride=2)
    channel padding values - 16, 32, 64

    Args:
        nn ([type]): [description]
    """

    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1
    ):
        super(FacialLMBasicBlock, self).__init__()

        self.stride = stride
        self.channel_pad = out_channels - in_channels

        # TFLite uses slightly different padding than PyTorch
        # on the depthwise conv layer when the stride is 2.
        if stride == 2:
            self.max_pool = nn.MaxPool2d(kernel_size=stride, stride=stride)
            padding = 0
        else:
            padding = (kernel_size - 1) // 2

        self.depthwiseconv_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=in_channels,
                bias=True,
            ),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
            ),
        )

        self.prelu = nn.PReLU(out_channels)

    def forward(self, x):
        """[summary]

        Args:
            x ([torch.Tensor]): [input tensor]

        Returns:
            [torch.Tensor]: [featues]
        """

        if self.stride == 2:
            h = F.pad(x, (0, 2, 0, 2), "constant", 0)
            x = self.max_pool(x)
        else:
            h = x

        if self.channel_pad > 0:
            x = F.pad(x, (0, 0, 0, 0, 0, self.channel_pad), "constant", 0)

        return self.prelu(self.depthwiseconv_conv(h) + x)


def pad_image(im, desired_size=192):
    """[summary]

    Args:
        im ([cv2 image]): [input image]
        desired_size (int, optional): [description]. Defaults to 64.

    Returns:
        [cv2 image]: [resized image]
    """
    old_size = im.shape[:2]  # old_size is in (height, width) format

    ratio = float(desired_size) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])

    # new_size should be in (width, height) format

    im = cv2.resize(im, (new_size[1], new_size[0]))

    delta_w = desired_size - new_size[1]
    delta_h = desired_size - new_size[0]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    color = [0, 0, 0]
    new_im = cv2.copyMakeBorder(
        im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )

    return new_im


class GetKeysDict:
    """
    maps pytorch keys to tflite keys
    """

    def __init__(self):
        self.facial_landmark_dict = {
            "confidence.2.depthwiseconv_conv.0.weight": "depthwise_conv2d_16/Kernel",
            "confidence.2.depthwiseconv_conv.0.bias": "depthwise_conv2d_16/Bias",
            "confidence.2.depthwiseconv_conv.1.weight": "conv2d_17/Kernel",
            "confidence.2.depthwiseconv_conv.1.bias": "conv2d_17/Bias",
            "confidence.2.prelu.weight": "p_re_lu_17/Alpha",
            "confidence.3.weight": "conv2d_18/Kernel",
            "confidence.3.bias": "conv2d_18/Bias",
            "confidence.4.weight": "p_re_lu_18/Alpha",
            "confidence.5.depthwiseconv_conv.0.weight": "depthwise_conv2d_17/Kernel",
            "confidence.5.depthwiseconv_conv.0.bias": "depthwise_conv2d_17/Bias",
            "confidence.5.depthwiseconv_conv.1.weight": "conv2d_19/Kernel",
            "confidence.5.depthwiseconv_conv.1.bias": "conv2d_19/Bias",
            "confidence.5.prelu.weight": "p_re_lu_19/Alpha",
            "confidence.6.weight": "conv2d_20/Kernel",
            "confidence.6.bias": "conv2d_20/Bias",
            "facial_landmarks.0.depthwiseconv_conv.0.weight": "depthwise_conv2d_22/Kernel",
            "facial_landmarks.0.depthwiseconv_conv.0.bias": "depthwise_conv2d_22/Bias",
            "facial_landmarks.0.depthwiseconv_conv.1.weight": "conv2d_27/Kernel",
            "facial_landmarks.0.depthwiseconv_conv.1.bias": "conv2d_27/Bias",
            "facial_landmarks.0.prelu.weight": "p_re_lu_25/Alpha",
            "facial_landmarks.1.weight": "conv2d_28/Kernel",
            "facial_landmarks.1.bias": "conv2d_28/Bias",
            "facial_landmarks.2.weight": "p_re_lu_26/Alpha",
            "facial_landmarks.3.depthwiseconv_conv.0.weight": "depthwise_conv2d_23/Kernel",
            "facial_landmarks.3.depthwiseconv_conv.0.bias": "depthwise_conv2d_23/Bias",
            "facial_landmarks.3.depthwiseconv_conv.1.weight": "conv2d_29/Kernel",
            "facial_landmarks.3.depthwiseconv_conv.1.bias": "conv2d_29/Bias",
            "facial_landmarks.3.prelu.weight": "p_re_lu_27/Alpha",
            "facial_landmarks.4.weight": "conv2d_30/Kernel",
            "facial_landmarks.4.bias": "conv2d_30/Bias",
        }


class _FaceMesh(nn.Module):
    """[MediaPipe facial_landmark model in Pytorch]

    Args:
        nn ([type]): [description]

    Returns:
        [type]: [description]
    """

    # 1x1x1x1404, 1x1x1x1
    # 1x1404x1x1

    def __init__(self):
        """[summary]"""
        super(_FaceMesh, self).__init__()

        self.backbone = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=16,
                kernel_size=3,
                stride=2,
                padding=0,
                bias=True,
            ),
            nn.PReLU(16),
            FacialLMBasicBlock(16, 16),
            FacialLMBasicBlock(16, 16),
            FacialLMBasicBlock(16, 32, stride=2),  # pad
            FacialLMBasicBlock(32, 32),
            FacialLMBasicBlock(32, 32),
            FacialLMBasicBlock(32, 64, stride=2),
            FacialLMBasicBlock(64, 64),
            FacialLMBasicBlock(64, 64),
            FacialLMBasicBlock(64, 128, stride=2),
            FacialLMBasicBlock(128, 128),
            FacialLMBasicBlock(128, 128),
            FacialLMBasicBlock(128, 128, stride=2),
            FacialLMBasicBlock(128, 128),
            FacialLMBasicBlock(128, 128),
        )

        # facial_landmark head
        # @TODO change name from self.confidence to self.facial_landmarks
        self.confidence = nn.Sequential(
            FacialLMBasicBlock(128, 128, stride=2),
            FacialLMBasicBlock(128, 128),
            FacialLMBasicBlock(128, 128),
            # ----
            nn.Conv2d(
                in_channels=128,
                out_channels=32,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
            ),
            nn.PReLU(32),
            FacialLMBasicBlock(32, 32),
            nn.Conv2d(
                in_channels=32,
                out_channels=1404,
                kernel_size=3,
                stride=3,
                padding=0,
                bias=True,
            ),
        )

        # confidence score head
        # @TODO change name from self.facial_landmarks to  self.confidence
        self.facial_landmarks = nn.Sequential(
            FacialLMBasicBlock(128, 128, stride=2),
            nn.Conv2d(
                in_channels=128,
                out_channels=32,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
            ),
            nn.PReLU(32),
            FacialLMBasicBlock(32, 32),
            nn.Conv2d(
                in_channels=32,
                out_channels=1,
                kernel_size=3,
                stride=3,
                padding=0,
                bias=True,
            ),
        )

    @torch.no_grad()
    def forward(self, x):
        """forward prop

        Args:
            x ([torch.Tensor]): [input Tensor]

        Returns:
            [list]: [facial_landmarks, confidence]
            facial_landmarks: 1 x 1404 x 1 x 1
            (368 x 3)
            (x, y, z)
            (x, y) corresponds to image pixel locations
            confidence: 1 x 1 x 1 x 1
            368 face landmarks
        """

        # @TODO remove
        with torch.no_grad():
            x = nn.ReflectionPad2d((1, 0, 1, 0))(x)
            features = self.backbone(x)

            # @TODO change the names
            confidence = self.facial_landmarks(features)

            facial_landmarks = self.confidence(features)

            return [facial_landmarks, confidence]
        # return [facial_landmarks.view(x.shape[0], -1), confidence.reshape(x.shape[0], -1)]

    def predict(self, img):
        """single image inference

        Args:
            img ([type]): [description]

        Returns:
            [type]: [description]
        """
        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img).permute((2, 0, 1))

        return self.batch_predict(img.unsqueeze(0))

    def batch_predict(self, x):
        """batch inference
        currently only single image inference is supported

        Args:
            x ([type]): [description]

        Returns:
            [type]: [description]
        """
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).permute((0, 3, 1, 2))

        facial_landmarks, confidence = self.forward(x)
        return facial_landmarks, confidence
        # return facial_landmarks.view(x.shape[0], -1), confidence.view(x.shape[0], -1)

    def test(self):
        """Sample Inference"""
        inp = torch.randn(1, 3, 192, 192)
        output = self(inp)
        print(output[0].shape, output[1].shape)


# m = FacialLM_Model()
# m.test()
"""
m = FacialLM_Model()
inp = torch.randn(1, 3, 192, 192)
output = m(inp)
print(output[0].shape, output[1].shape)
"""
