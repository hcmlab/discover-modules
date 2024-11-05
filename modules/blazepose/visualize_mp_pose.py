import dotenv
import os
import cv2
import numpy as np
from pathlib import Path
from discover_utils.data.handler.file_handler import FileHandler
from matplotlib import pyplot as plt

dotenv.load_dotenv()

base_dir = Path(os.getenv("DISCOVER_DATA_DIR"))
out_dir = Path(os.getenv("DISCOVER_TEST_DIR"))
stream_in_mp_fp = Path(out_dir / "blaze_pose_out_mp.stream")
video_in_fp = Path(base_dir / "test_files" / "test_video.mp4")

fh = FileHandler()
video = fh.load(video_in_fp)
mp = fh.load(stream_in_mp_fp)


def paint_skel(image, skel):
    annotated_image = np.copy(image)
    # Radius of circle
    radius = 5
    # Red color in BGR
    color = (0, 0, 255)
    # Line thickness of -1 px
    thickness = -1
    for i in range(33) :
        idx_x = i * 5
        idx_y = i * 5 + 1

        #annotated_image = cv2.circle(annotated_image, (int((joint.POS_X+1)/2*image.shape[1]), int((joint.POS_Y+1)/2* image.shape[0])), radius, color, thickness)
        annotated_image = cv2.circle(annotated_image, (int(skel[idx_x]*image.shape[1]), int(skel[idx_y]* image.shape[0])), radius, color, thickness)
    return annotated_image


for i in range(0,video.data.shape[0],30):
    img = np.asarray(video.data[i])
    skels = mp.data[i]
    plt.figure(1)
    plt.clf()
    annotated_image = paint_skel(img, skels)
    plt.imshow(annotated_image)
    plt.pause(0.3)