import numpy as np

def get_dim_labels():
    points = ["HEAD", "NECK", "TORSO", "WAIST", "LEFT_SHOULDER", "LEFT_ELBOW", "LEFT_WRIST", "LEFT_HAND",
              "RIGHT_SHOULDER", "RIGHT_ELBOW", "RIGHT_WRIST", "RIGHT_HAND", "LEFT_HIP", "LEFT_KNEE", "LEFT_ANKLE",
              "LEFT_FOOT", "RIGHT_HIP", "RIGHT_KNEE", "RIGHT_ANKLE", "RIGHT_FOOT", "FACE_NOSE", "FACE_LEFT_EAR",
              "FACE_RIGHT_EAR", "FACE_FOREHEAD", "FACE_CHIN"]
    joints = [
        "POS_X",
        "POS_Y",
        "POS_Z",
        "POS_CONF",
        "ROT_W",
        "ROT_X",
        "ROT_Y",
        "ROT_Z",
        "ROT_CONF",
        "ROT_REL_W",
        "ROT_REL_X",
        "ROT_REL_Y",
        "ROT_REL_Z",
        "ROT_REL_CONF"
    ]

    return [p + '_' + j for p in points for j in joints]

class SSISkeletonJointValue:
    def __init__(self):
        self.POS_X = 0
        self.POS_Y = 0
        self.POS_Z = 0
        self.POS_CONF = 0
        self.ROT_W = 0
        self.ROT_X = 0
        self.ROT_Y = 0
        self.ROT_Z = 0
        self.ROT_CONF = 0
        self.ROT_REL_W = 0
        self.ROT_REL_X = 0
        self.ROT_REL_Y = 0
        self.ROT_REL_Z = 0
        self.ROT_REL_CONF = 0


class SSISkeleton:

    def __init__(self):
        self.HEAD = SSISkeletonJointValue()
        self.NECK = SSISkeletonJointValue()
        self.TORSO = SSISkeletonJointValue()
        self.WAIST = SSISkeletonJointValue()
        self.LEFT_SHOULDER = SSISkeletonJointValue()
        self.LEFT_ELBOW = SSISkeletonJointValue()
        self.LEFT_WRIST = SSISkeletonJointValue()
        self.LEFT_HAND = SSISkeletonJointValue()
        self.RIGHT_SHOULDER = SSISkeletonJointValue()
        self.RIGHT_ELBOW = SSISkeletonJointValue()
        self.RIGHT_WRIST = SSISkeletonJointValue()
        self.RIGHT_HAND = SSISkeletonJointValue()
        self.LEFT_HIP = SSISkeletonJointValue()
        self.LEFT_KNEE = SSISkeletonJointValue()
        self.LEFT_ANKLE = SSISkeletonJointValue()
        self.LEFT_FOOT = SSISkeletonJointValue()
        self.RIGHT_HIP = SSISkeletonJointValue()
        self.RIGHT_KNEE = SSISkeletonJointValue()
        self.RIGHT_ANKLE = SSISkeletonJointValue()
        self.RIGHT_FOOT = SSISkeletonJointValue()
        self.FACE_NOSE = SSISkeletonJointValue()
        self.FACE_LEFT_EAR = SSISkeletonJointValue()
        self.FACE_RIGHT_EAR = SSISkeletonJointValue()
        self.FACE_FOREHEAD = SSISkeletonJointValue()
        self.FACE_CHIN = SSISkeletonJointValue()

    def to_numpy(self):
        skel = []

        for joint in [self.HEAD, self.NECK, self.TORSO, self.WAIST, self.LEFT_SHOULDER, self.LEFT_ELBOW,
                      self.LEFT_WRIST, self.LEFT_HAND, self.RIGHT_SHOULDER, self.RIGHT_ELBOW, self.RIGHT_WRIST,
                      self.RIGHT_HAND, self.LEFT_HIP, self.LEFT_KNEE, self.LEFT_ANKLE, self.LEFT_FOOT, self.RIGHT_HIP,
                      self.RIGHT_KNEE, self.RIGHT_ANKLE, self.RIGHT_FOOT, self.FACE_NOSE, self.FACE_LEFT_EAR,
                      self.FACE_RIGHT_EAR, self.FACE_FOREHEAD, self.FACE_CHIN]:
            skel.append(joint.POS_X)
            skel.append(joint.POS_Y)
            skel.append(joint.POS_Z)
            skel.append(joint.POS_CONF)
            skel.append(joint.ROT_W)
            skel.append(joint.ROT_X)
            skel.append(joint.ROT_Y)
            skel.append(joint.ROT_Z)
            skel.append(joint.ROT_CONF)
            skel.append(joint.ROT_REL_W)
            skel.append(joint.ROT_REL_X)
            skel.append(joint.ROT_REL_Y)
            skel.append(joint.ROT_REL_Z)
            skel.append(joint.ROT_REL_CONF)

        return np.array(skel, dtype=np.float32)


