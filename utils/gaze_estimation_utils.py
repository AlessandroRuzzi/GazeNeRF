import cv2
import numpy as np
import torch


def estimateHeadPose(landmarks, face_model, camera, distortion, iterate=True):
    ret, rvec, tvec = cv2.solvePnP(
        face_model, landmarks, camera, distortion, flags=cv2.SOLVEPNP_EPNP
    )

    ## further optimize
    if iterate:
        ret, rvec, tvec = cv2.solvePnP(
            face_model, landmarks, camera, distortion, rvec, tvec, True
        )

    return rvec, tvec


def convert_to_head_coordinate_system(face_model, gaze_location, hr, ht):
    rot = hr
    trans = ht
    head_rotation_matrix = cv2.Rodrigues(rot)[0]
    Fc = (
        np.dot(head_rotation_matrix, face_model.T) + trans
    )  # rotate and translate the face model
    two_eye_center = np.mean(Fc[:, 0:4], axis=1).reshape((3, 1))
    nose_center = np.mean(Fc[:, 4:6], axis=1).reshape((3, 1))
    # get the face center
    face_center = np.mean(
        np.concatenate((two_eye_center, nose_center), axis=1), axis=1
    ).reshape((3, 1))
    gaze_direction = gaze_location - face_center

    gaze_direction_head = np.dot(
        np.linalg.inv(head_rotation_matrix), gaze_direction
    ).T  # gaze direction in the head cooridnate system
    gaze_pitch_yaw = vector_to_pitchyaw(
        gaze_direction_head
    )  # gaze direction in pitch and yaw

    return gaze_pitch_yaw


def vector_to_pitchyaw(vector):
    vector = vector / np.linalg.norm(vector)
    out = np.zeros((1, 2))
    out[0, 0] = np.arcsin(vector[0, 1])  # theta
    out[0, 1] = np.arctan2(
        -1 * vector[0, 0], -1 * vector[0, 2]
    )  # phi   Here, I use minus to reverse x and z axis
    return out


def normalize(
    image, camera_matrix, camera_distortion, face_model_load, landmarks, img_dim
):

    landmarks = np.asarray(landmarks)

    # load face model
    landmark_use = [20, 23, 26, 29, 15, 19]  # we use eye corners and nose conners
    face_model = face_model_load[landmark_use, :]
    facePts = face_model.reshape(6, 1, 3)
    landmarks_sub = landmarks[[36, 39, 42, 45, 31, 35], :]
    landmarks_sub = landmarks_sub.astype(
        float
    )  # input to solvePnP function must be float type
    landmarks_sub = landmarks_sub.reshape(
        6, 1, 2
    )  # input to solvePnP requires such shape
    hr, ht = estimateHeadPose(landmarks_sub, facePts, camera_matrix, camera_distortion)

    # data normalization method
    img_normalized, landmarks_normalized = normalizeData_face(
        image, face_model, landmarks_sub, hr, ht, camera_matrix, img_dim
    )

    return img_normalized


def normalizeData_face(img, face_model, landmarks, hr, ht, cam, img_dim):
    ## normalized camera parameters
    focal_norm = 960  # focal length of normalized camera
    distance_norm = 600  # normalized distance between eye and camera
    roiSize = (img_dim, img_dim)  # size of cropped eye image

    ## compute estimated 3D positions of the landmarks
    ht = ht.reshape((3, 1))
    hR = cv2.Rodrigues(hr)[0]  # rotation matrix
    Fc = np.dot(hR, face_model.T) + ht  # rotate and translate the face model
    two_eye_center = np.mean(Fc[:, 0:4], axis=1).reshape((3, 1))
    nose_center = np.mean(Fc[:, 4:6], axis=1).reshape((3, 1))
    # get the face center
    face_center = np.mean(
        np.concatenate((two_eye_center, nose_center), axis=1), axis=1
    ).reshape((3, 1))

    ## ---------- normalize image ----------
    distance = np.linalg.norm(
        face_center
    )  # actual distance between eye and original camera

    z_scale = distance_norm / distance
    cam_norm = np.array(
        [  # camera intrinsic parameters of the virtual camera
            [focal_norm, 0, roiSize[0] / 2],
            [0, focal_norm, roiSize[1] / 2],
            [0, 0, 1.0],
        ]
    )
    S = np.array(
        [  # scaling matrix
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, z_scale],
        ]
    )

    hRx = hR[:, 0]
    forward = (face_center / distance).reshape(3)
    down = np.cross(forward, hRx)
    down /= np.linalg.norm(down)
    right = np.cross(down, forward)
    right /= np.linalg.norm(right)
    R = np.c_[right, down, forward].T  # rotation matrix R

    W = np.dot(
        np.dot(cam_norm, S), np.dot(R, np.linalg.inv(cam))
    )  # transformation matrix

    img_warped = cv2.warpPerspective(img, W, roiSize)  # warp the input image

    # head pose after normalization
    hR_norm = np.dot(R, hR)  # head pose rotation matrix in normalized space
    hr_norm = cv2.Rodrigues(hR_norm)[0]  # convert rotation matrix to rotation vectors

    # normalize the facial landmarks
    num_point = landmarks.shape[0]
    landmarks_warped = cv2.perspectiveTransform(landmarks, W)
    landmarks_warped = landmarks_warped.reshape(num_point, 2)

    return img_warped, landmarks_warped
