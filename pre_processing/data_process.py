import argparse
from calendar import day_abbr
import os

import cv2
import numpy as np
import pandas as pd
import scipy.io
import csv
from pre_processing.gen_all_masks import GenMask
from pre_processing.gen_landmark import Gen2DLandmarks
from utils.gaze_estimation_utils import estimateHeadPose


def main_process(
    dataset_name, img_dir, gpu_id, annotation_path, cam_calibration_path, face_model_load, log=True, save_path = None, previous_frame=None
):
    """
    Produce head and eye mask, landmarks, latent codes and gaze direction for images in a given directory.

    :img_dir: Path to the images to process
    :gpu_id: Id of the GPU to use
    :log: Boolean value that indicates if images will be logged
    """

    if dataset_name == "columbia":
        if save_path is not None and not os.path.exists(save_path):
            os.makedirs(save_path)
        tt = Gen2DLandmarks(gpu_id=gpu_id, log=log)
        ldms = tt.main_process(img_dir, previous_frame, columbia_path = save_path)

    images, cam_indices = select_normalization(
        dataset_name,img_dir, annotation_path, cam_calibration_path, face_model_load, save_path
    )

    if save_path is not None:
        img_dir = save_path

    tt = Gen2DLandmarks(gpu_id=gpu_id, log=log)
    ldms = tt.main_process(img_dir, previous_frame)

    tt = GenMask(gpu_id=gpu_id, log=log)
    head_masks, left_eye_masks, right_eye_masks = tt.main_process(img_dir=img_dir)

    return images, cam_indices, head_masks, ldms, left_eye_masks, right_eye_masks


def select_normalization(dataset_name,img_dir, annotation_path, cam_calibration_path, face_model_load, save_path):
    if dataset_name == "eth_xgaze":
        return xgaze_data_normalization(img_dir, annotation_path, cam_calibration_path, face_model_load, save_path)
    elif dataset_name == "mpii_face":
        return mpii_face_gaze_data_normalization(img_dir, annotation_path, cam_calibration_path, face_model_load, save_path)
    elif dataset_name == "columbia":
        return columbia_data_normalization(img_dir, annotation_path, cam_calibration_path, face_model_load, save_path)
    elif dataset_name == "gaze_capture":
        return gaze_capture_data_normalization(img_dir, annotation_path, cam_calibration_path, face_model_load, save_path)
    else:
        print("Dataset not supported")

def normalizeData_face(img, face_model, hr, ht, cam, img_dim, focal_norm, fc = None):
    focal_norm = focal_norm
    distance_norm = 680
    roiSize = (img_dim, img_dim)
    ht = ht.reshape((3, 1))
    hR = cv2.Rodrigues(hr)[0]
    if fc is None:
        Fc = np.dot(hR, face_model.T) + ht
        two_eye_center = np.mean(Fc[:, 0:4], axis=1).reshape((3, 1))
        mouth_center = np.mean(Fc[:, 4:6], axis=1).reshape((3, 1))
        face_center = np.mean(
            np.concatenate((two_eye_center, mouth_center), axis=1), axis=1
        ).reshape((3, 1))
    else:
        face_center = fc.reshape((3, 1))

    distance = np.linalg.norm(face_center)
    z_scale = distance_norm / distance
    cam_norm = np.array(
        [
            [focal_norm, 0, roiSize[0] / 2],
            [0, focal_norm, roiSize[1] / 2],
            [0, 0, 1.0],
        ]
    )

    S = np.array(
        [
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
    R = np.c_[right, down, forward].T

    W = np.dot(np.dot(cam_norm, S), np.dot(R, np.linalg.inv(cam)))
    img_warped = cv2.warpPerspective(img, W, roiSize)

    return img_warped


def xgaze_data_normalization(
    frame_folder, annotation_folder, camera_calibration_folder, face_model_file, save_path = None
):
    # frame_folder: path to the certain frame; e.g. 'xx/xx/subject0000/frame0000'
    # annotation_folder: path to the annotation files, suppose you saved all annotation .csv files in the same folder
    # camera_calibration_folder: path to the folder where all 18 camera calibration .xml files are stored
    # face_model_file: path to the file 'face_model.txt' from the ETH-XGaze

    if save_path is not None and not os.path.exists(save_path):
        os.makedirs(save_path)

    remove_image_name_list = []

    camera_matrix = {}
    camera_distortion = {}
    cam_translation = {}
    cam_rotation = {}

    img_list = []
    cam_ind_list = []

    for cam_id in range(0, 18):
        file_name = os.path.join(
            camera_calibration_folder, "cam" + str(cam_id).zfill(2) + ".xml"
        )
        fs = cv2.FileStorage(file_name, cv2.FILE_STORAGE_READ)
        camera_matrix[str(cam_id).zfill(2)] = fs.getNode("Camera_Matrix").mat()
        camera_distortion[str(cam_id).zfill(2)] = fs.getNode(
            "Distortion_Coefficients"
        ).mat()
        cam_translation[str(cam_id).zfill(2)] = fs.getNode("cam_translation").mat()
        cam_rotation[str(cam_id).zfill(2)] = fs.getNode("cam_rotation").mat()
        fs.release()

    face_model_load = np.loadtxt(face_model_file)
    landmark_use = [20, 23, 26, 29, 15, 19]
    face_model = face_model_load[landmark_use, :]

    frame_name = os.path.basename(frame_folder)
    subject_name = os.path.basename(os.path.basename(os.path.dirname(frame_folder)))
    annotation_file_path = os.path.join(annotation_folder, subject_name + ".csv")
    df = pd.read_csv(annotation_file_path, header=None)
    for image_name in sorted(os.listdir(frame_folder)):
        if image_name not in remove_image_name_list:
            image_path = os.path.join(frame_folder, image_name)
            if save_path is not None:
                normalized_image_path = os.path.join(
                    save_path, image_name[:-4] + "_resized.png"
                )
            else:
                normalized_image_path = os.path.join(
                    frame_folder, image_name[:-4] + "_resized.png"
                )
            cam_id = image_name.split(".")[0][-2:]
            image = cv2.imread(image_path)
            image = cv2.undistort(
                image, camera_matrix[cam_id], camera_distortion[cam_id]
            )
            if cam_id in ["03", "06", "13"]:
                (h, w) = image.shape[:2]
                center = (w / 2, h / 2)
                M = cv2.getRotationMatrix2D(center, 180, 1.0)
                image = cv2.warpAffine(image, M, (w, h))
            hr = np.array(
                [
                    float(df[(df[0] == frame_name) & (df[1] == image_name)][7]),
                    float(df[(df[0] == frame_name) & (df[1] == image_name)][8]),
                    float(df[(df[0] == frame_name) & (df[1] == image_name)][9]),
                ]
            ).reshape(3, 1)
            ht = np.array(
                [
                    float(df[(df[0] == frame_name) & (df[1] == image_name)][10]),
                    float(df[(df[0] == frame_name) & (df[1] == image_name)][11]),
                    float(df[(df[0] == frame_name) & (df[1] == image_name)][12]),
                ]
            ).reshape(3, 1)

            image_normalized = normalizeData_face(
                image, face_model, hr, ht, camera_matrix[cam_id], 512, 1600.0
            )
            img_list.append(image_normalized)
            cam_ind_list.append(cam_id)
            cv2.imwrite(normalized_image_path, image_normalized)

    return img_list, cam_ind_list

def mpii_face_gaze_data_normalization(
    frame_folder, annotation_folder, camera_calibration_folder, face_model_file, save_path = None
):
    # frame_folder: path to the certain frame; e.g. 'xx/xx/subject0000/frame0000'
    # annotation_folder: path to the annotation files, suppose you saved all annotation .csv files in the same folder
    # camera_calibration_folder: path to the folder where all 18 camera calibration .xml files are stored
    # face_model_file: path to the file 'face_model.txt' from the ETH-XGaze

    if save_path is not None and not os.path.exists(save_path):
        os.makedirs(save_path)

    remove_image_name_list = []

    camera_matrix = {}
    camera_distortion = {}

    img_list = []
    cam_ind_list = []

    
    file_name = os.path.join(
        os.path.dirname(frame_folder), "Calibration/Camera.mat"
    )
    mat = scipy.io.loadmat(file_name)
    camera_matrix = mat.get("cameraMatrix")
    camera_distortion = mat.get(
        "distCoeffs"
    )

    face_model_load = np.loadtxt(face_model_file)
    landmark_use = [20, 23, 26, 29, 15, 19]
    face_model = face_model_load[landmark_use, :]

    frame_name = os.path.basename(frame_folder)
    subject_name = os.path.basename(os.path.basename(os.path.dirname(frame_folder)))
    annotation_file_path = os.path.join(os.path.dirname(frame_folder), subject_name + ".txt")
    with open(annotation_file_path) as anno_file:
        df = list(csv.reader(anno_file, delimiter=" "))
    for image_name in sorted(os.listdir(frame_folder)):
        if image_name not in remove_image_name_list:
            image_path = os.path.join(frame_folder, image_name)
            if save_path is not None:
                normalized_image_path = os.path.join(
                    save_path, image_name[:-4] + "_resized.png"
                )
            else:
                normalized_image_path = os.path.join(
                    frame_folder, image_name[:-4] + "_resized.png"
                )
            image = cv2.imread(image_path)
            image = cv2.undistort(
                image, camera_matrix, camera_distortion
            )
            for i in range(len(df)):
                if df[i][0] == frame_name + "/" + image_name:
                    hr = np.array(
                        [
                            float(df[i][15]),
                            float(df[i][16]),
                            float(df[i][17]),
                        ]
                    ).reshape(3, 1)

                    ht = np.array(
                        [
                            float(df[i][18]),
                            float(df[i][19]),
                            float(df[i][20]),
                        ]
                    ).reshape(3, 1)

                    fc = np.array(
                        [
                            float(df[i][21]),
                            float(df[i][22]),
                            float(df[i][23]),
                        ]
                    ).reshape(3,1)

                    break

            image_normalized = normalizeData_face(
                image, face_model, hr, ht, camera_matrix, 512, 1400.0, fc
            )
            img_list.append(image_normalized)
            cam_ind_list.append(0)
            cv2.imwrite(normalized_image_path, image_normalized)

    return img_list, cam_ind_list


def columbia_data_normalization(
    frame_folder, annotation_folder, camera_calibration_folder, face_model_file, save_path = None
):
    # frame_folder: path to the certain frame; e.g. 'xx/xx/subject0000/frame0000'
    # annotation_folder: path to the annotation files, suppose you saved all annotation .csv files in the same folder
    # camera_calibration_folder: path to the folder where all 18 camera calibration .xml files are stored
    # face_model_file: path to the file 'face_model.txt' from the ETH-XGaze

    if save_path is not None and not os.path.exists(save_path):
        os.makedirs(save_path)

    remove_image_name_list = []

    camera_matrix_small = {}
    camera_distortion_small = {}

    img_list = []
    cam_ind_list = []

    
    file_name = os.path.join(
        camera_calibration_folder, "cam" + str(0).zfill(2) + ".xml"
    )
    fs = cv2.FileStorage(file_name, cv2.FILE_STORAGE_READ)
    camera_matrix_small[str(0).zfill(2)] = fs.getNode("Camera_Matrix").mat()
    camera_distortion_small[str(0).zfill(2)] = fs.getNode(
        "Distortion_Coefficients"
    ).mat()
    fs.release()

    face_model_load = np.loadtxt(face_model_file)
    landmark_use = [20, 23, 26, 29, 15, 19]
    face_model = face_model_load[landmark_use, :]
    for image_name in sorted(os.listdir(frame_folder)):
        if image_name not in remove_image_name_list and "jpg" in image_name:
            
            image_name_parse  = image_name.split(".")[0]
            image_name_parse  = image_name_parse.split("_")
            cam_dir = int(image_name_parse[2][:-1])
            gaze_ver = int(image_name_parse[3][:-1])
            gaze_hor = int(image_name_parse[4][:-1])

            image_path = os.path.join(frame_folder, image_name)
            if save_path is not None:
                normalized_image_path = os.path.join(
                    save_path, image_name[:-4] + "_resized.png"
                )
            else:
                normalized_image_path = os.path.join(
                    frame_folder, image_name[:-4] + "_resized.png"
                )
            image = cv2.imread(image_path)
            image = image[:,864:4320]
            image = cv2.resize(image, (512, 512), interpolation = cv2.INTER_AREA)
            landmark_path = os.path.join(save_path,image_name[:-4] + "_lm2d.txt")
            landmark = np.loadtxt(landmark_path)
            ldms = landmark.reshape(68, 2)
            landmarks_sub = ldms[[36, 39, 42, 45, 31, 35], :]
            landmarks_sub = landmarks_sub.astype(float)  # input to solvePnP function must be float type
            landmarks_sub = landmarks_sub.reshape(6, 1, 2)
            hr,ht = estimateHeadPose(landmarks_sub,face_model.reshape(6, 1, 3), camera_matrix_small["00"], camera_distortion_small["00"])

            image_normalized = normalizeData_face(
                image, face_model, hr, ht, camera_matrix_small["00"], 512, 1600.0
            )
            img_list.append({'image' : image_normalized, 'p': cam_dir , 'v': gaze_ver , 'h': gaze_hor, 'hr' : hr, 'ht' : ht })
            cam_ind_list.append(0)
            cv2.imwrite(normalized_image_path, image_normalized)

    return img_list, cam_ind_list

def gaze_capture_data_normalization(frame_folder, annotation_folder, camera_calibration_folder, face_model_file, save_path=None):
    undistort = Undistorter()
    if save_path is not None and not os.path.exists(save_path):
        os.makedirs(save_path)

    remove_image_name_list = []

    img_list = []
    cam_ind_list = []

    face_model_load = np.loadtxt(face_model_file)
    landmark_use = [20, 23, 26, 29, 15, 19]
    face_model = face_model_load[landmark_use, :]
    for image_name in sorted(os.listdir(frame_folder)):
        if image_name not in remove_image_name_list:
            image_path = os.path.join(frame_folder, image_name)
            if save_path is not None:
                normalized_image_path = os.path.join(
                    save_path, image_name[:-4] + "_resized.png"
                )
            else:
                normalized_image_path = os.path.join(
                    frame_folder, image_name[:-4] + "_resized.png"
                )
            image = cv2.imread(image_path)
            idx = sorted(os.listdir(frame_folder)).index(image_name)
            fx,fy,cx,cy = annotation_folder['camera_parameters'][idx,:]
            camera_matrix = np.array([[fx,0,cx],[0,fy,cy],[0,0,1]], dtype=np.float64)
            distortion = annotation_folder['distortion_parameters'][idx,:]
            image = undistort(
                image, camera_matrix, distortion, is_gazecapture=True
            )
            hr = annotation_folder['head_pose'][idx,:3].reshape(3,1)
            ht = annotation_folder['head_pose'][idx,3:].reshape(3,1)
            rotate_mat, _ = cv2.Rodrigues(hr)
            face_model_3d_coordinates = np.load(camera_calibration_folder)
            landmarks_3d = np.matmul(rotate_mat, face_model_3d_coordinates.T).T
            landmarks_3d += hr.T
            fc = np.mean(landmarks_3d[10:12,:], axis=0).reshape(3,1)
            g_t = annotation_folder['3d_gaze_target'][idx,:].reshape(3,1)
            g = g_t - fc
            image_normalized = normalizeData_face(
                image, face_model, hr, ht, camera_matrix, 512, 1200.0
            )
            img_list.append({'image':image_normalized, 'hr':hr, 'ht':ht, 'fc':fc, 'g':g})
            cam_ind_list.append(0)
            cv2.imwrite(normalized_image_path, image_normalized)
    return img_list, cam_ind_list



class Undistorter:

    _map = None
    _previous_parameters = None

    def __call__(self, image, camera_matrix, distortion, is_gazecapture=False):
        h, w, _ = image.shape
        all_parameters = np.concatenate([camera_matrix.flatten(),
                                         distortion.flatten(),
                                         [h, w]])
        if (self._previous_parameters is None
                or len(self._previous_parameters) != len(all_parameters)
                or not np.allclose(all_parameters, self._previous_parameters)):
            print('Distortion map parameters updated.')
            self._map = cv2.initUndistortRectifyMap(
                camera_matrix, distortion, R=None,
                newCameraMatrix=camera_matrix if is_gazecapture else None,
                size=(w, h), m1type=cv2.CV_32FC1)
            print('fx: %.2f, fy: %.2f, cx: %.2f, cy: %.2f' % (
                    camera_matrix[0, 0], camera_matrix[1, 1],
                    camera_matrix[0, 2], camera_matrix[1, 2]))
            self._previous_parameters = np.copy(all_parameters)

        # Apply
        return cv2.remap(image, self._map[0], self._map[1], cv2.INTER_LINEAR)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="The code to process a directory of images."
    )
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--img_dir", type=str, required=True)
    args = parser.parse_args()

    gpu_id = args.gpu_id
    img_dir = args.img_dir

    main_process(img_dir, gpu_id)