import argparse
import csv
import os
import shutil
import time
from glob import glob
import requests
from bs4 import BeautifulSoup

import h5py
import numpy as np
import math
import json

import wandb
from pre_processing.data_process import main_process
from surface_fitting.nl3dmm.fitting_nl3dmm import FittingNL3DMM
from surface_fitting.nl3dmm.gen_nl3dmm_render_res import GenRenderRes
from utils.gaze_estimation_utils import convert_to_head_coordinate_system
import cv2

def surface_fitting(gpu_id, log, img_dir, img_size, intermediate_size, batch_size):
    tt = FittingNL3DMM(
        img_size=img_size,
        intermediate_size=intermediate_size,
        gpu_id=gpu_id,
        batch_size=batch_size,
        img_dir=img_dir,
    )
    surface_list = tt.main_process()
    if log:
        tt = GenRenderRes(gpu_id, log)
        for res in surface_list:
            tt.render_3dmm_from_dict(temp_dict=res)

    return surface_list

def listFD(url, ext=""):
    page = requests.get(url).text
    soup = BeautifulSoup(page, "html.parser")
    return [
        "/" + node.get("href")
        for node in soup.find_all("a")
        if node.get("href").endswith(ext)
    ]

def calculate_gaze_direction(p,v,h):
    return np.array([(-v*math.pi)/180.0, ((p-h) * math.pi)/180.0]).reshape(2)

def calc_normalized_head_pose(face_model, hr, ht):
    hR = cv2.Rodrigues(hr)[0]
    ht = ht.reshape((3, 1))
    Fc = np.dot(hR, face_model.T) + ht
    two_eye_center = np.mean(Fc[:, 0:4], axis=1).reshape((3, 1))
    mouth_center = np.mean(Fc[:, 4:6], axis=1).reshape((3, 1))
    face_center = np.mean(np.concatenate((two_eye_center, mouth_center), axis=1), axis=1).reshape((3, 1))

    distance = np.linalg.norm(face_center)  # actual distance between eye and original camera

    hRx = hR[:, 0]
    forward = (face_center / distance).reshape(3)
    down = np.cross(forward, hRx)
    down /= np.linalg.norm(down)
    right = np.cross(down, forward)
    right /= np.linalg.norm(right)
    R = np.c_[right, down, forward].T  # rotation matrix R

    # head pose after normalization
    hR_norm = np.dot(R, hR)  # head pose rotation matrix in normalized space
    hr_norm = cv2.Rodrigues(hR_norm)[0]  # convert rotation matrix to rotation vectors

    head = hr_norm.reshape(1, 3)
    M = cv2.Rodrigues(head)[0]
    Zv = M[:, 2]
    head_2d = np.array([math.asin(Zv[1]), math.atan2(Zv[0], Zv[2])])

    return head_2d.reshape(2)


def eth_xgaze_dataset_iteration_full(
    dataset_name,
    gpu_id,
    log,
    dir,
    dataset_url,
    dataset_path,
    annotation_path,
    cam_calibration_path,
    face_model_path,
    subject_start,
    subject_end,
    is_over_write,
    img_size,
    intermediate_size,
    batch_size,
):
    if not os.path.exists(dir):
        os.makedirs(dir)
    for sub_id in range(subject_start, subject_end):
        start_time = time.time()
        save_index = 0

        used_cam = 18

        subject_folder = "./data/train/subject" + str(sub_id).zfill(4)
        local_folder = "./subject" + str(sub_id).zfill(4)
        url_path = os.path.join(dataset_url, subject_folder[2:])
        print(url_path)

        if dataset_path is not None:
            local_folder = os.path.join(dataset_path, "subject" + str(sub_id).zfill(4))
        else:
            os.system(
                f'wget -r --no-parent --no-verbose -nH --cut-dirs 6 --reject="index.html*" {url_path}'
            )

        if not os.path.exists(local_folder):
            continue

        print("Processing ", subject_folder)

        # output file
        hdf_fpath = os.path.join(dir, "xgaze_subject" + str(sub_id).zfill(4) + ".h5")
        if is_over_write:
            if os.path.exists(hdf_fpath):
                print("Overwrite the file ", hdf_fpath)
                os.remove(hdf_fpath)
        else:
            if os.path.exists(hdf_fpath):
                print("Skip the file ", subject_folder, " since it is already exist")
                continue

        output_h5_id = h5py.File(hdf_fpath, "w")
        print("output file save to ", hdf_fpath)

        output_frame_index = []
        output_cam_index = []

        output_landmarks = []
        output_face_patch = []

        output_left_eye_mask = []
        output_right_eye_mask = []
        output_head_mask = []

        output_latent_codes = []
        output_w2c_Rmat = []
        output_w2c_Tvec = []
        output_inmat = []
        output_c2w_Rmat = []
        output_c2w_Tvec = []
        output_inv_inmat = []

        output_pitchyaw = []
        output_pitchyaw_head_coordinate_system = []
        output_gaze_direction_3d = []
        output_face_head_pose =[]

        try:
            frames_list = [
                name
                for name in os.listdir(local_folder)
                if os.path.isdir(os.path.join(local_folder, name))
            ]
        except OSError as e:
            continue

        label_path = os.path.join(
            annotation_path,
            "subject" + str(sub_id).zfill(4) + ".csv",
        )
        with open(label_path) as anno_file:
            content = list(csv.reader(anno_file, delimiter=","))

        try:
            frame_before = frames_list[-1]
            previous_frame=None
            if dataset_path is None:
                save_path = None
            else:
                save_path = os.path.join("subject" + str(sub_id).zfill(4), frame_before)
            (
            images,
            cam_indices,
            head_masks,
            landmarks,
            left_eye_masks,
            right_eye_masks,
                ) = main_process(
            dataset_name,
            f"{local_folder}/{frame_before}",
            gpu_id,
            annotation_path,
            cam_calibration_path,
            face_model_path,
            log,
            save_path,
            previous_frame,
            )
            previous_frame=landmarks
            shutil.rmtree(save_path, ignore_errors=True)
        except:
            previous_frame=None
            frame_before = frames_list[-1]
            if dataset_path is None:
                save_path = None
            else:
                save_path = os.path.join("subject" + str(sub_id).zfill(4), frame_before)
            shutil.rmtree(save_path, ignore_errors=True)

        check = 0
        num_cam = 18
        frames_list.sort()
        surface_list_first_frame = []
        for idx, frame in enumerate(frames_list):
            face_model_load = np.loadtxt(face_model_path)
            frame_index = int(frame[5:])

            if dataset_path is None:
                save_path = None
            else:
                save_path = os.path.join("subject" + str(sub_id).zfill(4), frame)

            (
                images,
                cam_indices,
                head_masks,
                landmarks,
                left_eye_masks,
                right_eye_masks,
            ) = main_process(
                dataset_name,
                f"{local_folder}/{frame}",
                gpu_id,
                annotation_path,
                cam_calibration_path,
                face_model_path,
                log,
                save_path,
                previous_frame,
            )
            previous_frame=landmarks

            if save_path is not None:
                nl_3dmm_path = save_path
            else:
                nl_3dmm_path = f"{local_folder}/{frame}"

            surface_list_first_frame = surface_fitting(
                gpu_id,
                log,
                nl_3dmm_path,
                img_size,
                intermediate_size,
                batch_size,
            )

            if not output_frame_index:
                total_data = len(frames_list)
                output_frame_index = output_h5_id.create_dataset(
                    "frame_index",
                    shape=(total_data * used_cam, 1),
                    dtype=np.uint8,
                    chunks=(1, 1),
                )
                output_cam_index = output_h5_id.create_dataset(
                    "cam_index",
                    shape=(total_data * used_cam, 1),
                    dtype=np.uint8,
                    chunks=(1, 1),
                )
                output_landmarks = output_h5_id.create_dataset(
                    "facial_landmarks",
                    shape=(total_data * used_cam, 68, 2),
                    dtype=np.float,
                    chunks=(1, 68, 2),
                )
                output_face_patch = output_h5_id.create_dataset(
                    "face_patch",
                    shape=(total_data * used_cam, 512, 512, 3),
                    compression="lzf",
                    dtype=np.uint8,
                    chunks=(1, 512, 512, 3),
                )
                output_left_eye_mask = output_h5_id.create_dataset(
                    "left_eye_mask",
                    shape=(total_data * used_cam, 512, 512),
                    compression="lzf",
                    dtype=np.uint8,
                    chunks=(1, 512, 512),
                )
                output_right_eye_mask = output_h5_id.create_dataset(
                    "right_eye_mask",
                    shape=(total_data * used_cam, 512, 512),
                    compression="lzf",
                    dtype=np.uint8,
                    chunks=(1, 512, 512),
                )
                output_head_mask = output_h5_id.create_dataset(
                    "head_mask",
                    shape=(total_data * used_cam, 512, 512),
                    compression="lzf",
                    dtype=np.uint8,
                    chunks=(1, 512, 512),
                )
                output_latent_codes = output_h5_id.create_dataset(
                    "latent_codes",
                    shape=(total_data * used_cam, 306),
                    compression="lzf",
                    dtype=np.float,
                    chunks=(1, 306),
                )
                output_w2c_Rmat = output_h5_id.create_dataset(
                    "w2c_Rmat",
                    shape=(total_data * used_cam, 3, 3),
                    compression="lzf",
                    dtype=np.float,
                    chunks=(1, 3, 3),
                )
                output_w2c_Tvec = output_h5_id.create_dataset(
                    "w2c_Tvec",
                    shape=(total_data * used_cam, 3),
                    compression="lzf",
                    dtype=np.float,
                    chunks=(1, 3),
                )
                output_inmat = output_h5_id.create_dataset(
                    "inmat",
                    shape=(total_data * used_cam, 3, 3),
                    compression="lzf",
                    dtype=np.float,
                    chunks=(1, 3, 3),
                )
                output_c2w_Rmat = output_h5_id.create_dataset(
                    "c2w_Rmat",
                    shape=(total_data * used_cam, 3, 3),
                    compression="lzf",
                    dtype=np.float,
                    chunks=(1, 3, 3),
                )
                output_c2w_Tvec = output_h5_id.create_dataset(
                    "c2w_Tvec",
                    shape=(total_data * used_cam, 3),
                    compression="lzf",
                    dtype=np.float,
                    chunks=(1, 3),
                )
                output_inv_inmat = output_h5_id.create_dataset(
                    "inv_inmat",
                    shape=(total_data * used_cam, 3, 3),
                    compression="lzf",
                    dtype=np.float,
                    chunks=(1, 3, 3),
                )
                output_pitchyaw = output_h5_id.create_dataset(
                    "pitchyaw",
                    shape=(total_data * used_cam, 2),
                    compression="lzf",
                    dtype=np.float,
                    chunks=(1, 2),
                )
                output_pitchyaw_head_coordinate_system = output_h5_id.create_dataset(
                    "pitchyaw_head",
                    shape=(total_data * used_cam, 2),
                    compression="lzf",
                    dtype=np.float,
                    chunks=(1, 2),
                )
                output_gaze_direction_3d = output_h5_id.create_dataset(
                    "pitchyaw_3d",
                    shape=(total_data * used_cam, 3),
                    compression="lzf",
                    dtype=np.float,
                    chunks=(1, 3),
                )
                output_face_head_pose = output_h5_id.create_dataset(
                    "face_head_pose", 
                    shape=(total_data * used_cam, 2),
                    compression="lzf",
                    dtype=np.float, 
                    chunks=(1, 2))

            for i in range(len(images)):
                output_frame_index[save_index] = frame_index
                output_cam_index[save_index] = cam_indices[i]
                output_landmarks[save_index] = landmarks[i]
                output_face_patch[save_index] = images[i]
                output_left_eye_mask[save_index] = left_eye_masks[i]
                output_right_eye_mask[save_index] = right_eye_masks[i]
                output_head_mask[save_index] = head_masks[i]
                output_latent_codes[save_index] = (
                        surface_list_first_frame[0]["code"]
                        + surface_list_first_frame[1]["code"]
                        + surface_list_first_frame[2]["code"]
                    ) / 3.0
                output_w2c_Rmat[save_index] = surface_list_first_frame[i]["w2c_Rmat"]
                output_w2c_Tvec[save_index] = surface_list_first_frame[i]["w2c_Tvec"]
                output_inmat[save_index] = surface_list_first_frame[i]["inmat"]
                output_c2w_Rmat[save_index] = surface_list_first_frame[i]["c2w_Rmat"]
                output_c2w_Tvec[save_index] = surface_list_first_frame[i]["c2w_Tvec"]
                output_inv_inmat[save_index] = surface_list_first_frame[i]["inv_inmat"]
                output_pitchyaw[save_index] = [content[check][2], content[check][3]]

                gaze_direction_3d = (
                    np.array(content[check + int(cam_indices[i])][4:7])
                    .astype(np.float32)
                    .reshape(3, 1)
                )
                hr = np.array(
                    [
                        float(content[check + int(cam_indices[i])][7]),
                        float(content[check + int(cam_indices[i])][8]),
                        float(content[check + int(cam_indices[i])][9]),
                    ]
                ).reshape(3, 1)
                ht = np.array(
                    [
                        float(content[check + int(cam_indices[i])][10]),
                        float(content[check + int(cam_indices[i])][11]),
                        float(content[check + int(cam_indices[i])][12]),
                    ]
                ).reshape(3, 1)
                output_pitchyaw_head_coordinate_system[
                    save_index
                ] = convert_to_head_coordinate_system(
                    face_model_load,
                    gaze_direction_3d,
                    hr,
                    ht,
                )

                output_gaze_direction_3d[save_index] = [
                    content[check + int(cam_indices[i])][4],
                    content[check + int(cam_indices[i])][5],
                    content[check + int(cam_indices[i])][6],
                ]

                landmark_use = [20, 23, 26, 29, 15, 19]
                face_model = face_model_load[landmark_use, :]

                output_face_head_pose[save_index] = calc_normalized_head_pose(face_model, hr, ht)

                save_index += 1

            check += num_cam

            if save_path is not None:
                shutil.rmtree(save_path, ignore_errors=True)

        output_h5_id.close()
        print("close the h5 file")

        print("finish the subject: ", sub_id)
        elapsed_time = time.time() - start_time
        print("///////////////////////////////////")
        print(
            "Running time is {:02d}:{:02d}:{:02d}".format(
                int(elapsed_time // 3600),
                int(elapsed_time % 3600 // 60),
                int(elapsed_time % 60),
            )
        )
        
        shutil.rmtree("subject" + str(sub_id).zfill(4), ignore_errors=True)

    print("All Subjects were processed")

def mpii_face_gaze_dataset_iteration_full(
    dataset_name,
    gpu_id,
    log,
    dir,
    dataset_path,
    annotation_path,
    cam_calibration_path,
    face_model_path,
    subject_start,
    subject_end,
    is_over_write,
    img_size,
    intermediate_size,
    batch_size,
):
    if not os.path.exists(dir):
            os.makedirs(dir)
    for sub_id in range(subject_start, subject_end):
        start_time = time.time()
        save_index = 0
        local_folder = os.path.join(dataset_path, "p" + str(sub_id).zfill(2))
        

        if not os.path.exists(local_folder):
            continue

        # output file
        hdf_fpath = os.path.join(dir, "mpii_subject" + str(sub_id).zfill(4) + ".h5")
        if is_over_write:
            if os.path.exists(hdf_fpath):
                print("Overwrite the file ", hdf_fpath)
                os.remove(hdf_fpath)
        else:
            if os.path.exists(hdf_fpath):
                print("Skip the file ", hdf_fpath, " since it is already exist")
                continue

        output_h5_id = h5py.File(hdf_fpath, "w")
        print("output file save to ", hdf_fpath)

        output_frame_index = []
        output_cam_index = []

        output_landmarks = []
        output_face_patch = []

        output_left_eye_mask = []
        output_right_eye_mask = []
        output_head_mask = []

        output_latent_codes = []
        output_w2c_Rmat = []
        output_w2c_Tvec = []
        output_inmat = []
        output_c2w_Rmat = []
        output_c2w_Tvec = []
        output_inv_inmat = []

        output_pitchyaw = []
        output_face_center = []
        output_pitchyaw_head_coordinate_system = []
        output_gaze_direction_3d = []
        output_face_head_pose =[]

        try:
            frames_list = [
                name
                for name in os.listdir(local_folder)
                if os.path.isdir(os.path.join(local_folder, name))
            ]
        except OSError as e:
            continue

        label_path = os.path.join(
            dataset_path,
            "p" + str(sub_id).zfill(2),
            "p" + str(sub_id).zfill(2) + ".txt",
        )
        
        
        with open(label_path) as anno_file:
            content = list(csv.reader(anno_file, delimiter=" "))

        
        frames_list.sort()
        surface_list_first_frame = []
        for idx, frame in enumerate(frames_list):
            if 'day' not in frame:
                continue
            face_model_load = np.loadtxt(face_model_path)
            frame_index = int(frame[3:])
            
            save_path = os.path.join("p" + str(sub_id).zfill(2), frame)

            (
                images,
                cam_indices,
                head_masks,
                landmarks,
                left_eye_masks,
                right_eye_masks,
            ) = main_process(
                dataset_name,
                f"{local_folder}/{frame}",
                gpu_id,
                annotation_path,
                cam_calibration_path,
                face_model_path,
                log,
                save_path,
            )
            if save_path is not None:
                nl_3dmm_path = save_path
            else:
                nl_3dmm_path = f"{local_folder}/{frame}"  

            surface_list_first_frame = surface_fitting( 
                gpu_id,
                log,
                nl_3dmm_path,
                img_size,
                intermediate_size,
                batch_size,
            )

            if not output_frame_index:
                count_num_img = 0
                for frame in frames_list:
                    if 'day' not in frame:
                        continue
                    img_path_list = [x for x in glob("%s/*.jpg" % os.path.join(dataset_path, "p" + str(sub_id).zfill(2) , frame))]
                    count_num_img+= len(img_path_list)

                output_frame_index = output_h5_id.create_dataset(
                    "frame_index",
                    shape=(count_num_img, 1),
                    dtype=np.uint8,
                    chunks=(1, 1),
                )
                output_cam_index = output_h5_id.create_dataset(
                    "cam_index",
                    shape=(count_num_img, 1),
                    dtype=np.uint8,
                    chunks=(1, 1),
                )
                output_landmarks = output_h5_id.create_dataset(
                    "facial_landmarks",
                    shape=(count_num_img, 68, 2),
                    dtype=np.float,
                    chunks=(1, 68, 2),
                )
                output_face_patch = output_h5_id.create_dataset(
                    "face_patch",
                    shape=(count_num_img, 512, 512, 3),
                    compression="lzf",
                    dtype=np.uint8,
                    chunks=(1, 512, 512, 3),
                )            
                output_left_eye_mask = output_h5_id.create_dataset(
                    "left_eye_mask",
                    shape=(count_num_img, 512, 512),
                    compression="lzf",
                    dtype=np.uint8,
                    chunks=(1, 512, 512),
                )
                output_right_eye_mask = output_h5_id.create_dataset(
                    "right_eye_mask",
                    shape=(count_num_img, 512, 512),
                    compression="lzf",
                    dtype=np.uint8,
                    chunks=(1, 512, 512),
                )
                output_head_mask = output_h5_id.create_dataset(
                    "head_mask",
                    shape=(count_num_img, 512, 512),
                    compression="lzf",
                    dtype=np.uint8,
                    chunks=(1, 512, 512),
                )
                output_latent_codes = output_h5_id.create_dataset(
                    "latent_codes",
                    shape=(count_num_img, 306),
                    compression="lzf",
                    dtype=np.float,
                    chunks=(1, 306),
                )
                output_w2c_Rmat = output_h5_id.create_dataset(
                    "w2c_Rmat",
                    shape=(count_num_img, 3, 3),
                    compression="lzf",
                    dtype=np.float,
                    chunks=(1, 3, 3),
                )
                output_w2c_Tvec = output_h5_id.create_dataset(
                    "w2c_Tvec",
                    shape=(count_num_img, 3),
                    compression="lzf",
                    dtype=np.float,
                    chunks=(1, 3),
                )
                output_inmat = output_h5_id.create_dataset(
                    "inmat",
                    shape=(count_num_img, 3, 3),
                    compression="lzf",
                    dtype=np.float,
                    chunks=(1, 3, 3),
                )
                output_c2w_Rmat = output_h5_id.create_dataset(
                    "c2w_Rmat",
                    shape=(count_num_img, 3, 3),
                    compression="lzf",
                    dtype=np.float,
                    chunks=(1, 3, 3),
                )
                output_c2w_Tvec = output_h5_id.create_dataset(
                    "c2w_Tvec",
                    shape=(count_num_img, 3),
                    compression="lzf",
                    dtype=np.float,
                    chunks=(1, 3),
                )
                output_inv_inmat = output_h5_id.create_dataset(
                    "inv_inmat",
                    shape=(count_num_img, 3, 3),
                    compression="lzf",
                    dtype=np.float,
                    chunks=(1, 3, 3),
                )
                output_pitchyaw = output_h5_id.create_dataset(
                    "pitchyaw",
                    shape=(count_num_img, 2),
                    compression="lzf",
                    dtype=np.float,
                    chunks=(1, 2),
                )
                output_face_center = output_h5_id.create_dataset(
                    "face_center",
                    shape=(count_num_img, 3),
                    compression="lzf",
                    dtype=np.float,
                    chunks=(1, 3),
                )
                output_pitchyaw_head_coordinate_system = output_h5_id.create_dataset(
                    "pitchyaw_head",
                    shape=(count_num_img, 2),
                    compression="lzf",
                    dtype=np.float,
                    chunks=(1, 2),
                )
                output_gaze_direction_3d = output_h5_id.create_dataset(
                    "pitchyaw_3d",
                    shape=(count_num_img, 3),
                    compression="lzf",
                    dtype=np.float,
                    chunks=(1, 3),
                )
                output_face_head_pose = output_h5_id.create_dataset(
                    "face_head_pose", 
                    shape=(count_num_img, 2),
                    compression="lzf",
                    dtype=np.float, 
                    chunks=(1, 2))
            

            for i in range(len(images)):
                output_frame_index[save_index] = frame_index
                output_cam_index[save_index] = cam_indices[i]
                output_landmarks[save_index] = landmarks[i]
                output_face_patch[save_index] = images[i]
                output_left_eye_mask[save_index] = left_eye_masks[i]
                output_right_eye_mask[save_index] = right_eye_masks[i]
                output_head_mask[save_index] = head_masks[i] 
                output_latent_codes[save_index] = surface_list_first_frame[i]["code"]
                output_w2c_Rmat[save_index] = surface_list_first_frame[i]["w2c_Rmat"]
                output_w2c_Tvec[save_index] = surface_list_first_frame[i]["w2c_Tvec"]  
                output_inmat[save_index] = surface_list_first_frame[i]["inmat"]
                output_c2w_Rmat[save_index] = surface_list_first_frame[i]["c2w_Rmat"]
                output_c2w_Tvec[save_index] = surface_list_first_frame[i]["c2w_Tvec"]
                output_inv_inmat[save_index] = surface_list_first_frame[i]["inv_inmat"]
                output_pitchyaw[save_index] = [float(content[save_index][1]), float(content[save_index][2])]
                output_face_center[save_index] = [
                                                    float(content[save_index][21]),
                                                    float(content[save_index][22]),
                                                    float(content[save_index][23]),
                                                ]
                gaze_direction_3d = (
                    np.array(content[save_index][24:27])
                    .astype(np.float32)
                    .reshape(3, 1)
                )
                hr = np.array(
                    [
                        float(content[save_index][15]),
                        float(content[save_index][16]),
                        float(content[save_index][17]),
                    ]
                ).reshape(3, 1)
                ht = np.array(
                    [
                        float(content[save_index][18]),
                        float(content[save_index][19]),
                        float(content[save_index][20]),
                    ]
                ).reshape(3, 1)
                output_pitchyaw_head_coordinate_system[
                    save_index
                ] = convert_to_head_coordinate_system(
                    face_model_load,
                    gaze_direction_3d,
                    hr,
                    ht,
                )

                output_gaze_direction_3d[save_index] = [
                    float(content[save_index][24]),
                    float(content[save_index][25]),
                    float(content[save_index][26]),
                ]
                
                landmark_use = [20, 23, 26, 29, 15, 19]
                face_model = face_model_load[landmark_use, :]
                output_face_head_pose[save_index] = calc_normalized_head_pose(face_model, hr, ht)

                save_index += 1

            shutil.rmtree(save_path, ignore_errors=True)

        output_h5_id.close()
        print("close the h5 file")

        print("finish the subject: ", sub_id)
        elapsed_time = time.time() - start_time
        print("///////////////////////////////////")
        print(
            "Running time is {:02d}:{:02d}:{:02d}".format(
                int(elapsed_time // 3600),
                int(elapsed_time % 3600 // 60),
                int(elapsed_time % 60),
            )
        )

        shutil.rmtree("p" + str(sub_id).zfill(2))

    print("All Subjects were processed")

        

def columbia_gaze_dataset_iteration_full(
    dataset_name,
    gpu_id,
    log,
    dir,
    dataset_path,
    annotation_path,
    cam_calibration_path,
    face_model_path,
    subject_start,
    subject_end,
    is_over_write,
    img_size,
    intermediate_size,
    batch_size,
):
    if not os.path.exists(dir):
            os.makedirs(dir)
    for sub_id in range(subject_start, subject_end):
        start_time = time.time()
        save_index = 0
        local_folder = os.path.join(dataset_path, str(sub_id).zfill(4))
        

        if not os.path.exists(local_folder):
            continue

        # output file
        hdf_fpath = os.path.join(dir, "columbia_subject" + str(sub_id).zfill(4) + ".h5")
        if is_over_write:
            if os.path.exists(hdf_fpath):
                print("Overwrite the file ", hdf_fpath)
                os.remove(hdf_fpath)
        else:
            if os.path.exists(hdf_fpath):
                print("Skip the file ", hdf_fpath, " since it is already exist")
                continue

        output_h5_id = h5py.File(hdf_fpath, "w")
        print("output file save to ", hdf_fpath)

        face_model_load = np.loadtxt(face_model_path)

        output_cam_index = []

        output_landmarks = []
        output_face_patch = []

        output_left_eye_mask = []
        output_right_eye_mask = []
        output_head_mask = []

        output_latent_codes = []
        output_w2c_Rmat = []
        output_w2c_Tvec = []
        output_inmat = []
        output_c2w_Rmat = []
        output_c2w_Tvec = []
        output_inv_inmat = []

        output_pitchyaw_head_coordinate_system = []
        output_params_gaze_direction = []
        output_face_head_pose =[]
       
        save_path = os.path.join(str(sub_id).zfill(2))

        (
            images,
            cam_indices,
            head_masks,
            landmarks,
            left_eye_masks,
            right_eye_masks,
        ) = main_process(
            dataset_name,
            f"{local_folder}",
            gpu_id,
            annotation_path,
            cam_calibration_path,
            face_model_path,
            log,
            save_path, 
        )
        if save_path is not None:
            nl_3dmm_path = save_path
        else:
            nl_3dmm_path = f"{local_folder}"  
        
        surface_list_first_frame = surface_fitting( 
            gpu_id,
            log,
            nl_3dmm_path,
            img_size,
            intermediate_size,
            batch_size,
        )
        
        if not output_cam_index:
            count_num_img = 105

            output_cam_index = output_h5_id.create_dataset(
                "cam_index",
                shape=(count_num_img, 1),
                dtype=np.uint8,
                chunks=(1, 1),
            )
            output_landmarks = output_h5_id.create_dataset(
                "facial_landmarks",
                shape=(count_num_img, 68, 2),
                dtype=np.float,
                chunks=(1, 68, 2),
            )
            output_face_patch = output_h5_id.create_dataset(
                "face_patch",
                shape=(count_num_img, 512, 512, 3),
                compression="lzf",
                dtype=np.uint8,
                chunks=(1, 512, 512, 3),
            )     
            output_left_eye_mask = output_h5_id.create_dataset(
                "left_eye_mask",
                shape=(count_num_img, 512, 512),
                compression="lzf",
                dtype=np.uint8,
                chunks=(1, 512, 512),
            )
            output_right_eye_mask = output_h5_id.create_dataset(
                "right_eye_mask",
                shape=(count_num_img, 512, 512),
                compression="lzf",
                dtype=np.uint8,
                chunks=(1, 512, 512),
            )
            output_head_mask = output_h5_id.create_dataset(
                "head_mask",
                shape=(count_num_img, 512, 512),
                compression="lzf",
                dtype=np.uint8,
                chunks=(1, 512, 512),
            )
            output_latent_codes = output_h5_id.create_dataset(
                "latent_codes",
                shape=(count_num_img, 306),
                compression="lzf",
                dtype=np.float,
                chunks=(1, 306),
            )
            output_w2c_Rmat = output_h5_id.create_dataset(
                "w2c_Rmat",
                shape=(count_num_img, 3, 3),
                compression="lzf",
                dtype=np.float,
                chunks=(1, 3, 3),
            )
            output_w2c_Tvec = output_h5_id.create_dataset(
                "w2c_Tvec",
                shape=(count_num_img, 3),
                compression="lzf",
                dtype=np.float,
                chunks=(1, 3),
            )
            output_inmat = output_h5_id.create_dataset(
                "inmat",
                shape=(count_num_img, 3, 3),
                compression="lzf",
                dtype=np.float,
                chunks=(1, 3, 3),
            )
            output_c2w_Rmat = output_h5_id.create_dataset(
                "c2w_Rmat",
                shape=(count_num_img, 3, 3),
                compression="lzf",
                dtype=np.float,
                chunks=(1, 3, 3),
            )
            output_c2w_Tvec = output_h5_id.create_dataset(
                "c2w_Tvec",
                shape=(count_num_img, 3),
                compression="lzf",
                dtype=np.float,
                chunks=(1, 3),
            )
            output_inv_inmat = output_h5_id.create_dataset(
                "inv_inmat",
                shape=(count_num_img, 3, 3),
                compression="lzf",
                dtype=np.float,
                chunks=(1, 3, 3),
            )
            output_pitchyaw_head_coordinate_system = output_h5_id.create_dataset(
                "pitchyaw_head",
                shape=(count_num_img, 2),
                compression="lzf",
                dtype=np.float,
                chunks=(1, 2),
            )
            output_params_gaze_direction = output_h5_id.create_dataset(
                "pitchyaw_3d",
                shape=(count_num_img, 3),
                compression="lzf",
                dtype=np.float,
                chunks=(1, 3),
            )
            output_face_head_pose = output_h5_id.create_dataset(
                "face_head_pose", 
                shape=(count_num_img, 2),
                compression="lzf",
                dtype=np.float, 
                chunks=(1, 2))

        for i in range(len(images)):
            output_cam_index[save_index] = cam_indices[i]
            output_landmarks[save_index] = landmarks[i]
            output_face_patch[save_index] = images[i]['image']
            output_left_eye_mask[save_index] = left_eye_masks[i]
            output_right_eye_mask[save_index] = right_eye_masks[i]
            output_head_mask[save_index] = head_masks[i] 
            output_latent_codes[save_index] = surface_list_first_frame[i]["code"]
            output_w2c_Rmat[save_index] = surface_list_first_frame[i]["w2c_Rmat"]
            output_w2c_Tvec[save_index] = surface_list_first_frame[i]["w2c_Tvec"]  
            output_inmat[save_index] = surface_list_first_frame[i]["inmat"]
            output_c2w_Rmat[save_index] = surface_list_first_frame[i]["c2w_Rmat"]
            output_c2w_Tvec[save_index] = surface_list_first_frame[i]["c2w_Tvec"]
            output_inv_inmat[save_index] = surface_list_first_frame[i]["inv_inmat"]
            
           
            output_pitchyaw_head_coordinate_system[
                save_index
            ] = calculate_gaze_direction(
                p = images[i]['p'],
                v = images[i]['v'],
                h = images[i]['h'],
            )

            output_params_gaze_direction[save_index] = [
                images[i]['p'],
                images[i]['v'],
                images[i]['h'],
            ]

            landmark_use = [20, 23, 26, 29, 15, 19]
            face_model = face_model_load[landmark_use, :]
            output_face_head_pose[save_index] = calc_normalized_head_pose(face_model, images[i]['hr'], images[i]['ht'])

            save_index += 1
            

        output_h5_id.close()
        print("close the h5 file")

        print("finish the subject: ", sub_id)
        elapsed_time = time.time() - start_time
        print("///////////////////////////////////")
        print(
            "Running time is {:02d}:{:02d}:{:02d}".format(
                int(elapsed_time // 3600),
                int(elapsed_time % 3600 // 60),
                int(elapsed_time % 60),
            )
        )

        shutil.rmtree(save_path, ignore_errors=True)

    print("All Subjects were processed")
        


def gaze_capture_dataset_iteration_full(
    dataset_name,
    gpu_id,
    log,
    dir,
    dataset_path,
    annotation_path,
    cam_calibration_path,
    face_model_path,
    subject_start,
    subject_end,
    is_over_write,
    img_size,
    intermediate_size,
    batch_size,
):
    if not os.path.exists(dir):
            os.makedirs(dir)

    with open('./data/gaze_capture/gazecapture_split.json') as load_f:
        subjects_list = json.load(load_f)
    test_subjects_list = subjects_list['test']

    for test_subject in test_subjects_list:
        gazecapture_supplement = h5py.File('./data/gaze_capture/GazeCapture_supplementary.h5', 'r')
        if int(test_subject) <= 741:
            continue
        if test_subject in gazecapture_supplement.keys():
            start_time = time.time()
            save_index = 0
            local_folder = os.path.join(dataset_path, test_subject, 'frames')

            if not os.path.exists(local_folder):
                continue

            #output file
            hdf_fpath = os.path.join(dir, "gaze_capture_subject" + test_subject + ".h5")
            if is_over_write:
                if os.path.exists(hdf_fpath):
                    print("Overwrite the file ", hdf_fpath)
                    os.remove(hdf_fpath)
            else:
                if os.path.exists(hdf_fpath):
                    print("Skip the file ", hdf_fpath, " since it is already exist")
                    continue

            output_h5_id = h5py.File(hdf_fpath, "w")
            print("output file save to ", hdf_fpath)
            face_model_load = np.loadtxt(face_model_path)
            output_cam_index = []

            output_landmarks = []
            output_face_patch = []

            output_left_eye_mask = []
            output_right_eye_mask = []
            output_head_mask = []

            output_latent_codes = []
            output_w2c_Rmat = []
            output_w2c_Tvec = []
            output_inmat = []
            output_c2w_Rmat = []
            output_c2w_Tvec = []
            output_inv_inmat = []

            output_pitchyaw_head_coordinate_system = []
            output_params_gaze_direction = []

            save_path = os.path.join(test_subject)
            cam_calibration_path = './data/gaze_capture/sfm_face_coordinates.npy'
            annotation_path = gazecapture_supplement[test_subject]

            (
                images,
                cam_indices,
                head_masks,
                landmarks,
                left_eye_masks,
                right_eye_masks,
            ) = main_process(
                dataset_name,
                f"{local_folder}",
                gpu_id,
                annotation_path,
                cam_calibration_path,
                face_model_path,
                log,
                save_path, 
            )
            if save_path is not None:
                nl_3dmm_path = save_path
            else:
                nl_3dmm_path = f"{local_folder}"
        
            surface_list_first_frame = surface_fitting( 
                gpu_id,
                log,
                nl_3dmm_path,
                img_size,
                intermediate_size,
                batch_size,
            )

            count_num_img = 100

            output_cam_index = output_h5_id.create_dataset(
                "cam_index",
                shape=(count_num_img, 1),
                dtype=np.uint8,
                chunks=(1, 1),
            )
            output_landmarks = output_h5_id.create_dataset(
                "facial_landmarks",
                shape=(count_num_img, 68, 2),
                dtype=np.float,
                chunks=(1, 68, 2),
            )
            output_face_patch = output_h5_id.create_dataset(
                "face_patch",
                shape=(count_num_img, 512, 512, 3),
                compression="lzf",
                dtype=np.uint8,
                chunks=(1, 512, 512, 3),
            )     
            output_left_eye_mask = output_h5_id.create_dataset(
                "left_eye_mask",
                shape=(count_num_img, 512, 512),
                compression="lzf",
                dtype=np.uint8,
                chunks=(1, 512, 512),
            )
            output_right_eye_mask = output_h5_id.create_dataset(
                "right_eye_mask",
                shape=(count_num_img, 512, 512),
                compression="lzf",
                dtype=np.uint8,
                chunks=(1, 512, 512),
            )
            output_head_mask = output_h5_id.create_dataset(
                "head_mask",
                shape=(count_num_img, 512, 512),
                compression="lzf",
                dtype=np.uint8,
                chunks=(1, 512, 512),
            )
            output_latent_codes = output_h5_id.create_dataset(
                "latent_codes",
                shape=(count_num_img, 306),
                compression="lzf",
                dtype=np.float,
                chunks=(1, 306),
            )
            output_w2c_Rmat = output_h5_id.create_dataset(
                "w2c_Rmat",
                shape=(count_num_img, 3, 3),
                compression="lzf",
                dtype=np.float,
                chunks=(1, 3, 3),
            )
            output_w2c_Tvec = output_h5_id.create_dataset(
                "w2c_Tvec",
                shape=(count_num_img, 3),
                compression="lzf",
                dtype=np.float,
                chunks=(1, 3),
            )
            output_inmat = output_h5_id.create_dataset(
                "inmat",
                shape=(count_num_img, 3, 3),
                compression="lzf",
                dtype=np.float,
                chunks=(1, 3, 3),
            )
            output_c2w_Rmat = output_h5_id.create_dataset(
                "c2w_Rmat",
                shape=(count_num_img, 3, 3),
                compression="lzf",
                dtype=np.float,
                chunks=(1, 3, 3),
            )
            output_c2w_Tvec = output_h5_id.create_dataset(
                "c2w_Tvec",
                shape=(count_num_img, 3),
                compression="lzf",
                dtype=np.float,
                chunks=(1, 3),
            )
            output_inv_inmat = output_h5_id.create_dataset(
                "inv_inmat",
                shape=(count_num_img, 3, 3),
                compression="lzf",
                dtype=np.float,
                chunks=(1, 3, 3),
            )
            output_pitchyaw_head_coordinate_system = output_h5_id.create_dataset(
                "pitchyaw_head",
                shape=(count_num_img, 2),
                compression="lzf",
                dtype=np.float,
                chunks=(1, 2),
            )
            output_params_gaze_direction = output_h5_id.create_dataset(
                "pitchyaw_3d",
                shape=(count_num_img, 3),
                compression="lzf",
                dtype=np.float,
                chunks=(1, 3),
            )
            output_face_head_pose = output_h5_id.create_dataset(
                "face_head_pose", 
                shape=(count_num_img, 2),
                compression="lzf",
                dtype=np.float, 
                chunks=(1, 2))

            for i in range(len(images)):
                output_cam_index[save_index] = cam_indices[i]
                output_landmarks[save_index] = landmarks[i]
                output_face_patch[save_index] = images[i]['image']
                output_left_eye_mask[save_index] = left_eye_masks[i]
                output_right_eye_mask[save_index] = right_eye_masks[i]
                output_head_mask[save_index] = head_masks[i] 
                output_latent_codes[save_index] = surface_list_first_frame[i]["code"]
                output_w2c_Rmat[save_index] = surface_list_first_frame[i]["w2c_Rmat"]
                output_w2c_Tvec[save_index] = surface_list_first_frame[i]["w2c_Tvec"]  
                output_inmat[save_index] = surface_list_first_frame[i]["inmat"]
                output_c2w_Rmat[save_index] = surface_list_first_frame[i]["c2w_Rmat"]
                output_c2w_Tvec[save_index] = surface_list_first_frame[i]["c2w_Tvec"]
                output_inv_inmat[save_index] = surface_list_first_frame[i]["inv_inmat"]
                landmark_use = [20, 23, 26, 29, 15, 19]
                face_model = face_model_load[landmark_use, :]
                output_face_head_pose[save_index] = calc_normalized_head_pose(face_model, images[i]['hr'], images[i]['ht'])
                gaze_direction_3d = images[i]['g']
                output_pitchyaw_head_coordinate_system[
                    save_index
                ] = convert_to_head_coordinate_system(
                    face_model_load,
                    gaze_direction_3d,
                    images[i]['hr'],
                    images[i]['ht'],
                )

                output_params_gaze_direction[save_index] = [
                    gaze_direction_3d[0,:][0],
                    gaze_direction_3d[1,:][0],
                    gaze_direction_3d[2,:][0],
                ]
                save_index += 1

            shutil.rmtree(save_path, ignore_errors=True)
            gazecapture_supplement.close()

            output_h5_id.close()
            print("close the h5 file")

            print("finish the subject: ", test_subject)
            elapsed_time = time.time() - start_time
            print("///////////////////////////////////")
            print(
                "Running time is {:02d}:{:02d}:{:02d}".format(
                    int(elapsed_time // 3600),
                    int(elapsed_time % 3600 // 60),
                    int(elapsed_time % 60),
                )
            )

            shutil.rmtree(save_path, ignore_errors=True)

    print("All Subjects were processed")

def dataset_processing(
    dataset_name,
    gpu_id,
    log,
    dir,
    dataset_url,
    dataset_path,
    annotation_path,
    cam_calibration_path,
    face_model_path,
    subject_start,
    subject_end,
    is_over_write,
    img_size,
    intermediate_size,
    batch_size,
):
    if dataset_name == "eth_xgaze":
        eth_xgaze_dataset_iteration_full(
            dataset_name,
            gpu_id,
            log,
            dir,
            dataset_url,
            dataset_path,
            annotation_path,
            cam_calibration_path,
            face_model_path,
            subject_start,
            subject_end,
            is_over_write,
            img_size,
            intermediate_size,
            batch_size,
        )
    elif dataset_name == "mpii_face":
        mpii_face_gaze_dataset_iteration_full(
            dataset_name,
            gpu_id,
            log,
            dir,
            dataset_path,
            annotation_path,
            cam_calibration_path,
            face_model_path,
            subject_start,
            subject_end,
            is_over_write,
            img_size,
            intermediate_size,
            batch_size,
        )
    elif dataset_name == "gaze_capture":
        gaze_capture_dataset_iteration_full(
            dataset_name,
            gpu_id,
            log,
            dir,
            dataset_path,
            annotation_path,
            cam_calibration_path,
            face_model_path,
            subject_start,
            subject_end,
            is_over_write,
            img_size,
            intermediate_size,
            batch_size,
        )
    elif dataset_name == "columbia":
        columbia_gaze_dataset_iteration_full(
            dataset_name,
            gpu_id,
            log,
            dir,
            dataset_path,
            annotation_path,
            cam_calibration_path,
            face_model_path,
            subject_start,
            subject_end,
            is_over_write,
            img_size,
            intermediate_size,
            batch_size,
        )
    else:
        print("Dataset not supported")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="The code to pre-process a specific dataset"
    )
    parser.add_argument("--log", type=bool, default=False)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="eth_xgaze",
        help="Accepted dataset: eth_xgaze, columbia, gaze_capture, mpii_face",
    )
    parser.add_argument(
        "--annotation_path", type=str, default="data/eth_xgaze/annotations"
    )
    parser.add_argument(
        "--cam_calibration_path", type=str, default="data/eth_xgaze/cam_original"
    )
    parser.add_argument(
        "-face_model_path", type=str, default="data/eth_xgaze/face_model.txt"
    )
    parser.add_argument(
        "--output_dir", type=str, default=""
    )
    parser.add_argument(
        "--dataset_url",
        type=str,
        default="",
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default=None,
    )
    parser.add_argument("--subject_start", type=int, default=1)  # min 0
    parser.add_argument("--subject_end", type=int, default=57)  # max 120
    parser.add_argument("--is_over_write", type=bool, default=True)
    parser.add_argument("--batch_size", type=int, default=18)
    parser.add_argument("--intermediate_size", type=int, default=256)
    parser.add_argument("--img_size", type=int, default=512)
    args = parser.parse_args()

    if args.log:
        wandb.init(project="pre_processing")

    dataset_processing(
        args.dataset_name,
        args.gpu_id,
        args.log,
        args.output_dir,
        args.dataset_url,
        args.dataset_dir,
        args.annotation_path,
        args.cam_calibration_path,
        args.face_model_path,
        args.subject_start,
        args.subject_end,
        args.is_over_write,
        args.img_size,
        args.intermediate_size,
        args.batch_size,
    )
