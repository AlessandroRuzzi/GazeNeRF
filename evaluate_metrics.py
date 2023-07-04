import argparse
import os
import random
from unittest.mock import CallableMixin

import cv2
import numpy as np
import torch
import scipy.io

import wandb
from gaze_estimation.xgaze_baseline_resnet import gaze_network
from gaze_estimation.orig_xgaze import gaze_network_orig
from gaze_estimation.gaze_estimator_resnet import GazeHeadResNet
from utils.metrics_utils import (evaluate_consistency,
                                 evaluate_personal_calibration,
                                 evaluate_input_target_images,
                                 evaluate_gaze_transfer)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train options")
    add_arg = parser.add_argument
    add_arg("--gpu_id", type=int, default=0)
    add_arg("--batch_size", type=int, default=1)
    add_arg("--num_workers", type=int, default=0)
    add_arg("--num_epochs", type=int, default=0)
    add_arg("--num_iterations", type=int, default=1)
    add_arg("--step_decay", type=int, default=1000)
    add_arg("--learning_rate", type=float, default=0.0001)
    add_arg("--vgg_importance", type=float, default=1.0)
    add_arg("--eye_loss_importance", type=float, default=200.0)
    add_arg("--num_images", type=int, default=[100,100,52,50])
    add_arg("--data_names", type=list, default=["eth_xgaze", "mpii_face_gaze", "columbia", "gaze_capture"])
    add_arg("--img_dir", type=list, default=["", "", "", ""])
    add_arg("--bg_type", type=str, default="white")
    add_arg("--checkpoint_dir", type=str, default=None)
    add_arg("--optimizer", type=str, default="adam")
    add_arg("--evaluation_type", type=str, default="input_target_images")
    add_arg("--state_dict_name", type=str, default="tmp.json")
    add_arg("--model_path", type=str, default="checkpoints/2_mlp.json")
    add_arg("--log", type=bool, default=False)
    add_arg("--resume", type=bool, default=True)
    add_arg("--verbose", type=bool, default=True)
    add_arg("--use_vgg_loss", type=bool, default=True)
    add_arg("--use_l1_loss", type=bool, default=True)
    add_arg("--use_angular_loss", type=bool, default=True)
    add_arg("--use_patch_gan_loss", type=bool, default=False)
    add_arg("--include_vd", type=bool, default=False)
    add_arg("--hier_sampling", type=bool, default=False)
    add_arg("--enable_ffhq", type=bool, default=False)
    add_arg("--enable_eth_xgaze", type=bool, default=True)
    add_arg("--fit_image", type=bool, default=False)
    return parser.parse_args()

def load_cams(args):
    cam_matrix = {}
    cam_distortion = {}
    cam_translation = {}
    cam_rotation = {}

    for name in args.data_names:
        cam_matrix[name] = []
        cam_distortion[name] = []
        cam_translation[name] = []
        cam_rotation[name] = []
    

    for cam_id in range(18):
        cam_file_name = "data/eth_xgaze/cam/cam" + str(cam_id).zfill(2) + ".xml"
        fs = cv2.FileStorage(cam_file_name, cv2.FILE_STORAGE_READ)
        cam_matrix["eth_xgaze"].append(fs.getNode("Camera_Matrix").mat())
        cam_distortion["eth_xgaze"].append(fs.getNode("Distortion_Coefficients").mat())
        cam_translation["eth_xgaze"].append(fs.getNode("cam_translation"))
        cam_rotation["eth_xgaze"].append(fs.getNode("cam_rotation"))
        fs.release()

    for i in range(15):
        file_name = os.path.join(
        "data/mpii_face_gaze/cam", "Camera" + str(i).zfill(2) + ".mat"
        )
        mat = scipy.io.loadmat(file_name)
        cam_matrix["mpii_face_gaze"].append(mat.get("cameraMatrix"))
        cam_distortion["mpii_face_gaze"].append(mat.get(
            "distCoeffs"
        ))

    cam_file_name = "data/columbia/cam/cam" + str(0).zfill(2) + ".xml"
    fs = cv2.FileStorage(cam_file_name, cv2.FILE_STORAGE_READ)
    cam_matrix["columbia"] = fs.getNode("Camera_Matrix").mat()
    cam_distortion["columbia"] = fs.getNode("Distortion_Coefficients").mat()

    cam_file_name = "data/gaze_capture/cam/cam" + str(0).zfill(2) + ".xml"
    fs = cv2.FileStorage(cam_file_name, cv2.FILE_STORAGE_READ)
    cam_matrix["gaze_capture"] = fs.getNode("Camera_Matrix").mat()
    cam_distortion["gaze_capture"] = fs.getNode("Distortion_Coefficients").mat()

    return cam_matrix,cam_distortion, cam_translation, cam_rotation


def main():

    torch.manual_seed(45)  # cpu
    torch.cuda.manual_seed(55)  # gpu
    np.random.seed(65)  # numpy
    random.seed(75)  # random and transforms
    torch.backends.cudnn.deterministic = True  # cudnn
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    torch.set_num_threads(1)

    img_dim = 224

    if torch.cuda.is_available():
        device = torch.device("cuda:%s" % 0)
    else:
        device = "cpu"

    # Initialization
    args = parse_args()

    if args.log:
        wandb.init(project="metric evalaution", config={"gpu_id": 0})
        wandb.config.update(args)

    path = "configs/config_models/epoch_7_resnet_correct_ckpt.pth.tar"
    model = gaze_network().to(device)

    state_dict = torch.load(path, map_location=torch.device("cpu"))
    model.load_state_dict(state_dict=state_dict["model_state"])
    model.eval()

    print("Done")

    face_model_load = np.loadtxt("data/eth_xgaze/face_model.txt")
    cam_matrix, cam_distortion, cam_translation, cam_rotation = load_cams(args)
    

    # load mirror calibration information
    fs = cv2.FileStorage(
        "data/eth_xgaze/cam/mirror_position.xml", cv2.FILE_STORAGE_READ
    )
    pix2mm = fs.getNode("pix2mm").mat()
    screen_translation = fs.getNode("screen_translation").mat()
    screen_rotation = fs.getNode("screen_rotation").mat()
    fs.release()

    if args.evaluation_type == "input_target_images":
        evaluate_input_target_images(
            device,
            args,
            model,
            cam_matrix,
            cam_distortion,
            face_model_load,
            img_dim,
            args.evaluation_type,
        )
    elif args.evaluation_type == "personal_calibration":
        for i in range(20, 101, 10):
            args.num_images = i
            args.num_epochs = 100
            args.num_iterations = 1
            evaluate_personal_calibration(
                device,
                args,
                model,
                cam_matrix,
                cam_distortion,
                face_model_load,
                img_dim,
                args.evaluation_type,
                pix2mm,
                screen_translation,
                screen_rotation,
                cam_translation,
                cam_rotation,
            )
    elif args.evaluation_type == "consistency":
                args.num_iterations = 1
                evaluate_consistency(
                    device,
                    args,
                    model,
                    cam_matrix,
                    cam_distortion,
                    face_model_load,
                    img_dim,
                    args.evaluation_type,
                )
    elif args.evaluation_type == "gaze_transfer":
                args.num_epochs = 100
                evaluate_gaze_transfer(
                    device,
                    args,
                    model,
                    cam_matrix,
                    cam_distortion,
                    face_model_load,
                    img_dim,
                    args.evaluation_type,
                )
    else:
        print("Wrong evaluation type")


if __name__ == "__main__":
    main()
