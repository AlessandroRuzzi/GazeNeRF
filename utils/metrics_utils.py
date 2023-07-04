import csv
import imp
import json
from math import perm
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from piq import DISTS, LPIPS, psnr, ssim, FID
from sklearn.linear_model import Ridge
from torchvision import transforms
import random
import h5py

import wandb
from datasets.eth_xgaze import get_train_loader
from datasets.eth_xgaze import get_val_loader as xgaze_get_val_loader
from datasets.mpii_face_gaze import get_val_loader as mpii_get_val_loader
from datasets.columbia import get_val_loader as columbia_get_val_loader
from datasets.gaze_capture import get_val_loader as gaze_capture_get_val_loader
from trainer.gazenerf_trainer import get_trainer
from utils.gaze_estimation_utils import normalize
from utils.logging import log_evaluation_image, log_one_subject_evaluation_results, log_all_datasets_evaluation_results
from datasets.standard_image_dataset import get_data_loader as image_get_data_loader
from face_recognition.evaluation_similarity import evaluation_similarity

trans = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.ToTensor(),  # this also convert pixel value from [0,255] to [0,1]
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.Resize(size=(224, 224)),
    ]
)

trans_resize = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.ToTensor(),  # this also convert pixel value from [0,255] to [0,1]
        #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.Resize(size=(224, 224)),
    ]
)

subjects_gaze_direction = []


def draw_arrow(batch_images, pitchyaw):
    """Draw gaze angle on given image with a given eye positions."""
    pos = [256, 256]
    length = 40.0
    thickness = 2
    color = (0, 0, 255)
    image_out = (batch_images.detach().cpu().permute(0, 2, 3, 1).numpy() * 255).astype(
        np.uint8
    )[0]
    image_out = Image.fromarray(image_out)
    image_out = np.array(image_out)
    if len(image_out.shape) == 2 or image_out.shape[2] == 1:
        image_out = cv2.cvtColor(image_out, cv2.COLOR_GRAY2BGR)
    dx = -length * np.sin(pitchyaw[1]) * np.cos(pitchyaw[0])
    dy = -length * np.sin(pitchyaw[0])
    cv2.arrowedLine(
        image_out,
        tuple(np.round(pos).astype(np.int32)),
        tuple(np.round([pos[0] + dx, pos[1] + dy]).astype(int)),
        color,
        thickness,
        cv2.LINE_AA,
        tipLength=0.2,
    )
    img = Image.fromarray(image_out)
    img.show()


def nn_angular_distance(a, b):
    sim = F.cosine_similarity(a, b, eps=1e-6)
    sim = F.hardtanh(sim, -1.0, 1.0)
    return torch.acos(sim) * (180 / np.pi)


def pitchyaw_to_vector(pitchyaws):
    sin = torch.sin(pitchyaws)
    cos = torch.cos(pitchyaws)
    return torch.stack([cos[:, 0] * sin[:, 1], sin[:, 0], cos[:, 0] * cos[:, 1]], 1)


def gaze_angular_loss(y, y_hat):
    y = pitchyaw_to_vector(y)
    y_hat = pitchyaw_to_vector(y_hat)
    loss = nn_angular_distance(y, y_hat)
    return torch.mean(loss)


def variance_of_laplacian(image):
    return cv2.Laplacian(image, cv2.CV_64F).var()

def calculate_FID(gt_images, pred_images):
    first_dl, second_dl = image_get_data_loader(gt_images), image_get_data_loader(pred_images)
    fid_metric = FID()
    first_feats = fid_metric.compute_feats(first_dl)
    second_feats = fid_metric.compute_feats(second_dl)
    fid: torch.Tensor = fid_metric(first_feats, second_feats)
    return fid


def select_dataloader(name, subject, idx, img_dir, batch_size, num_images, num_workers, is_shuffle, evaluate):
    if name == "eth_xgaze":
        return (name, subject, idx, xgaze_get_val_loader(data_dir=img_dir, batch_size=batch_size, num_val_images= num_images, num_workers= num_workers, is_shuffle= is_shuffle, subject=subject, evaluate=evaluate))
    elif name == "mpii_face_gaze":
        return (name, subject, idx, mpii_get_val_loader(data_dir=img_dir, batch_size=batch_size, num_val_images= num_images, num_workers= num_workers, is_shuffle= is_shuffle, subject=subject, evaluate=evaluate))
    elif name == "columbia":
        return (name, subject, idx, columbia_get_val_loader(data_dir=img_dir, batch_size=batch_size, num_val_images= num_images, num_workers= num_workers, is_shuffle= is_shuffle, subject=subject, evaluate=evaluate))
    elif name == "gaze_capture":
        return (name, subject, idx, gaze_capture_get_val_loader(data_dir=img_dir, batch_size=batch_size, num_val_images= num_images, num_workers= num_workers, is_shuffle= is_shuffle, subject=subject, evaluate=evaluate))
    else:
        print("Dataset not supported")

def select_cam_matrix(name,cam_matrix,cam_distortion,cam_ind, subject):
    if name == "eth_xgaze":
        return cam_matrix[name][cam_ind], cam_distortion[name][cam_ind]
    elif name == "mpii_face_gaze":
        camera_matrix = cam_matrix[name][int(subject[-5:-3])]
        camera_matrix[0, 2] = 256.0
        camera_matrix[1, 2] = 256.0
        return camera_matrix, cam_distortion[name][int(subject[-5:-3])]
    elif name == "columbia":
        return cam_matrix[name], cam_distortion[name]
    elif name == "gaze_capture":
        return cam_matrix[name], cam_distortion[name]
    else:
        print("Dataset not supported")

def evaluate_input_target_images(
    device, args, model, cam_matrix, cam_distortion, face_model_load, img_dim, method
):

    val_keys = {}
    for name in args.data_names:
        file_path = os.path.join("data", name, "train_test_split.json")
        with open(file_path, "r") as f:
            datastore = json.load(f)
        val_keys[name] = datastore["val"]

    dataloader_all = []

    for idx,name in enumerate(args.data_names):
        for subject in val_keys[name]:
            dataloader_all.append(select_dataloader(name, subject, idx, args.img_dir[idx], args.batch_size, args.num_images[idx], args.num_workers, is_shuffle=False, evaluate="target"))   

    dict_angular_loss = {}
    dict_angular_head_loss = {}
    dict_ssim_loss = {}
    dict_psnr_loss = {}
    dict_lpips_loss = {}
    dict_l1_loss = {}
    dict_num_images = {}

    dict_similarity = {}

    dict_fid = {}
    dict_gt_images = {}
    dict_pred_images = {}
    full_images_gt_list = []
    full_images_pred_list = []

    for name in args.data_names:
        dict_angular_loss[name] = 0.0
        dict_angular_head_loss[name] = 0.0
        dict_ssim_loss[name] = 0.0
        dict_psnr_loss[name] = 0.0
        dict_lpips_loss[name] = 0.0
        dict_l1_loss[name] = 0.0
        dict_num_images[name] = 0

        dict_similarity[name] = 0.0

        dict_fid[name] = 0.0
        dict_gt_images[name] = []
        dict_pred_images[name] = []

    for name, subject, index_dataset, dataloader in dataloader_all:

        angular_loss = 0.0
        angular_head_loss = 0.0
        ssim_loss = 0.0
        psnr_loss = 0.0
        lpips_loss = 0.0
        l1_loss = 0.0
        num_images = 0

        similarity = 0.0

        fid = 0.0
        gt_list = []
        pred_list = []

        for i, (
            batch_images_1,
            batch_head_mask_1,
            batch_left_eye_mask_1,
            batch_right_eye_mask_1,
            batch_nl3dmm_para_dict_1,
            ldms_1,
            cam_ind_1,
            idx_1,
            key_1,
            batch_images_2,
            batch_head_mask_2,
            batch_left_eye_mask_2,
            batch_right_eye_mask_2,
            batch_nl3dmm_para_dict_2,
            ldms_2,
            cam_ind_2,
            idx_2,
            key_2,
        ) in enumerate(dataloader):
            trainer = get_trainer(
                checkpoint_dir=args.checkpoint_dir,
                batch_size=args.batch_size,
                gpu=args.gpu_id,
                resume=args.resume,
                include_vd=args.include_vd,
                hier_sampling=args.hier_sampling,
                log=args.log,
                lr=args.learning_rate,
                num_iter=args.num_iterations,
                optimizer=args.optimizer,
                step_decay=args.step_decay,
                vgg_importance=args.vgg_importance,
                eye_loss_importance=args.eye_loss_importance,
                fit_image=args.fit_image,
                model_path=args.model_path,
                state_dict_name=args.state_dict_name,
                use_vgg_loss=args.use_vgg_loss,
                use_l1_loss=args.use_l1_loss,
                use_angular_loss=args.use_angular_loss,
                use_patch_gan_loss=args.use_patch_gan_loss,
            )
            _, _, _, dataloader_tmp = select_dataloader(name, subject, index_dataset, args.img_dir[index_dataset], args.batch_size, args.num_images[index_dataset], args.num_workers, is_shuffle=False, evaluate="target")

            trainer.train_single_image(dataloader_tmp, args.num_epochs, i, method)

            
            ldms = ldms_2[0]
            batch_head_mask = torch.reshape(batch_head_mask_2, (1, 1, 512, 512))
            batch_images = batch_images_2
            cam_ind = cam_ind_2
            batch_left_eye_mask = batch_left_eye_mask_2
            batch_right_eye_mask = batch_right_eye_mask_2
            batch_nl3dmm_para_dict = batch_nl3dmm_para_dict_2

            camera_matrix, camera_distortion = select_cam_matrix(name, cam_matrix,cam_distortion, cam_ind, subject)

            nonhead_mask = batch_head_mask < 0.5
            nonhead_mask_c3b = nonhead_mask.expand(-1, 3, -1, -1)
            batch_images[nonhead_mask_c3b] = 1.0

            target_image_quality = torch.reshape(
                batch_images , (1, 3, 512, 512)
            ).to(device)
    
            batch_images_norm = normalize(
                (batch_images.detach().cpu().permute(0, 2, 3, 1).numpy() * 255).astype(
                    np.uint8
                )[0],
                camera_matrix,
                camera_distortion,
                face_model_load,
                ldms,
                img_dim,
            )

            target_normalized_log = batch_images_norm
            batch_images_norm = torch.reshape(
                trans(batch_images_norm), (1, 3, img_dim, img_dim)
            ).to(
                device
            )  
            pitchyaw_gt, head_gt = model(batch_images_norm)

            pred = trainer.predict_single_image(
                0,
                dataloader,
                batch_images,
                batch_head_mask,
                batch_left_eye_mask,
                batch_right_eye_mask,
                batch_nl3dmm_para_dict,
            )

            pred = torch.nan_to_num(pred, nan=1.0)

            pred_image_quality = torch.reshape(
                pred , (1, 3, 512, 512)
            ).to(device)

            batch_images_norm_pre = normalize(
                (pred.detach().cpu().permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)[0],
                camera_matrix,
                camera_distortion,
                face_model_load,
                ldms,
                img_dim,
            )
            pred_normalized_log = batch_images_norm_pre
            batch_images_norm_pre = torch.reshape(
                trans(batch_images_norm_pre), (1, 3, img_dim, img_dim)
            ).to(device)
            pitchyaw_gen, head_gen = model(batch_images_norm_pre)

            loss = gaze_angular_loss(pitchyaw_gt, pitchyaw_gen).detach().cpu().numpy()
            angular_loss += loss
            num_images += 1
            dict_angular_loss[name] += loss
            dict_num_images[name] += 1
            print("Gaze Angular Error: ", angular_loss / num_images, loss, num_images)

            loss = gaze_angular_loss(head_gt, head_gen).detach().cpu().numpy()
            angular_head_loss += loss
            dict_angular_head_loss[name] += loss
            print("Head Angular Error: ", angular_head_loss / num_images, loss, num_images)

            sim_gt = ( torch.reshape(
                trans_resize(batch_images[0,:]) , (1, 3, img_dim, img_dim)
            ).to(device).detach().cpu().permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)[0]

            sim_gen = ( torch.reshape(
                trans_resize(pred[0,:]) , (1, 3, img_dim, img_dim)
            ).to(device).detach().cpu().permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)[0]
            try:
                loss = evaluation_similarity(sim_gt, sim_gen)
            except:
                loss = -0.1
            similarity += loss
            dict_similarity[name] += loss
            print("Similarity Score: ", similarity / num_images, loss, num_images)

            gt_list.append(target_image_quality[0,:])
            pred_list.append(pred_image_quality[0,:])

            dict_gt_images[name].append(target_image_quality[0,:])
            dict_pred_images[name].append(pred_image_quality[0,:])

            full_images_gt_list.append(target_image_quality[0,:])
            full_images_pred_list.append(pred_image_quality[0,:])
            loss = (
                ssim(target_image_quality, pred_image_quality, data_range=1.0)
                .detach()
                .cpu()
                .numpy()
            )
            ssim_loss += loss
            dict_ssim_loss[name] += loss
            print("SSIM: ", ssim_loss / num_images, loss, num_images)

            loss = (
                psnr(target_image_quality, pred_image_quality, data_range=1.0)
                .detach()
                .cpu()
                .numpy()
            )
            psnr_loss += loss
            dict_psnr_loss[name] += loss
            print("PSNR: ", psnr_loss / num_images, loss, num_images)

            lpips_metric = LPIPS()
            loss = lpips_metric(target_image_quality, pred_image_quality).detach().cpu().numpy()
            lpips_loss += loss
            dict_lpips_loss[name] += loss
            print("LPIPS: ", lpips_loss / num_images, loss, num_images)

            loss = (
                torch.nn.functional.l1_loss(target_image_quality, pred_image_quality)
                .detach()
                .cpu()
                .numpy()
            )
            l1_loss += loss
            dict_l1_loss[name] += loss
            print("L1 Distance: ", l1_loss / num_images, loss, num_images)


            if args.log:
                log_evaluation_image(pred_normalized_log, target_normalized_log, batch_images_1, batch_images_2, pred)
        
        fid = calculate_FID(gt_images= gt_list, pred_images= pred_list)
        
        if args.log:    
            log_one_subject_evaluation_results(angular_loss, angular_head_loss, ssim_loss, psnr_loss, lpips_loss,
                                                l1_loss, num_images, fid, similarity)
    for name in args.data_names:
        dict_fid[name]  = calculate_FID(gt_images= dict_gt_images[name], pred_images= dict_pred_images[name])
        
    full_fid = calculate_FID(gt_images= full_images_gt_list, pred_images= full_images_pred_list)

    if args.log:
        log_all_datasets_evaluation_results(args.data_names, dict_angular_loss, dict_angular_head_loss, dict_ssim_loss, dict_psnr_loss, dict_lpips_loss,
                                                dict_l1_loss, dict_num_images, dict_fid, full_fid, dict_similarity)




def evaluate_personal_calibration(
    device,
    args,
    model,
    cam_matrix,
    cam_distortion,
    face_model_load,
    img_dim,
    method,
    pix2mm,
    screen_translation,
    screen_rotation,
    cam_translation,
    cam_rotation,
):
    refer_list_file = os.path.join("data/eth_xgaze", "train_test_split.json")

    with open(refer_list_file, "r") as f:
        datastore = json.load(f)

    val_keys = datastore["val_gaze"]

    for t, subject in enumerate(val_keys):

        for iter in range(args.num_iterations):

            train_dataloader = xgaze_get_val_loader(data_dir=args.img_dir[0], batch_size=1, num_val_images=200, num_workers= 0, is_shuffle= False, subject=subject, evaluate='landmark')

            predicted_images = []
            fit_iterations_counter = 0

            random_fit_images_num = []
            for i in range(args.num_images):
                random_fit_images_num.append(random.randint(0,199))
            

            save_path = "/local/home/aruzzi/personal_calibration_files_4/" 
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            hdf_path = save_path + subject[:-3] + "_nsample_" +  str(args.num_images) + "_iter_" + str(iter) + ".h5"
            output_h5_id = h5py.File(hdf_path, "w")

            output_face_patch = []


            trainer = get_trainer(
                checkpoint_dir=args.checkpoint_dir,
                batch_size=args.batch_size,
                gpu=0,
                resume=args.resume,
                include_vd=args.include_vd,
                hier_sampling=args.hier_sampling,
                log=args.log,
                lr=args.learning_rate,
                num_iter=args.num_iterations,
                optimizer=args.optimizer,
                step_decay=args.step_decay,
                vgg_importance=args.vgg_importance,
                eye_loss_importance=args.eye_loss_importance,
                fit_image=args.fit_image,
                model_path=args.model_path,
                state_dict_name=args.state_dict_name,
                use_vgg_loss=args.use_vgg_loss,
                use_l1_loss=args.use_l1_loss,
                use_angular_loss=args.use_angular_loss,
                use_patch_gan_loss=args.use_patch_gan_loss,
            )

            trainer.net.train()
            trainer.prepare_optimizer_opt()

            ## Fit GazeNeRF on random images

            while (fit_iterations_counter < args.num_epochs) : 
                for i, (
                    batch_images,
                    batch_head_mask,
                    batch_left_eye_mask,
                    batch_right_eye_mask,
                    batch_nl3dmm_para_dict,
                    ldms,
                    cam_ind,
                ) in enumerate(train_dataloader):
                    if fit_iterations_counter == args.num_epochs:
                        break
                    if i in random_fit_images_num:
                        trainer.prepare_data(
                                batch_images,
                                batch_head_mask,
                                batch_left_eye_mask,
                                batch_right_eye_mask,
                                batch_nl3dmm_para_dict,
                            )
                        loss_dict = trainer.perform_fitting(i, cam_ind, ldms, 1)
                        del loss_dict
                        fit_iterations_counter+=1


            ## predict images to fine tune the pre traind gaze estimator

            if not output_face_patch:
                output_face_patch = output_h5_id.create_dataset(
                "face_patch",
                shape=(200, 224, 224, 3),
                compression="lzf",
                dtype=np.uint8,
                chunks=(1, 224, 224, 3),
            )     

            counter_save_index = 0
               
            for i, (
                batch_images,
                batch_head_mask,
                batch_left_eye_mask,
                batch_right_eye_mask,
                batch_nl3dmm_para_dict,
                ldms,
                cam_ind,
            ) in enumerate(train_dataloader):
                camera_matrix, camera_distortion = select_cam_matrix("eth_xgaze", cam_matrix,cam_distortion, cam_ind, subject)

                ldms = ldms[0]
                if i not in random_fit_images_num:

                    pred = trainer.predict_single_image(
                        0,
                        train_dataloader,
                        batch_images,
                        batch_head_mask,
                        batch_left_eye_mask,
                        batch_right_eye_mask,
                        batch_nl3dmm_para_dict,
                    )
                    batch_images_norm_pre = normalize(
                        (pred.detach().cpu().permute(0, 2, 3, 1).numpy() * 255).astype(
                            np.uint8
                        )[0],
                        camera_matrix,
                        camera_distortion,
                        face_model_load,
                        ldms,
                        img_dim,
                    )

                    image = cv2.cvtColor(batch_images_norm_pre, cv2.COLOR_RGB2BGR)

                    output_face_patch[counter_save_index] = image
                    counter_save_index +=1

                    if args.log and i%10 == 0:
                        res_img = np.concatenate(
                            [
                                batch_images_norm_pre.reshape(1, 224, 224, 3),
                            ],
                            axis=2,
                        )
                        img = Image.fromarray(res_img[0])
                        log_image = wandb.Image(img)
                        wandb.log(
                            {" Target Normalized | Prediction Normalized ": log_image}
                        )
                else:
                    batch_head_mask = torch.reshape(batch_head_mask, (1, 1, 512, 512))
                    nonhead_mask = batch_head_mask < 0.5
                    nonhead_mask_c3b = nonhead_mask.expand(-1, 3, -1, -1)
                    batch_images[nonhead_mask_c3b] = 1.0
                    batch_images_norm = normalize(
                        (batch_images.detach().cpu().permute(0, 2, 3, 1).numpy() * 255).astype(
                            np.uint8
                        )[0],
                        camera_matrix,
                        camera_distortion,
                        face_model_load,
                        ldms,
                        img_dim,
                    )

                    image = cv2.cvtColor(batch_images_norm, cv2.COLOR_RGB2BGR)

                    output_face_patch[counter_save_index] = image
                    counter_save_index +=1


            output_h5_id.close()


def evaluate_consistency(
    device,
    args,
    model,
    cam_matrix,
    cam_distortion,
    face_model_load,
    img_dim,
    method,
):

    refer_list_file = os.path.join("data/eth_xgaze", "train_test_split.json")

    with open(refer_list_file, "r") as f:
        datastore = json.load(f)

    val_keys = datastore["val_gaze"]

    subj_gaze_dict = {}
    subj_gaze_dict_not_modified = {}

    for subject in val_keys:
        subj_gaze_dict[subject] = 0.0
        subj_gaze_dict_not_modified[subject] = 0.0

    for t, subject in enumerate(val_keys):
        for iter in range(args.num_iterations):
            train_dataloader = xgaze_get_val_loader(data_dir=args.img_dir[0], batch_size=1, num_val_images=200, num_workers= 0, is_shuffle= False, subject=subject, evaluate='landmark')

            
            random_fit_images_num = random.randint(0,199)

            trainer = get_trainer(
                checkpoint_dir=args.checkpoint_dir,
                batch_size=args.batch_size,
                gpu=0,
                resume=args.resume,
                include_vd=args.include_vd,
                hier_sampling=args.hier_sampling,
                log=args.log,
                lr=args.learning_rate,
                num_iter=args.num_iterations,
                optimizer=args.optimizer,
                step_decay=args.step_decay,
                vgg_importance=args.vgg_importance,
                eye_loss_importance=args.eye_loss_importance,
                fit_image=args.fit_image,
                model_path=args.model_path,
                state_dict_name=args.state_dict_name,
                use_vgg_loss=args.use_vgg_loss,
                use_l1_loss=args.use_l1_loss,
                use_angular_loss=args.use_angular_loss,
                use_patch_gan_loss=args.use_patch_gan_loss,
            )

            trainer.train_single_image(train_dataloader, args.num_epochs, random_fit_images_num, method)

            eval_dataloader = xgaze_get_val_loader(data_dir=args.img_dir[0], batch_size=1, num_val_images=200, num_workers= 0, is_shuffle= False, subject=subject, evaluate='landmark')

            gaze_labels_predict = []

            for i, (
                batch_images,
                batch_head_mask,
                batch_left_eye_mask,
                batch_right_eye_mask,
                batch_nl3dmm_para_dict,
                ldms,
                cam_ind,
            ) in enumerate(eval_dataloader):
                if i != random_fit_images_num:

                    camera_matrix, camera_distortion = select_cam_matrix("eth_xgaze", cam_matrix,cam_distortion, cam_ind, subject)

                    ldms = ldms[0]

                    pred = trainer.predict_single_image(
                        0,
                        train_dataloader,
                        batch_images,
                        batch_head_mask,
                        batch_left_eye_mask,
                        batch_right_eye_mask,
                        batch_nl3dmm_para_dict,
                    )
                    batch_images_norm_pre = normalize(
                        (pred.detach().cpu().permute(0, 2, 3, 1).numpy() * 255).astype(
                            np.uint8
                        )[0],
                        camera_matrix,
                        camera_distortion,
                        face_model_load,
                        ldms,
                        img_dim,
                    )
                    batch_images_norm_pre = torch.reshape(
                        trans(batch_images_norm_pre), (1, 3, img_dim, img_dim)
                    ).to(device)

                    pitchyaw_gen, head_gen = model(batch_images_norm_pre)

                    gaze_labels_predict.append(pitchyaw_gen.detach().cpu())
                
            gaze_labels_modified = []

            for i, (
                batch_images,
                batch_head_mask,
                batch_left_eye_mask,
                batch_right_eye_mask,
                batch_nl3dmm_para_dict,
                ldms,
                cam_ind,
            ) in enumerate(eval_dataloader):
                if i != random_fit_images_num:

                    camera_matrix, camera_distortion = select_cam_matrix("eth_xgaze", cam_matrix,cam_distortion, cam_ind, subject)

                    ldms = ldms[0]

                    #0.349 # 20 degrees  #0.262 # 15 degrees   #0.175 # 10 degrees   #0.087 # 5 degrees
                    batch_nl3dmm_para_dict["pitchyaw"][:,0] += 0.087

                    pred = trainer.predict_single_image(
                        0,
                        train_dataloader,
                        batch_images,
                        batch_head_mask,
                        batch_left_eye_mask,
                        batch_right_eye_mask,
                        batch_nl3dmm_para_dict,
                    )
                    batch_images_norm_pre = normalize(
                        (pred.detach().cpu().permute(0, 2, 3, 1).numpy() * 255).astype(
                            np.uint8
                        )[0],
                        camera_matrix,
                        camera_distortion,
                        face_model_load,
                        ldms,
                        img_dim,
                    )
                    batch_images_norm_pre = torch.reshape(
                        trans(batch_images_norm_pre), (1, 3, img_dim, img_dim)
                    ).to(device)

                    pitchyaw_mod, head_mod = model(batch_images_norm_pre)
                    gaze_labels_modified.append(pitchyaw_mod.detach().cpu())

            loss = 0.0
            loss_not_modified = 0.0
            for i in range(len(gaze_labels_predict)):
                loss +=  abs(gaze_labels_predict[i][0,0] - gaze_labels_modified[i][0,0])  # How to calculate the angular loss only of the pitch?
                loss_not_modified +=  abs(gaze_labels_predict[i][0,1] - gaze_labels_modified[i][0,1])
            loss /= len(gaze_labels_modified)
            loss_not_modified /= len(gaze_labels_modified)
            subj_gaze_dict[subject] += loss
            subj_gaze_dict_not_modified[subject] += loss_not_modified

        subj_gaze_dict[subject] /= args.num_iterations
        subj_gaze_dict_not_modified[subject] /= args.num_iterations

    loss = 0.0
    loss_not_modified = 0.0
    for subject in val_keys:
        print(subj_gaze_dict[subject])
        loss += subj_gaze_dict[subject]
        print(subj_gaze_dict_not_modified[subject])
        loss_not_modified +=subj_gaze_dict_not_modified[subject]

    print("Final results:")
    print(loss/15.0)
    print(loss_not_modified/15.0)

def evaluate_gaze_transfer(
    device,
    args,
    model,
    cam_matrix,
    cam_distortion,
    face_model_load,
    img_dim,
    method,
):
    refer_list_file = os.path.join("data/eth_xgaze", "train_test_split.json")

    with open(refer_list_file, "r") as f:
        datastore = json.load(f)
    val_keys = datastore["val_gaze"]

    dataloader_all = []

    for subject in val_keys:
        dataloader_all.append(xgaze_get_val_loader(data_dir=args.img_dir[0], batch_size=1, num_val_images=200, num_workers= 0, is_shuffle= False, subject=subject, evaluate='landmark'))


    for index, dataloader in enumerate(dataloader_all):
        if index == len(dataloader_all) -1:
            break
        for i, (
                batch_images,
                batch_head_mask,
                batch_left_eye_mask,
                batch_right_eye_mask,
                batch_nl3dmm_para_dict,
                ldms,
                cam_ind,
            ) in enumerate(dataloader):
                if cam_ind == 0:
                    trainer = get_trainer(
                        checkpoint_dir=args.checkpoint_dir,
                        batch_size=args.batch_size,
                        gpu=0,
                        resume=args.resume,
                        include_vd=args.include_vd,
                        hier_sampling=args.hier_sampling,
                        log=args.log,
                        lr=args.learning_rate,
                        num_iter=args.num_iterations,
                        optimizer=args.optimizer,
                        step_decay=args.step_decay,
                        vgg_importance=args.vgg_importance,
                        eye_loss_importance=args.eye_loss_importance,
                        fit_image=args.fit_image,
                        model_path=args.model_path,
                        state_dict_name=args.state_dict_name,
                        use_vgg_loss=args.use_vgg_loss,
                        use_l1_loss=args.use_l1_loss,
                        use_angular_loss=args.use_angular_loss,
                        use_patch_gan_loss=args.use_patch_gan_loss,
                    )
                    gaze_direction = trainer.optimize_gaze_direction(dataloader= xgaze_get_val_loader(data_dir=args.img_dir[0], batch_size=1, num_val_images=200, num_workers= 0, is_shuffle= False, subject=val_keys[index], evaluate='landmark'), n_epochs=args.num_epochs, index= i, method= method)
                    for j, (
                        batch_images_target,
                        batch_head_mask_target,
                        batch_left_eye_mask_target,
                        batch_right_eye_mask_target,
                        batch_nl3dmm_para_dict_target,
                        ldms_target,
                        cam_ind_target,
                    ) in enumerate(dataloader_all[index +1]):
                        if cam_ind_target == 0:
                            trainer_target = get_trainer(
                                checkpoint_dir=args.checkpoint_dir,
                                batch_size=args.batch_size,
                                gpu=0,
                                resume=args.resume,
                                include_vd=args.include_vd,
                                hier_sampling=args.hier_sampling,
                                log=args.log,
                                lr=args.learning_rate,
                                num_iter=args.num_iterations,
                                optimizer=args.optimizer,
                                step_decay=args.step_decay,
                                vgg_importance=args.vgg_importance,
                                eye_loss_importance=args.eye_loss_importance,
                                fit_image=args.fit_image,
                                model_path=args.model_path,
                                state_dict_name=args.state_dict_name,
                                use_vgg_loss=args.use_vgg_loss,
                                use_l1_loss=args.use_l1_loss,
                                use_angular_loss=args.use_angular_loss,
                                use_patch_gan_loss=args.use_patch_gan_loss,
                            )
                            trainer_target.train_single_image(xgaze_get_val_loader(data_dir=args.img_dir[0], batch_size=1, num_val_images=200, num_workers= 0, is_shuffle= False, subject=val_keys[index+1], evaluate='landmark'), 40, j, method)
                            batch_nl3dmm_para_dict_target["pitchyaw"] = gaze_direction
                            pred = trainer_target.predict_single_image(
                                            j,
                                            xgaze_get_val_loader(data_dir=args.img_dir[0], batch_size=1, num_val_images=200, num_workers= 0, is_shuffle= False, subject=val_keys[index+1], evaluate='landmark'),
                                            batch_images_target,
                                            batch_head_mask_target,
                                            batch_left_eye_mask_target,
                                            batch_right_eye_mask_target,
                                            batch_nl3dmm_para_dict_target,
                                        )

                            if args.log :
                                res_img = np.concatenate(
                                            [
                                                (batch_images.detach().cpu().permute(0, 2, 3, 1).numpy() * 255
                                                        ).astype(np.uint8),
                                                (batch_images_target.detach().cpu().permute(0, 2, 3, 1).numpy() * 255
                                                        ).astype(np.uint8),
                                                (pred.detach().cpu().permute(0, 2, 3, 1).numpy() * 255
                                                        ).astype(np.uint8),
                                            ],
                                            axis=2,
                                        )       
                                img = Image.fromarray(res_img[0])
                                log_image = wandb.Image(img)
                                wandb.log(
                                    {" Target Normalized | Prediction Normalized ": log_image}
                                )
                            break

                            
