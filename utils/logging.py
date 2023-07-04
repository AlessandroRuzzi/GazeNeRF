import logging
import sys
import wandb
import h5py
import cv2
import numpy as np
from PIL import Image


def config_logging(verbose, log_file=None, append=False):
    log_format = "%(asctime)s %(levelname)s %(message)s"
    log_level = logging.DEBUG if verbose else logging.INFO
    stream_handler = logging.StreamHandler(stream=sys.stdout)
    stream_handler.setLevel(log_level)
    handlers = [stream_handler]
    if log_file is not None:

        file_mode = "a" if append else "w"
        file_handler = logging.FileHandler(log_file, mode=file_mode)
        file_handler.setLevel(log_level)
        handlers.append(file_handler)
    logging.basicConfig(level=log_level, format=log_format, handlers=handlers)

def log_one_number(number, description):
    wandb.log({description: number})

def log_mask(img_to_log, mask, description, class_labels):
    image_gt = wandb.Image(img_to_log, caption="Image")
    mask_img = wandb.Image(
                    image_gt,
                    masks={
                        "predictions": {
                            "mask_data": mask,
                            "class_labels": class_labels,
                        }
                    },
                )
    wandb.log({description: mask_img})


def log_all_images(img_tensor, pred_dict):
    gt_img = (
            img_tensor.detach().cpu()
            .permute(0, 2, 3, 1).numpy()
            * 255
        ).astype(np.uint8)
    coarse_fg_rgb = pred_dict["coarse_dict"]["merge_img"]
    coarse_fg_rgb = (
        coarse_fg_rgb.detach().cpu()
        .permute(0, 2, 3, 1).numpy()
        * 255
    ).astype(np.uint8)

    coarse_fg_rgb_face = pred_dict["coarse_dict"]["merge_img_face"]
    coarse_fg_rgb_face = (
        coarse_fg_rgb_face.detach().cpu()
        .permute(0, 2, 3, 1).numpy()
        * 255
    ).astype(np.uint8)
    coarse_fg_rgb_eyes = pred_dict["coarse_dict"]["merge_img_eyes"]
    coarse_fg_rgb_eyes = (
        coarse_fg_rgb_eyes.detach().cpu()
        .permute(0, 2, 3, 1).numpy()
        * 255
    ).astype(np.uint8)
    coarse_fg_rgb_bg_img = pred_dict["coarse_dict"]["bg_img"]
    coarse_fg_rgb_bg_img = coarse_fg_rgb_bg_img.expand(gt_img.shape[0], coarse_fg_rgb_bg_img.shape[1], coarse_fg_rgb_bg_img.shape[2], coarse_fg_rgb_bg_img.shape[3])
    coarse_fg_rgb_bg_img = (
        coarse_fg_rgb_bg_img.detach().cpu()
        .permute(0, 2, 3, 1).numpy()
        * 255
    ).astype(np.uint8)

    res_img = np.concatenate([gt_img, coarse_fg_rgb, coarse_fg_rgb_face, coarse_fg_rgb_eyes, coarse_fg_rgb_bg_img], axis=1)

    img = Image.fromarray(res_img[0])
    log_image = wandb.Image(img)
    wandb.log({"Prediction": log_image})


def log_one_image(img_tensor, pred_dict):
    gt_img = (
            img_tensor.detach().cpu()
            .permute(0, 2, 3, 1).numpy()
            * 255
        ).astype(np.uint8)

    coarse_fg_rgb = pred_dict["coarse_dict"]["merge_img"]
    coarse_fg_rgb = (
        coarse_fg_rgb.detach().cpu()
        .permute(0, 2, 3, 1).numpy()
        * 255
    ).astype(np.uint8)
    res_img = np.concatenate([gt_img, coarse_fg_rgb], axis=1)

    img = Image.fromarray(res_img[0])
    log_image = wandb.Image(img)
    wandb.log({"Prediction": log_image})

def log_evaluation_image(batch_images_norm_pre, target_normalized_log, batch_images_1, batch_images_2, pred):
    res_img = np.concatenate(
                    [
                        target_normalized_log.reshape(1, 224, 224, 3),
                        batch_images_norm_pre.reshape(1, 224, 224, 3),
                    ],
                    axis=2,
                )
    img = Image.fromarray(res_img[0])
    log_image = wandb.Image(img)
    wandb.log({" Target Normalized | Prediction Normalized ": log_image})

    res_img = np.concatenate(
        [
            (
                batch_images_1.detach().cpu().permute(0, 2, 3, 1).numpy() * 255
            ).astype(np.uint8),
            (
                batch_images_2.detach().cpu().permute(0, 2, 3, 1).numpy() * 255
            ).astype(np.uint8),
            (   pred.detach().cpu().permute(0, 2, 3, 1).numpy() * 255).astype(
                np.uint8
            )
        ],
        axis=2,
    )
    img = Image.fromarray(res_img[0])
    log_image = wandb.Image(img)
    wandb.log({" Input | Target | Prediction ": log_image})

def log_simple_image(img, description):
    log_image = wandb.Image(img)
    wandb.log({description: log_image})

def log_one_subject_evaluation_results(angular_loss, angular_head_loss, ssim_loss, psnr_loss, lpips_loss,
                                         l1_loss, num_images, fid, similarity):
    wandb.log(
                {
                    "Subject Angular Error": angular_loss / num_images,
                    "Subject Angular Head Error": angular_head_loss / num_images,
                    "Subject SSIM": ssim_loss / num_images,
                    "Subject PSNR": psnr_loss / num_images,
                    "Subject LPIPS": lpips_loss / num_images,
                    "Subject L1 Distance: ": l1_loss / num_images,
                    "Subject FID: ": fid,
                    "Subject Similarity: ": similarity,
                }
            )

def log_all_datasets_evaluation_results(data_names, dict_angular_loss, dict_angular_head_loss, dict_ssim_loss, dict_psnr_loss, dict_lpips_loss,
                                        dict_l1_loss, dict_num_images, dict_fid, full_fid, dict_similarity):

    angular_loss = 0.0
    angular_head_loss = 0.0
    ssim_loss = 0.0
    psnr_loss = 0.0
    lpips_loss = 0.0
    l1_loss = 0.0
    similarity = 0.0
    num_images = 0


    for name in data_names:
        angular_loss += dict_angular_loss[name]
        angular_head_loss += dict_angular_head_loss[name]
        ssim_loss += dict_ssim_loss[name] 
        psnr_loss += dict_psnr_loss[name]
        lpips_loss += dict_lpips_loss[name]
        l1_loss += dict_l1_loss[name]
        similarity += dict_similarity[name]
        num_images += dict_num_images[name]

        wandb.log(
                {
                    name + " Angular Error": dict_angular_loss[name] / dict_num_images[name],
                    name + " Angular Head Error": dict_angular_head_loss[name] / dict_num_images[name],
                    name + " SSIM": dict_ssim_loss[name] / dict_num_images[name],
                    name + " PSNR": dict_psnr_loss[name] / dict_num_images[name],
                    name + " LPIPS": dict_lpips_loss[name] / dict_num_images[name],
                    name + " L1 Distance: ": dict_l1_loss[name] / dict_num_images[name],
                    name + " FID: ": dict_fid[name],
                    name + " Similarity: ": dict_similarity[name],
                }
            )
    wandb.log(
                {
                    " FULL Angular Error": angular_loss / num_images,
                    " FULL Angular Head Error": angular_head_loss / num_images,
                    " FULL SSIM": ssim_loss / num_images,
                    " FULL PSNR": psnr_loss / num_images,
                    " FULL LPIPS": lpips_loss / num_images,
                    " FULL L1 Distance: ": l1_loss / num_images,
                    " FULL Similarity": similarity / num_images,
                    " FULL FID: ": full_fid,
                }
            )
    

    

def log_losses(loss_dict, use_vgg_loss, use_patch_gan_loss, use_angular_loss, epoch, prefix = "TRAIN "):
    wandb.log({prefix + "Total Loss Batch": loss_dict["total_loss"]})
    if use_vgg_loss:
        wandb.log({prefix + "VGG Face Loss Batch": loss_dict["vgg_face_loss"]})
        wandb.log({prefix + "VGG Loss Batch": loss_dict["vgg"]})
    if use_patch_gan_loss:
        wandb.log({prefix + "Generator Patch GAN Loss Batch": loss_dict["gen_patch_gan_loss"]})
    if use_angular_loss:
        wandb.log({prefix + "Angular Loss Batch": loss_dict["angular"]})
        wandb.log({prefix + "Eye Region Loss Batch": loss_dict["eye_region_loss"]})
    if epoch > -1:
        wandb.log({prefix + "Head Loss Batch": loss_dict["head_loss"]})

    wandb.log({prefix + "Iden Code Loss Batch": loss_dict["iden_code"]})
    wandb.log({prefix + "Expr Code Loss Batch": loss_dict["expr_code"]})
    wandb.log({prefix + "Appea Code Loss Batch": loss_dict["appea_code"]})
    wandb.log({prefix + "BG Code Loss Batch": loss_dict["bg_code"]})
    wandb.log({prefix + "BG Loss Batch": loss_dict["bg_loss"]}) 
    wandb.log({prefix + "Face Loss Batch": loss_dict["face_loss"]})
    wandb.log({prefix + "Eyes Loss Batch": loss_dict["eyes_loss"]})
    wandb.log({prefix + "Non Head Loss Batch": loss_dict["nonhead_loss"]})
    wandb.log({prefix + "Delta Eular Loss Batch": loss_dict["delta_eular"]})
    wandb.log({prefix + "Delta Tvec Loss Batch": loss_dict["delta_tvec"]})

def log_one_h5_subject(path):
    file = h5py.File(path, "r")
    for i in range(file["face_patch"].shape[0]):
        image_gt = file["face_patch"][i, :]

        class_labels = {0: "background", 255: "eye region"}
        mask_img = wandb.Image(
            image_gt,
            masks={
                "predictions": {
                    "mask_data": file["eye_mask"][i, :],
                    "class_labels": class_labels,
                }
            },
        )
        wandb.log({"Eye Mask": mask_img})

        class_labels = {0: "background", 255: "face"}
        mask_img = wandb.Image(
            image_gt,
            masks={
                "predictions": {
                    "mask_data": file["head_mask"][i, :],
                    "class_labels": class_labels,
                }
            },
        )
        wandb.log({"Face Segmentation": mask_img})

        lms = file["facial_landmarks"][i, :]
        for j in range(lms.shape[0]):
            x = int(lms[j][0])
            y = int(lms[j][1])
            cv2.circle(image_gt, (x, y), radius=2, color=(102, 102, 255), thickness=1)

        log_image = wandb.Image(image_gt)
        wandb.log({"Facial Landmarks": log_image})

        if i % 13 == 0:
            print(
                "--------------gaze_position----------------- "
                + str(file["pitchyaw"][i, :])
            )
            print(
                "--------------code----------------- "
                + str(file["latent_codes"][i, :10])
            )