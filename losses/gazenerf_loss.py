import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from torchvision import transforms
import torch.nn as nn

from gaze_estimation.xgaze_baseline_vgg import gaze_network

trans = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.ToTensor(),  # this also convert pixel value from [0,255] to [0,1]
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

trans_eval = transforms.Compose([transforms.Resize(size=(224, 224))])


def discriminator_loss(real, fake, device):
        GANLoss = nn.BCEWithLogitsLoss(reduction='mean')
        real_size = list(real.size())
        fake_size = list(fake.size())
        real_label = torch.zeros(real_size, dtype=torch.float32).to(device)
        fake_label = torch.ones(fake_size, dtype=torch.float32).to(device)

        discriminator_loss = (GANLoss(fake, fake_label) + GANLoss(real, real_label)) / 2

        return discriminator_loss

def generator_loss(fake,device):
        GANLoss = nn.BCEWithLogitsLoss(reduction='mean')
        fake_size = list(fake.size())
        fake_label = torch.zeros(fake_size, dtype=torch.float32).to(device)
        return GANLoss(fake, fake_label)


class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, resize=True):
        """
        Init function for the perceptual loss using VGG16 model.
        :resize: Boolean value that indicates if to resize the images
        """

        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        blocks.append(torchvision.models.vgg16(pretrained=True).features[:4].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[4:9].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[9:16].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[16:23].eval())
        for bl in blocks:
            for p in bl.parameters():
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.resize = resize
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def forward(self, input, target, feature_layers=[0, 1, 2, 3], style_layers=[]):
        """
        Forward function that calculate a perceptual loss using the VGG16 model.
        :input: Generated image
        :target: Groundtruth image
        :feature_layers: Which layers to use from the VGG16 model
        :style_layers: Style layers to use
        :return: Returns a perceptual loss between the groundtruth and generated images
        """

        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        input = (input - self.mean) / self.std
        target = (target - self.mean) / self.std
        if self.resize:
            input = self.transform(
                input, mode="bilinear", size=(224, 224), align_corners=False
            )
            target = self.transform(
                target, mode="bilinear", size=(224, 224), align_corners=False
            )
        loss = 0.0
        x = input
        y = target
        for i, block in enumerate(self.blocks):
            x = block(x)
            y = block(y)
            if i in feature_layers:
                loss += torch.nn.functional.l1_loss(x, y)
            if i in style_layers:
                act_x = x.reshape(x.shape[0], x.shape[1], -1)
                act_y = y.reshape(y.shape[0], y.shape[1], -1)
                gram_x = act_x @ act_x.permute(0, 2, 1)
                gram_y = act_y @ act_y.permute(0, 2, 1)
                loss += torch.nn.functional.l1_loss(gram_x, gram_y)
        return loss


class GazePerceptualLoss(torch.nn.Module):
    def __init__(self, resize=True, device=None):
        """
        Init function for the gaze perceptual loss using VGG16 model.
        :resize: Boolean value that indicates if to resize the images
        """

        super(GazePerceptualLoss, self).__init__()
        self.model = gaze_network().to(device)
        self.device = device
        path = "configs/config_models/epoch_60_512_ckpt.pth.tar'"
        state_dict = torch.load(path, map_location=torch.device("cpu"))
        self.model.load_state_dict(state_dict=state_dict["model_state"])
        self.model.eval()
        self.img_dim = 224
        self.transform = torch.nn.functional.interpolate
        self.resize = resize
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

        self.cam_matrix = []
        self.cam_distortion = []
        self.face_model_load = np.loadtxt(
            "data/eth_xgaze/face_model.txt"
        )  # Generic face model with 3D facial landmarks

        for cam_id in range(18):
            cam_file_name = "data/eth_xgaze/cam/cam" + str(cam_id).zfill(2) + ".xml"
            fs = cv2.FileStorage(cam_file_name, cv2.FILE_STORAGE_READ)
            self.cam_matrix.append(fs.getNode("Camera_Matrix").mat())
            self.cam_distortion.append(fs.getNode("Distortion_Coefficients").mat())
            fs.release()

    def nn_angular_distance(self, a, b):
        sim = F.cosine_similarity(a, b, eps=1e-6)
        sim = F.hardtanh(sim, -1.0, 1.0)
        return torch.acos(sim) * (180 / np.pi)

    def pitchyaw_to_vector(self, pitchyaws):
        sin = torch.sin(pitchyaws)
        cos = torch.cos(pitchyaws)
        return torch.stack([cos[:, 0] * sin[:, 1], sin[:, 0], cos[:, 0] * cos[:, 1]], 1)

    def gaze_angular_loss(self, y, y_hat):
        y = self.pitchyaw_to_vector(y)
        y_hat = self.pitchyaw_to_vector(y_hat)
        loss = self.nn_angular_distance(y, y_hat)
        return torch.mean(loss)

    def forward(self, input, target, cam_ind, ldms):
        """
        Forward function that calculate a perceptual loss using the VGG16 model.
        :input: Generated image
        :target: Groundtruth image
        :feature_layers: Which layers to use from the VGG16 model
        :style_layers: Style layers to use
        :return: Returns a perceptual loss between the groundtruth and generated images
        """
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        input = (input - self.mean) / self.std
        target = (target - self.mean) / self.std
        if self.resize:
            input = trans_eval(input)
            target = trans_eval(target)

        ldms = ldms[0]
        x = input
        y = target


        gaze_x, head_x = self.model(x)

        gaze_y, head_x = self.model(y)

        angular_loss = self.gaze_angular_loss(gaze_y.detach(), gaze_x)

        return angular_loss


class GazeNeRFLoss(object):
    def __init__(
        self,
        eye_loss_importance,
        vgg_importance,
        bg_type="white",
        use_vgg_loss=True,
        use_l1_loss=False,
        use_angular_loss=False,
        use_patch_gan_loss=False,
        device=None,
    ) -> None:
        """
        Init function for GazeNeRFLoss.
        :eye_loss_importance: Weight to give to the mse loss of the eye region
        :bg_type:  Backgorund color, can be black or white
        :use_vgg_loss: Boolean value that indicate if to use the perceptual loss
        :device: Device to use
        """

        super().__init__()

        self.vgg_importance = vgg_importance
        self.eye_loss_importance = eye_loss_importance
        self.eye_region_importance = 1.0

        if bg_type == "white":
            self.bg_value = 1.0
        elif bg_type == "black":
            self.bg_value = 0.0
        else:
            self.bg_type = None
            print("Error BG type. ")
            exit(0)

        self.use_vgg_loss = use_vgg_loss
        self.use_l1_loss = use_l1_loss
        self.use_angular_loss = use_angular_loss
        self.use_patch_gan_loss = use_patch_gan_loss
        if self.use_vgg_loss:
            assert device is not None
            self.device = device
            self.vgg_loss_func = VGGPerceptualLoss(resize=True).to(self.device)
        if self.use_angular_loss:
            self.gaze_loss_func = GazePerceptualLoss(resize=True, device=device).to(
                self.device
            )

    @staticmethod
    def calc_cam_loss(delta_cam_info):
        """
        Function that calculate the camera loss, if camera parameters is enabled.
        :delta_cam_info: Camera parameters
        :return: Returns the camera loss
        """

        delta_eulur_loss = torch.mean(
            delta_cam_info["delta_eulur"] * delta_cam_info["delta_eulur"]
        )
        delta_tvec_loss = torch.mean(
            delta_cam_info["delta_tvec"] * delta_cam_info["delta_tvec"]
        )

        return {"delta_eular": delta_eulur_loss, "delta_tvec": delta_tvec_loss}

    def increase_eye_importance(self):
        if self.use_l1_loss:
            self.eye_region_importance += 1.0
            self.eye_loss_importance += 30.0
        else:
            self.eye_region_importance += 30.0
            self.eye_loss_importance += 30.0

    def calc_code_loss(self, opt_code_dict):
        """
        Function that calculate the disentanglement loss for all the latent codes.
        :opt_code_dict: Dictionary with all the latent code to optmize
        :return: Returns a dictionary with the losses calculated
        """

        iden_code_opt = opt_code_dict["iden"]
        expr_code_opt = opt_code_dict["expr"]
        iden_loss = torch.mean(iden_code_opt * iden_code_opt)
        expr_loss = torch.mean(expr_code_opt * expr_code_opt)

        appea_loss = torch.mean(opt_code_dict["appea"] * opt_code_dict["appea"])

        bg_code = opt_code_dict["bg"]
        if bg_code is None:
            bg_loss = torch.as_tensor(
                0.0, dtype=iden_loss.dtype, device=iden_loss.device
            )
        else:
            bg_loss = torch.mean(bg_code * bg_code)

        res_dict = {
            "iden_code": iden_loss,
            "expr_code": expr_loss,
            "appea_code": appea_loss,
            "bg_code": bg_loss,
        }

        return res_dict

    def calc_data_loss(
        self,
        data_dict,
        gt_rgb,
        head_mask_c1b,
        face_mask_c1b,
        nonhead_mask_c1b,
        full_eyes_mask_c1b,
        eyes_mask_c1b,
        cam_ind,
        ldms,
        epoch,
        batch_num,
        discriminator,
    ):
        """
        Function that calculate the L2 loss for head, eye and non-head regions. It also calculate the perceptual loss.
        :data_dict: Dictionary with gazenerf prediction
        :gt_rgb: Ground truth image
        :head_mask_c1b: Face mask
        :nonhead_mask_c1b: Non-head region mask
        :eye_mask_c1b: Eye region mask
        :return: Returns a dictionary with the losses calculated
        """

        bg_value = self.bg_value

        bg_img = data_dict["bg_img"]
        bg_loss = torch.mean((bg_img - bg_value) * (bg_img - bg_value))

        res_img_face = data_dict["merge_img_face"]
        res_img_eyes = data_dict["merge_img_eyes"]
        res_img = data_dict["merge_img"]

        head_mask_c3b = head_mask_c1b.expand(-1, 3, -1, -1)
        face_mask_c3b = face_mask_c1b.expand(-1, 3, -1, -1)
        eyes_mask_c3b = eyes_mask_c1b.expand(-1, 3, -1, -1)

        if self.use_l1_loss:
            head_loss = torch.nn.functional.l1_loss(res_img[head_mask_c3b], gt_rgb[head_mask_c3b])
            eyes_loss = torch.nn.functional.l1_loss(
                res_img_eyes[eyes_mask_c3b], gt_rgb[eyes_mask_c3b]
            )
            face_loss = torch.nn.functional.l1_loss(
                res_img_face[face_mask_c3b], gt_rgb[face_mask_c3b]
            )
            
        else:
            head_loss = F.mse_loss(res_img[head_mask_c3b], gt_rgb[head_mask_c3b])
            eyes_loss = F.mse_loss(res_img_eyes[eyes_mask_c3b], gt_rgb[eyes_mask_c3b])
            face_loss = F.mse_loss(res_img_face[face_mask_c3b], gt_rgb[face_mask_c3b])

        nonhead_mask_c3b = nonhead_mask_c1b.expand(-1, 3, -1, -1)
        
        temp_tensor = res_img[nonhead_mask_c3b]

        tv = temp_tensor - bg_value
        nonhaed_loss = torch.mean(tv * tv)

        res = {
            "bg_loss": bg_loss,
            "eyes_loss": eyes_loss,
            "face_loss": face_loss,
            "nonhead_loss": nonhaed_loss,
        }
        
        if epoch > -1:
            res["head_loss"] = head_loss
            
        if self.use_vgg_loss:
            masked_face_gt_img = gt_rgb.clone()
            masked_face_gt_img[~face_mask_c3b] = bg_value
            temp_res_img_face = data_dict["merge_img_face"]
            vgg_loss_face = self.vgg_loss_func(temp_res_img_face, masked_face_gt_img)
            
            masked_eyes_gt_img = gt_rgb.clone()
            masked_eyes_gt_img[~eyes_mask_c3b] = bg_value
            temp_res_img_eyes = data_dict["merge_img_eyes"]
            vgg_loss_eyes = self.vgg_loss_func(temp_res_img_eyes, masked_eyes_gt_img)
            
            res["vgg_face_loss"] = vgg_loss_face
            res["vgg_eyes_loss"] = vgg_loss_eyes
            
            
            masked_gt_img = gt_rgb.clone()
            masked_gt_img[nonhead_mask_c3b] = bg_value
            temp_res_img = res_img
            vgg_loss = self.vgg_loss_func(temp_res_img, masked_gt_img)
            res["vgg"] = vgg_loss * self.vgg_importance
        
        if epoch > -1:
            masked_gt_img = gt_rgb.clone()
            masked_gt_img[nonhead_mask_c3b] = bg_value
            temp_res_img = res_img
    
            if self.use_angular_loss:
                angular = self.gaze_loss_func(temp_res_img, masked_gt_img, cam_ind, ldms)
                res["angular"] = (angular / 60000.0) * self.eye_loss_importance  
        if self.use_patch_gan_loss:
           # Warm up period for generator losses
           warm_up_coeff = torch.tensor(max(min(1.0 / 10.0, (200000 * epoch + batch_num) / 200000), 0.0))
           patch_pred = data_dict["merge_img"]
           fake = discriminator(trans_eval(patch_pred))
           res["gen_patch_gan_loss"] = generator_loss(fake=fake, device = self.device) * warm_up_coeff
        return res

    def calc_total_loss(
        self,
        delta_cam_info,
        opt_code_dict,
        pred_dict,
        gt_rgb,
        face_mask_tensor,
        full_eye_mask_tensor,
        left_eye_mask_tensor,
        right_eye_mask_tensor,
        cam_ind,
        ldms,
        epoch,
        batch_num,
        discriminator = None
    ):
        """
        Function that calcluate all the losses of gaze nerf.
        :delta_cam_info: Camera parameters
        :opt_code_dict: Dictionary with all the latent code to optmize
        :pred_dict: Dictionary with gazenerf prediction
        :gt_rgb: Ground truth image
        :face_mask_tensor: Face mask
        :eye_mask_tensor: Eye region mask
        :return: Returns a dictionary with the weighted losses calculated
        """
        
        head_mask =  torch.logical_and(face_mask_tensor >= 0.5, full_eye_mask_tensor < 0.5)
        face_mask = torch.logical_and(face_mask_tensor >= 0.5, torch.logical_and(left_eye_mask_tensor < 0.5, right_eye_mask_tensor < 0.5))
        eyes_mask = torch.logical_or(left_eye_mask_tensor >= 0.5, right_eye_mask_tensor >= 0.5)
        full_eyes_mask = None
        nonhead_mask = face_mask_tensor < 0.5
        

        coarse_data_dict = pred_dict["coarse_dict"]
        loss_dict = self.calc_data_loss(
            coarse_data_dict,
            gt_rgb,
            head_mask,
            face_mask,
            nonhead_mask,
            full_eyes_mask,
            eyes_mask,
            cam_ind,
            ldms,
            epoch,
            batch_num,
            discriminator,
        )

        total_loss = 0.0
        for k in loss_dict:
            total_loss += loss_dict[k]

        # cam loss
        if delta_cam_info is not None:
            loss_dict.update(self.calc_cam_loss(delta_cam_info))
            total_loss += (
                0.001 * loss_dict["delta_eular"] + 0.001 * loss_dict["delta_tvec"]
            )

        # code loss
        loss_dict.update(self.calc_code_loss(opt_code_dict))
        total_loss += (
            0.001 * loss_dict["iden_code"]
            + 1.0 * loss_dict["expr_code"]
            + 0.001 * loss_dict["appea_code"]
            + 0.01 * loss_dict["bg_code"]
        )
        loss_dict["total_loss"] = total_loss
        return loss_dict