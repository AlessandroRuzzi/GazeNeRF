import logging
import os
import random

import imageio
import numpy
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

import wandb
from configs.gazenerf_options import BaseOptions
from losses.gazenerf_loss import GazeNeRFLoss
from models.gaze_nerf import GazeNeRFNet
from utils.logging import (log_all_images, log_losses, log_one_image,
                           log_one_number)
from utils.render_utils import RenderUtils
from models.discriminator import PatchGAN
from losses.gazenerf_loss import discriminator_loss, generator_loss
import traceback

from .base import BaseTrainer

logger = logging.getLogger(__name__)

inv_trans = transforms.Compose(
    [
        transforms.Normalize(
            mean=[0.0, 0.0, 0.0], std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
        ),
        transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1.0, 1.0, 1.0]),
    ]
)

trans_eval = transforms.Compose([transforms.Resize(size=(224, 224))])


class GazeNerfTrainer(BaseTrainer):
    """Trainer code for GazeNerf"""

    def __init__(
        self,
        lr,
        num_iter,
        step_decay,
        optimizer,
        vgg_importance,
        eye_loss_importance,
        model_path,
        state_dict_name,
        use_vgg_loss,
        use_angular_loss,
        use_patch_gan_loss,
        is_gradual_loss=False,
        use_l1_loss=False,
        checkpoint_dir=None,
        batch_size=16,
        gpu=None,
        resume=False,
        include_vd=False,
        hier_sampling=False,
        log=False,
        fit_image=False,
    ):
        super(GazeNerfTrainer, self).__init__(checkpoint_dir, batch_size, gpu, log)

        if torch.cuda.is_available():
            self.device = torch.device("cuda:%s" % gpu)
            torch.cuda.empty_cache()
        else:
            self.device = "cpu"
        self.opt_cam = True
        self.resume = resume
        self.lr = lr
        self.optimizer_name = optimizer
        self.vgg_importance = vgg_importance
        self.eye_loss_importance = eye_loss_importance
        self.num_iter = num_iter
        self.step_decay = step_decay
        self.include_vd = include_vd
        self.use_vgg_loss = use_vgg_loss
        self.is_gradual_loss = is_gradual_loss
        self.use_l1_loss = use_l1_loss
        self.use_angular_loss = use_angular_loss
        self.use_patch_gan_loss = use_patch_gan_loss
        self.hier_sampling = hier_sampling
        self.view_num = 45
        self.duration = 3.0 / self.view_num
        self.fit_image = fit_image
        self.state_dict_name = state_dict_name
        self.model_path = model_path
        self.base_expr_fix = torch.load("configs/config_files/tensor.pt")

        self.build_info()
        self.build_tool_funcs()

    def build_info(self):
        para_dict = None
        if self.resume:
            check_dict = torch.load(self.model_path, map_location=torch.device("cpu"))
            self.check_dict = check_dict

        self.opt = BaseOptions(para_dict)

        net = GazeNeRFNet(
            self.opt, include_vd=self.include_vd, hier_sampling=self.hier_sampling
        )
        if self.use_patch_gan_loss:
            self.discriminator = PatchGAN(input_nc=3).to(self.device)
        else:
            self.discriminator = None

        if self.resume:
            net.load_state_dict(check_dict["net"])

            logger.info(
                "Successfully loaded checkpoint model",
            )

        self.featmap_size = self.opt.featmap_size
        self.pred_img_size = self.opt.pred_img_size
        self.net = net.to(self.device)
        self.net.train()
        self.logger.info(self.net)
        self.logger.info(
            "Number of parameters: %i", sum(p.numel() for p in self.net.parameters())
        )

    def build_tool_funcs(self):
        self.loss_utils = GazeNeRFLoss(
            eye_loss_importance=self.eye_loss_importance,
            vgg_importance=self.vgg_importance,
            use_vgg_loss=self.use_vgg_loss,
            use_l1_loss=self.use_l1_loss,
            use_angular_loss=self.use_angular_loss,
            use_patch_gan_loss=self.use_patch_gan_loss,
            device=self.device,
        )
        self.render_utils = RenderUtils(view_num=45, device=self.device, opt=self.opt)

        self.xy = self.render_utils.ray_xy
        self.uv = self.render_utils.ray_uv

        self.xy = self.xy.expand([self.batch_size, self.xy.size(1), self.xy.size(2)])
        self.uv = self.uv.expand([self.batch_size, self.uv.size(1), self.uv.size(2)])

    def state_dict(self):
        """Trainer state dict for checkpointing"""
        return dict(
            model=self.model.state_dict(),
            optimizer=self.optimizer.state_dict(),
        )

    def save_checkpoint(
        self,
        current_epoch: int,
        current_loss: float,
    ) -> None:
        """Save (or checkpoint) the model and training state."""

        self.state = {
            # random states
            "torch_random_state": torch.get_rng_state(),
            "numpy_random_state": numpy.random.get_state(),
            "random_random_state": random.getstate(),
            # epoch info
            "checkpoint_epoch": current_epoch,
            "checkpoint_loss": current_loss,
            "resume_epoch": current_epoch + 1,  # resume from next epoch
            # model state
            "net": self.net.state_dict(),
            "para": self.opt,
            "optimizer": self.optimizer.state_dict(),
            "iden_offset": self.iden_offset,
            "expr_offset": self.expr_offset,
            "appea_offset": self.appea_offset,
            "delta_EulurAngles": self.delta_EulurAngles,
            "delta_Tvecs": self.delta_Tvecs,
        }

        os.makedirs(self.checkpoint_dir, exist_ok=True)
        state_file = os.path.join(
            self.checkpoint_dir,
            self.state_dict_name
            + "_"
            + str(current_epoch + 1)
            + ".json",
        )
        torch.save(self.state, state_file)

    def load_checkpoint(self) -> None:
        """Load model and training state from a checkpoint."""
        if self.disable_checkpoints:
            logger.warning("Skipping checkpoints")
            return

        logger.debug(
            "Trying to load saved model and training state from %s",
            self.checkpoint_path,
        )

        state_file = os.path.join(self.checkpoint_path, self.state_dict_name)

        if not os.path.exists(state_file):
            logger.info(
                "Starting fresh: could not load a checkpoint from %s",
                self.checkpoint_path,
            )
            return

        # loads model and optimizer inplace
        self.state = torch.load(state_file)
        self.model.load_state_dict(self.state["net"])
        self.optimizer.load_state_dict(self.state["optimizer"])

        # loads random state
        torch.set_rng_state(self.state["torch_random_state"])
        numpy.random.set_state(self.state["numpy_random_state"])
        random.setstate(self.state["random_random_state"])

        logger.info(
            "Successfully loaded checkpoint from epoch %s loss %s",
            self.state["checkpoint_epoch"],
            round(self.state["checkpoint_loss"], 3),
        )

    def get_optimizer(self, params_group):
        if self.optimizer_name == "adam":
            self.optimizer = torch.optim.Adam(params_group, betas=(0.9, 0.999))
        elif self.optimizer_name == "sgd":
            self.optimizer = torch.optim.SGD(params_group)
        else:
            raise NotImplementedError
        self.lr_func = lambda epoch: 0.1 ** (epoch / self.step_decay)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda=self.lr_func
        )

        if self.use_patch_gan_loss:
            self.discriminator_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr, weight_decay=1e-4)

        if self.resume and not self.fit_image:
            self.optimizer.load_state_dict(self.check_dict["optimizer"])
            logger.info(
                "Successfully loaded checkpoint optimizer",
            )

    def prepare_data(
        self, img, head_mask, left_eye_mask, right_eye_mask, nl3dmm_para_dict
    ):
        # process imgs
        self.para_dict = nl3dmm_para_dict
        gt_img_size = self.pred_img_size
        self.img_tensor = img.clone().to(self.device)
        self.head_mask_tensor = head_mask.clone().unsqueeze(1).to(self.device)
        self.left_eye_mask_tensor = left_eye_mask.clone().unsqueeze(1).to(self.device)
        self.right_eye_mask_tensor = right_eye_mask.clone().unsqueeze(1).to(self.device)

        base_code = nl3dmm_para_dict["code"].clone().detach().to(self.device)
        self.base_iden = (
            base_code[:, : self.opt.iden_code_dims]
            .type(torch.FloatTensor)
            .to(self.device)
        )
        self.base_expr = (
            base_code[
                :,
                self.opt.iden_code_dims : self.opt.iden_code_dims
                + self.opt.expr_code_dims,
            ]
            .type(torch.FloatTensor)
            .to(self.device)
        )
        self.base_text = (
            base_code[
                :,
                self.opt.iden_code_dims
                + self.opt.expr_code_dims : self.opt.iden_code_dims
                + self.opt.expr_code_dims
                + self.opt.text_code_dims,
            ]
            .type(torch.FloatTensor)
            .to(self.device)
        )
        self.base_illu = (
            base_code[
                :,
                self.opt.iden_code_dims
                + self.opt.expr_code_dims
                + self.opt.text_code_dims :,
            ]
            .type(torch.FloatTensor)
            .to(self.device)
        )
        self.base_gaze_direction = (
            nl3dmm_para_dict["pitchyaw"]
            .clone()
            .detach()
            .type(torch.FloatTensor)
            .to(self.device)
        )

        self.base_expr = (
            self.base_expr_fix.expand([self.batch_size, -1])
            .detach()
            .type(torch.FloatTensor)
            .to(self.device)
        )

        self.base_c2w_Rmat = nl3dmm_para_dict["c2w_Rmat"].clone().detach()
        self.base_c2w_Tvec = nl3dmm_para_dict["c2w_Tvec"].clone().detach().unsqueeze(-1)
        self.base_w2c_Rmat = nl3dmm_para_dict["w2c_Rmat"].clone().detach()
        self.base_w2c_Tvec = nl3dmm_para_dict["w2c_Tvec"].clone().detach().unsqueeze(-1)

        temp_inmat = nl3dmm_para_dict["inmat"].clone().detach()
        temp_inmat[:, :2, :] *= self.featmap_size / gt_img_size

        temp_inv_inmat = torch.zeros_like(temp_inmat)
        temp_inv_inmat[:, 0, 0] = 1.0 / temp_inmat[:, 0, 0]
        temp_inv_inmat[:, 1, 1] = 1.0 / temp_inmat[:, 1, 1]
        temp_inv_inmat[:, 0, 2] = -(temp_inmat[:, 0, 2] / temp_inmat[:, 0, 0])
        temp_inv_inmat[:, 1, 2] = -(temp_inmat[:, 1, 2] / temp_inmat[:, 1, 1])
        temp_inv_inmat[:, 2, 2] = 1.0

        self.temp_inmat = temp_inmat
        self.temp_inv_inmat = temp_inv_inmat

        self.cam_info = {
            "batch_Rmats": self.base_c2w_Rmat.type(torch.FloatTensor).to(self.device),
            "batch_Tvecs": self.base_c2w_Tvec.type(torch.FloatTensor).to(self.device),
            "batch_inv_inmats": self.temp_inv_inmat.type(torch.FloatTensor).to(
                self.device
            ),
        }

    def build_code_and_cam(self, iter):

        pos = iter * self.batch_size

        # code
        shape_code = (
            torch.cat(
                [
                    self.base_iden + self.iden_offset[pos : pos + self.batch_size],
                    self.base_expr + self.expr_offset[pos : pos + self.batch_size],
                ],
                dim=-1,
            )
            .type(torch.FloatTensor)
            .to(self.device)
        )
        appea_code = (
            (
                torch.cat([self.base_text, self.base_illu], dim=-1)
                + self.appea_offset[pos : pos + self.batch_size]
            )
            .type(torch.FloatTensor)
            .to(self.device)
        )
        gaze_code = self.base_gaze_direction.type(torch.FloatTensor).to(self.device)

        opt_code_dict = {
            "bg": None,
            "iden": self.iden_offset[pos : pos + self.batch_size],
            "expr": self.expr_offset[pos : pos + self.batch_size],
            "appea": self.appea_offset[pos : pos + self.batch_size],
        }
        code_info = {
            "bg_code": None,
            "shape_code": shape_code,
            "appea_code": appea_code,
            "gaze_code": gaze_code,
        }

        # cam
        if self.opt_cam:
            delta_cam_info = {
                "delta_eulur": self.delta_EulurAngles[pos : pos + self.batch_size],
                "delta_tvec": self.delta_Tvecs[pos : pos + self.batch_size],
            }
            batch_delta_Rmats = self.eulurangle2Rmat(
                self.delta_EulurAngles[pos : pos + self.batch_size]
            )
            base_Rmats = self.cam_info["batch_Rmats"]
            base_Tvecs = self.cam_info["batch_Tvecs"]
            cur_Rmats = batch_delta_Rmats.bmm(base_Rmats)
            cur_Tvecs = (
                batch_delta_Rmats.bmm(base_Tvecs)
                + self.delta_Tvecs[pos : pos + self.batch_size]
            )

            batch_inv_inmat = self.cam_info["batch_inv_inmats"]  # [N, 3, 3]
            batch_cam_info = {
                "batch_Rmats": cur_Rmats,
                "batch_Tvecs": cur_Tvecs,
                "batch_inv_inmats": batch_inv_inmat,
            }

        else:
            delta_cam_info = None
            batch_cam_info = self.cam_info

        return code_info, opt_code_dict, batch_cam_info, delta_cam_info

    def prepare_optimizer_opt(self):
        if self.resume and not self.fit_image:
            self.delta_EulurAngles = torch.FloatTensor(
                self.check_dict["delta_EulurAngles"].detach().numpy()
            ).to(self.device)
            self.delta_Tvecs = torch.FloatTensor(
                self.check_dict["delta_Tvecs"].detach().numpy()
            ).to(self.device)
            self.iden_offset = torch.FloatTensor(
                self.check_dict["iden_offset"].detach().numpy()
            ).to(self.device)
            self.expr_offset = torch.FloatTensor(
                self.check_dict["expr_offset"].detach().numpy()
            ).to(self.device)
            self.appea_offset = torch.FloatTensor(
                self.check_dict["appea_offset"].detach().numpy()
            ).to(self.device)
        else:
            self.delta_EulurAngles = torch.zeros(
                (self.train_len, 3), dtype=torch.float32
            ).to(self.device)
            self.delta_Tvecs = torch.zeros(
                (self.train_len, 3, 1), dtype=torch.float32
            ).to(self.device)
            self.iden_offset = torch.zeros(
                (self.train_len, 100), dtype=torch.float32
            ).to(self.device)

            self.expr_offset = torch.zeros(
                (self.train_len, 79), dtype=torch.float32
            ).to(self.device)
            self.appea_offset = torch.zeros(
                (self.train_len, 127), dtype=torch.float32
            ).to(self.device)

        if self.opt_cam:
            self.enable_gradient(
                [
                    self.iden_offset,
                    self.expr_offset,
                    self.appea_offset,
                    self.delta_EulurAngles,
                    self.delta_Tvecs,
                ]
            )
        else:
            self.enable_gradient(
                [
                    self.iden_offset,
                    self.expr_offset,
                    self.appea_offset,
                ]
            )

        init_learn_rate = self.lr

        params_group = [
            {"params": self.net.parameters(), "lr": init_learn_rate},
            {"params": [self.iden_offset], "lr": init_learn_rate * 1.0},
            {"params": [self.expr_offset], "lr": init_learn_rate * 0.1},
            {"params": [self.appea_offset], "lr": init_learn_rate * 1.0},
        ]

        if self.opt_cam:
            params_group += [
                {"params": [self.delta_EulurAngles], "lr": init_learn_rate * 0.1},
                {"params": [self.delta_Tvecs], "lr": init_learn_rate * 0.1},
            ]

        self.get_optimizer(params_group)

    def perform_fitting(self, i, cam_ind, ldms, epoch):

        with torch.set_grad_enabled(True):
            (
                code_info,
                opt_code_dict,
                cam_info,
                delta_cam_info,
            ) = self.build_code_and_cam(i)
            pred_dict = self.net("train", self.xy, self.uv, **code_info, **cam_info)

            if self.use_patch_gan_loss:
                for param in self.discriminator.parameters():
                    param.requires_grad = True
                patch_gt = self.img_tensor.clone()
                nonhead_mask = self.head_mask_tensor < 0.5
                nonhead_mask_c3b = nonhead_mask.expand(-1, 3, -1, -1)
                patch_gt[nonhead_mask_c3b] = 1.0
                patch_pred = pred_dict["coarse_dict"]["merge_img"].detach().clone()
                real = self.discriminator(trans_eval(patch_gt))
                fake = self.discriminator(trans_eval(patch_pred.detach()))
                disc_loss = discriminator_loss(real=real, fake=fake, device=self.device)
                gen_loss =generator_loss(fake=fake, device=self.device)
                self.discriminator_optimizer.zero_grad()
                disc_loss.backward()
                self.discriminator_optimizer.step()

                for param in self.discriminator.parameters():
                    param.requires_grad = False
                if self.log and i % 300 == 0:
                    log_one_number(float(disc_loss.detach()), "TRAIN " + "Discriminator Patch GAN Loss Batch")

            batch_loss_dict = self.loss_utils.calc_total_loss(
                delta_cam_info=delta_cam_info,
                opt_code_dict=opt_code_dict,
                pred_dict=pred_dict,
                gt_rgb=self.img_tensor,
                face_mask_tensor=self.head_mask_tensor,
                full_eye_mask_tensor = self.para_dict["eye_mask"].to(self.device),
                left_eye_mask_tensor=self.left_eye_mask_tensor,
                right_eye_mask_tensor=self.right_eye_mask_tensor,
                cam_ind=cam_ind,
                ldms=ldms,
                epoch=epoch,
                batch_num = i,
                discriminator=self.discriminator,
            )

        self.optimizer.zero_grad()
        batch_loss_dict["total_loss"].backward()
        self.optimizer.step()

        if self.log and i % 20 == 0:
            log_all_images(self.img_tensor, pred_dict)

        del pred_dict
        return batch_loss_dict

    def train_epoch(self, data_loader, epoch):
        """Train for one epoch"""

        self.net.train()

        train_epoch_loss = 0

        batch_loop_bar = tqdm(data_loader, leave=False, desc="Batch Progress")

        # Loop over training batches
        for i, (
            batch_images,
            batch_head_mask,
            batch_left_eye_mask,
            batch_right_eye_mask,
            batch_nl3dmm_para_dict,
            ldms,
            cam_ind,
        ) in enumerate(batch_loop_bar):

            try:
                flag = 0
                for j in range(batch_images.shape[0]):
                    if (
                        len(torch.unique(batch_head_mask[j,:])) == 1 or  (len(torch.unique(batch_left_eye_mask[j,:])) == 1 and len(torch.unique(batch_right_eye_mask[j,:])) == 1 )
                    ):
                        flag = 1
                        break
                if flag == 1:
                    continue
            except:
                continue

            self.prepare_data(
                batch_images,
                batch_head_mask,
                batch_left_eye_mask,
                batch_right_eye_mask,
                batch_nl3dmm_para_dict,
            )
            try:
                loss_dict = self.perform_fitting(i, cam_ind, ldms, epoch)
                train_epoch_loss += float(loss_dict["total_loss"].detach())
                batch_loop_bar.set_postfix(loss=float(loss_dict["total_loss"].detach()))
            except Exception:
                traceback.print_exc()
                continue

            if self.log and i % 300 == 0:
                log_losses(
                    loss_dict,
                    self.use_vgg_loss,
                    self.use_patch_gan_loss,
                    self.use_angular_loss,
                    epoch,
                )
            del loss_dict

        train_epoch_loss = train_epoch_loss / (i + 1)
        if self.log:
            log_one_number(train_epoch_loss, "Total Loss Epoch")

        # Return summary
        return dict(loss=float(train_epoch_loss))

    @torch.no_grad()
    def evaluate(self, data_loader):
        """ "Evaluate the model"""

        sum_loss = {}

        sum_loss["total_loss"] = 0

        self.net.eval()
        self.val_len = len(data_loader.dataset)
        self.prepare_optimizer_opt_val()

        batch_loop_bar = tqdm(data_loader, leave=False, desc="Validation Progress")

        for i, (
            batch_images,
            batch_head_mask,
            batch_eye_mask,
            batch_nl3dmm_para_dict,
            ldms,
            cam_ind,
        ) in enumerate(batch_loop_bar):
            try:
                if len(torch.unique(batch_eye_mask)) == 1:
                    continue
            except:
                continue
            self.prepare_data_val(
                batch_images,
                batch_head_mask,
                batch_eye_mask,
                batch_nl3dmm_para_dict,
            )
            with torch.set_grad_enabled(False):
                (
                    code_info,
                    opt_code_dict,
                    cam_info,
                    delta_cam_info,
                ) = self.build_code_and_cam_val(i)
                pred_dict = self.net("test", self.xy, self.uv, **code_info, **cam_info)

                loss_dict = self.loss_utils.calc_total_loss(
                    delta_cam_info=delta_cam_info,
                    opt_code_dict=opt_code_dict,
                    pred_dict=pred_dict,
                    gt_rgb=self.img_tensor,
                    face_mask_tensor=self.head_mask_tensor,
                    eye_mask_tensor=self.eye_mask_tensor,
                    cam_ind=cam_ind,
                    ldms=ldms,
                )

                sum_loss["total_loss"] += float(loss_dict["total_loss"].detach())

                batch_loop_bar.set_postfix(loss=float(loss_dict["total_loss"].detach()))

                if self.log and i % 20 == 0:
                    wandb.log({"VAL total loss batch": loss_dict["total_loss"]})
                    if self.use_vgg_loss:
                        wandb.log({"VAL vgg loss batch": loss_dict["vgg"]})
                    if self.use_patch_gan_loss:
                        wandb.log({"VAL gaze_vgg loss batch": loss_dict["gaze_vgg"]})
                    if self.use_angular_loss:
                        wandb.log({"VAL angular loss batch": loss_dict["angular"]})
                    wandb.log({"VAL iden_code loss batch": loss_dict["iden_code"]})
                    wandb.log({"VAL expr_code loss batch": loss_dict["expr_code"]})
                    wandb.log({"VAL gaze_code loss batch": loss_dict["gaze_code"]})
                    wandb.log({"VAL appea_code loss batch": loss_dict["appea_code"]})
                    wandb.log({"VAL bg_code loss batch": loss_dict["bg_code"]})
                    wandb.log({"VAL bg_loss loss batch": loss_dict["bg_loss"]})
                    wandb.log({"VAL head loss batch": loss_dict["head_loss"]})
                    wandb.log({"VAL eye loss batch": loss_dict["eye_loss"]})
                    wandb.log({"VAL non head loss batch": loss_dict["nonhead_loss"]})
                    wandb.log({"VAL delta_eular loss batch": loss_dict["delta_eular"]})
                    wandb.log({"VAL delta_tvec loss batch": loss_dict["delta_tvec"]})
                del loss_dict

                gt_img = (
                    # inv_trans(
                    self.img_tensor.detach().cpu()
                    # .permute(0, 2, 3, 1)
                    # )
                    .permute(0, 2, 3, 1).numpy()
                    * 255
                ).astype(np.uint8)

                coarse_fg_rgb = pred_dict["coarse_dict"]["merge_img"]
                coarse_fg_rgb = (
                    # inv_trans(
                    coarse_fg_rgb.detach().cpu()
                    # )
                    .permute(0, 2, 3, 1).numpy()
                    * 255
                ).astype(np.uint8)
                res_img = np.concatenate([gt_img, coarse_fg_rgb], axis=1)

                img = Image.fromarray(res_img[0])
                if self.log and i % 20 == 0:
                    log_image = wandb.Image(img)
                    wandb.log({"Validation": log_image})

        val_loss = sum_loss["total_loss"] / (i + 1)
        if self.log:
            wandb.log({"VAL total loss epoch": val_loss})
        del pred_dict
        # Return summary
        return dict(loss=float(val_loss))

    @torch.no_grad()
    def prepare_optimizer_opt_val(self):
        self.val_delta_EulurAngles = torch.zeros(
            (self.val_len, 3), dtype=torch.float32
        ).to(self.device)
        self.val_delta_Tvecs = torch.zeros(
            (self.val_len, 3, 1), dtype=torch.float32
        ).to(self.device)

        self.val_iden_offset = torch.zeros((self.val_len, 100), dtype=torch.float32).to(
            self.device
        )

        self.val_expr_offset = torch.zeros((self.val_len, 79), dtype=torch.float32).to(
            self.device
        )
        self.val_gaze_offset = torch.zeros((self.val_len, 2), dtype=torch.float32).to(
            self.device
        )
        self.val_appea_offset = torch.zeros(
            (self.val_len, 127), dtype=torch.float32
        ).to(self.device)

    @torch.no_grad()
    def build_code_and_cam_val(self, iter):
        pos = iter * self.batch_size

        # code
        shape_code = (
            torch.cat(
                [
                    self.base_iden + self.val_iden_offset[pos : pos + self.batch_size],
                    self.base_expr + self.val_expr_offset[pos : pos + self.batch_size],
                    self.base_gaze_direction
                    + self.val_gaze_offset[pos : pos + self.batch_size],
                ],
                dim=-1,
            )
            .type(torch.FloatTensor)
            .to(self.device)
        )
        appea_code = (
            (
                torch.cat([self.base_text, self.base_illu], dim=-1)
                + self.val_appea_offset[pos : pos + self.batch_size]
            )
            .type(torch.FloatTensor)
            .to(self.device)
        )

        opt_code_dict = {
            "bg": None,
            "iden": self.val_iden_offset[pos : pos + self.batch_size],
            "expr": self.val_expr_offset[pos : pos + self.batch_size],
            "appea": self.val_appea_offset[pos : pos + self.batch_size],
            "gaze": self.val_gaze_offset[pos : pos + self.batch_size],
        }

        code_info = {
            "bg_code": None,
            "shape_code": shape_code,
            "appea_code": appea_code,
        }

        # cam
        if self.opt_cam:
            delta_cam_info = {
                "delta_eulur": self.val_delta_EulurAngles[pos : pos + self.batch_size],
                "delta_tvec": self.val_delta_Tvecs[pos : pos + self.batch_size],
            }
            batch_delta_Rmats = self.eulurangle2Rmat(
                self.val_delta_EulurAngles[pos : pos + self.batch_size]
            )
            base_Rmats = self.cam_info["batch_Rmats"]
            base_Tvecs = self.cam_info["batch_Tvecs"]
            cur_Rmats = batch_delta_Rmats.bmm(base_Rmats)
            cur_Tvecs = (
                batch_delta_Rmats.bmm(base_Tvecs)
                + self.val_delta_Tvecs[pos : pos + self.batch_size]
            )

            batch_inv_inmat = self.cam_info["batch_inv_inmats"]  # [N, 3, 3]
            batch_cam_info = {
                "batch_Rmats": cur_Rmats,
                "batch_Tvecs": cur_Tvecs,
                "batch_inv_inmats": batch_inv_inmat,
            }

        else:
            delta_cam_info = None
            batch_cam_info = self.cam_info

        return code_info, opt_code_dict, batch_cam_info, delta_cam_info

    @torch.no_grad()
    def prepare_data_val(self, img, head_mask, eye_mask, nl3dmm_para_dict):
        # process imgs
        gt_img_size = self.pred_img_size
        self.img_tensor = img.to(self.device)
        self.head_mask_tensor = head_mask.unsqueeze(1).to(self.device)
        self.eye_mask_tensor = eye_mask.unsqueeze(1).to(self.device)

        base_code = nl3dmm_para_dict["code"].detach().to(self.device)
        self.base_iden = (
            base_code[:, : self.opt.iden_code_dims]
            .type(torch.FloatTensor)
            .to(self.device)
        )
        self.base_expr = (
            base_code[
                :,
                self.opt.iden_code_dims : self.opt.iden_code_dims
                + self.opt.expr_code_dims,
            ]
            .type(torch.FloatTensor)
            .to(self.device)
        )
        self.base_text = (
            base_code[
                :,
                self.opt.iden_code_dims
                + self.opt.expr_code_dims : self.opt.iden_code_dims
                + self.opt.expr_code_dims
                + self.opt.text_code_dims,
            ]
            .type(torch.FloatTensor)
            .to(self.device)
        )
        self.base_illu = (
            base_code[
                :,
                self.opt.iden_code_dims
                + self.opt.expr_code_dims
                + self.opt.text_code_dims :,
            ]
            .type(torch.FloatTensor)
            .to(self.device)
        )

        self.base_gaze_direction = (
            nl3dmm_para_dict["pitchyaw"]
            .detach()
            .type(torch.FloatTensor)
            .to(self.device)
        )

        self.base_illu = (
            self.base_illu_fix.detach().type(torch.FloatTensor).to(self.device)
        )
        self.base_expr = (
            self.base_expr_fix.detach().type(torch.FloatTensor).to(self.device)
        )

        self.base_c2w_Rmat = nl3dmm_para_dict["c2w_Rmat"].detach()
        self.base_c2w_Tvec = nl3dmm_para_dict["c2w_Tvec"].detach().unsqueeze(-1)
        self.base_w2c_Rmat = nl3dmm_para_dict["w2c_Rmat"].detach()
        self.base_w2c_Tvec = nl3dmm_para_dict["w2c_Tvec"].detach().unsqueeze(-1)

        temp_inmat = nl3dmm_para_dict["inmat"].detach()
        temp_inmat[:, :2, :] *= self.featmap_size / gt_img_size

        temp_inv_inmat = torch.zeros_like(temp_inmat)
        temp_inv_inmat[:, 0, 0] = 1.0 / temp_inmat[:, 0, 0]
        temp_inv_inmat[:, 1, 1] = 1.0 / temp_inmat[:, 1, 1]
        temp_inv_inmat[:, 0, 2] = -(temp_inmat[:, 0, 2] / temp_inmat[:, 0, 0])
        temp_inv_inmat[:, 1, 2] = -(temp_inmat[:, 1, 2] / temp_inmat[:, 1, 1])
        temp_inv_inmat[:, 2, 2] = 1.0

        self.temp_inmat = temp_inmat
        self.temp_inv_inmat = temp_inv_inmat

        self.cam_info = {
            "batch_Rmats": self.base_c2w_Rmat.type(torch.FloatTensor).to(self.device),
            "batch_Tvecs": self.base_c2w_Tvec.type(torch.FloatTensor).to(self.device),
            "batch_inv_inmats": self.temp_inv_inmat.type(torch.FloatTensor).to(
                self.device
            ),
        }

    @torch.no_grad()
    def evaluate_single_image(self, data_loader, key, val):
        """ "Evaluate the model"""

        self.net.eval()
        self.train_len = len(data_loader.dataset)
        if not self.fit_image:
            self.prepare_optimizer_opt()

        for i, (
            batch_images,
            batch_head_mask,
            batch_left_eye_mask,
            batch_right_eye_mask,
            batch_nl3dmm_para_dict,
            _,
            _,
        ) in enumerate(data_loader):
            if i == 0:  # 13: #104:
                self.prepare_data(
                    batch_images,
                    batch_head_mask,
                    batch_left_eye_mask,
                    batch_right_eye_mask,
                    batch_nl3dmm_para_dict,
                )

                with torch.set_grad_enabled(False):
                    (
                        code_info,
                        opt_code_dict,
                        cam_info,
                        delta_cam_info,
                    ) = self.build_code_and_cam(i)
                    if val:
                        fit_name = "not_fitted"
                    else:
                        fit_name = "fitted"

                    path_name_head = f"logs/{key[:-3]}_{fit_name}_head.gif"
                    path_name_gaze = f"logs/{key[:-3]}_{fit_name}_gaze.gif"
                    path_name_both = f"logs/{key[:-3]}_{fit_name}_both.gif"

                    imgs = self.render_utils.render_novel_views(self.net, code_info)

                    imageio.mimsave(path_name_both, imgs, "GIF", duration=self.duration)
                    imgs = self.render_utils.render_novel_views_gaze(
                        self.net, code_info, cam_info
                    )
                    imageio.mimsave(path_name_gaze, imgs, "GIF", duration=self.duration)
                    imgs = self.render_utils.render_novel_views(
                        self.net, code_info, False
                    )
                    imageio.mimsave(path_name_head, imgs, "GIF", duration=self.duration)

                    break

    def train_single_image(
        self, dataloader, n_epochs, index, method, gaze_direction=None
    ):
        self.net.train()
        self.train_len = len(dataloader.dataset)
        self.prepare_optimizer_opt()

        batch_loop_bar = tqdm(range(0, n_epochs), leave=True, desc="Batch Progress")

        # Loop over training batches
        for i in batch_loop_bar:
            self.net.train()

            dataloader.dataset.modify_index(index)

            if method == "input_target_images":
                for j, (
                    ind1,
                    ind2,
                    ind3,
                    ind4,
                    ind5,
                    ldms,
                    cam_ind,
                    _,
                    _,
                    _,
                    _,
                    _,
                    _,
                    _,
                    _,
                    _,
                    _,
                    _,
                ) in enumerate(dataloader):
                    (
                        batch_images,
                        batch_head_mask,
                        batch_left_eye_mask,
                        batch_right_eye_mask,
                        batch_nl3dmm_para_dict,
                        ldms,
                        cam_ind,
                    ) = (ind1, ind2, ind3, ind4, ind5, ldms, cam_ind)
                    break
            elif (
                method == "consistency"
                or method == "gaze_transfer"
                or method == "personal_calibration"
                or method == "one_fit"
            ):
                for j, (ind1, ind2, ind3, ind4, ind5, ldms, cam_ind) in enumerate(
                    dataloader
                ):
                    (
                        batch_images,
                        batch_head_mask,
                        batch_left_eye_mask,
                        batch_right_eye_mask,
                        batch_nl3dmm_para_dict,
                        ldms,
                        cam_ind,
                    ) = (ind1, ind2, ind3, ind4, ind5, ldms, cam_ind)
                    break

            self.prepare_data(
                batch_images,
                batch_head_mask,
                batch_left_eye_mask,
                batch_right_eye_mask,
                batch_nl3dmm_para_dict,
            )
            if gaze_direction is not None:
                self.base_gaze_direction = gaze_direction

            loss_dict = self.perform_fitting(0, cam_ind, ldms, False)
            batch_loop_bar.set_postfix(loss=float(loss_dict["total_loss"].detach()))
            del loss_dict

        dataloader.dataset.modify_index(None)

    @torch.no_grad()
    def predict_single_image(
        self,
        i,
        dataloader,
        batch_images,
        batch_head_mask,
        batch_left_eye_mask,
        batch_right_eye_mask,
        batch_nl3dmm_para_dict,
    ):
        """ "Evaluate the model"""
        self.net.eval()
        self.train_len = len(dataloader.dataset)
        if not self.fit_image:
            self.prepare_optimizer_opt()

        self.prepare_data(
            batch_images,
            batch_head_mask,
            batch_left_eye_mask,
            batch_right_eye_mask,
            batch_nl3dmm_para_dict,
        )
        with torch.set_grad_enabled(False):
            (
                code_info,
                opt_code_dict,
                cam_info,
                delta_cam_info,
            ) = self.build_code_and_cam(i)
            pred_dict = self.net("test", self.xy, self.uv, **code_info, **cam_info)
            coarse_fg_rgb = pred_dict["coarse_dict"]["merge_img"]

        return coarse_fg_rgb

    def optimize_gaze_direction(
        self,
        dataloader,
        n_epochs,
        index,
        method,
        is_gaze=True,
        gaze_direction=None,
        first_time=True,
    ):
        gaze_list = []
        for j in range(1):
            self.net.eval()
            self.train_len = len(dataloader.dataset)
            self.gaze_direction = torch.FloatTensor([[0.0 , 0.0]]).to(
                self.device
            )
            if gaze_direction is not None:
                self.gaze_direction = gaze_direction
            init_learn_rate = 0.01
            patiance = 35
            num_iter = 0
            best_gaze = self.gaze_direction
            best_loss = 100.0

            if not is_gaze:
                self.delta_EulurAngles = torch.zeros(
                    (self.train_len, 3), dtype=torch.float32
                ).to(self.device)
                self.delta_Tvecs = torch.zeros(
                    (self.train_len, 3, 1), dtype=torch.float32
                ).to(self.device)

                self.iden_offset = torch.zeros(
                    (self.train_len, 100), dtype=torch.float32
                ).to(self.device)

                self.expr_offset = torch.zeros(
                    (self.train_len, 79), dtype=torch.float32
                ).to(self.device)
                self.gaze_offset = torch.zeros(
                    (self.train_len, 2), dtype=torch.float32
                ).to(self.device)
                self.appea_offset = torch.zeros(
                    (self.train_len, 127), dtype=torch.float32
                ).to(self.device)

                self.enable_gradient(
                    [
                        # self.gaze_direction,
                        # self.gaze_offset,
                        self.iden_offset,
                        self.expr_offset,
                        self.appea_offset,
                        self.delta_EulurAngles,
                        self.delta_Tvecs,
                    ]
                )

                params_group = [
                    # {"params": [self.gaze_direction], "lr": init_learn_rate},
                    # {"params": [self.gaze_offset], "lr": init_learn_rate * 0.1},
                    {"params": [self.iden_offset], "lr": init_learn_rate * 1.0},
                    {"params": [self.expr_offset], "lr": init_learn_rate * 0.1},
                    {"params": [self.appea_offset], "lr": init_learn_rate * 1.0},
                    {"params": [self.delta_EulurAngles], "lr": init_learn_rate * 0.1},
                    {"params": [self.delta_Tvecs], "lr": init_learn_rate * 0.1},
                ]
            else:
                if first_time:
                    self.delta_EulurAngles = torch.zeros(
                        (self.train_len, 3), dtype=torch.float32
                    ).to(self.device)
                    self.delta_Tvecs = torch.zeros(
                        (self.train_len, 3, 1), dtype=torch.float32
                    ).to(self.device)

                    self.iden_offset = torch.zeros(
                        (self.train_len, 100), dtype=torch.float32
                    ).to(self.device)

                    self.expr_offset = torch.zeros(
                        (self.train_len, 79), dtype=torch.float32
                    ).to(self.device)
                    self.appea_offset = torch.zeros(
                        (self.train_len, 127), dtype=torch.float32
                    ).to(self.device)

                self.enable_gradient(
                    [
                        self.gaze_direction,
                        # self.iden_offset,
                        # self.expr_offset,
                        # self.appea_offset,
                        # self.delta_EulurAngles,
                        # self.delta_Tvecs,
                    ]
                )

                params_group = [
                    {"params": [self.gaze_direction], "lr": init_learn_rate},
                    # {"params": [self.iden_offset], "lr": init_learn_rate * 1.0},
                    # {"params": [self.expr_offset], "lr": init_learn_rate * 0.1},
                    # {"params": [self.appea_offset], "lr": init_learn_rate * 1.0},
                    # {"params": [self.delta_EulurAngles], "lr": init_learn_rate * 0.1},
                    # {"params": [self.delta_Tvecs], "lr": init_learn_rate * 0.1},
                ]

            self.get_optimizer(params_group)

            batch_loop_bar = tqdm(range(0, n_epochs), leave=True, desc="Batch Progress")

            # Loop over training batches
            for i in batch_loop_bar:
                if num_iter > patiance:
                    break
                self.net.eval()
                if method == "input_target_images":
                    for j, (
                        ind1,
                        ind2,
                        ind3,
                        ind4,
                        ldms,
                        cam_ind,
                        _,
                        _,
                        _,
                        _,
                        _,
                        _,
                        _,
                        _,
                        _,
                        _,
                    ) in enumerate(dataloader):
                        if j == index:
                            (
                                batch_images,
                                batch_head_mask,
                                batch_eye_mask,
                                batch_nl3dmm_para_dict,
                                ldms,
                                cam_ind,
                            ) = (ind1, ind2, ind3, ind4, ldms, cam_ind)
                            break
                elif method == "consistency" or method == "gaze_transfer" or method == "personal_calibration":
                    for j, (ind1, ind2, ind3, ind4, ind5, ldms, cam_ind) in enumerate(
                        dataloader
                    ):
                        if j == index:
                            (
                                batch_images,
                                batch_head_mask,
                                batch_left_eye_mask,
                                batch_right_eye_mask,
                                batch_nl3dmm_para_dict,
                                ldms,
                                cam_ind,
                            ) = (ind1, ind2, ind3, ind4, ind5, ldms, cam_ind)
                            break

                self.prepare_data(
                    batch_images,
                    batch_head_mask,
                    batch_left_eye_mask,
                    batch_right_eye_mask,
                    batch_nl3dmm_para_dict,
                )
                
                self.base_gaze_direction = self.gaze_direction
                loss_dict = self.perform_fitting(0, cam_ind, ldms, 1)
                loss = float(loss_dict["total_loss"].detach())
                if loss < best_loss:
                    best_loss = loss
                    best_gaze = self.gaze_direction
                    num_iter = 0
                else:
                    num_iter += 1
                batch_loop_bar.set_postfix(loss=float(loss_dict["total_loss"].detach()))
                del loss_dict
            gaze_list.append(best_gaze.detach().cpu().numpy())
        return torch.mean(torch.FloatTensor(np.array(gaze_list)), 0)


def get_trainer(**kwargs):
    return GazeNerfTrainer(**kwargs)