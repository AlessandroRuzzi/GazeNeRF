import math
from collections import OrderedDict

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def rotation_matrix_2d(pseudo_label, inverse=False):
    cos = torch.cos(pseudo_label)
    sin = torch.sin(pseudo_label)
    ones = torch.ones_like(cos[:, 0])
    zeros = torch.zeros_like(cos[:, 0])
    matrices_1 = torch.stack(
        [ones, zeros, zeros, zeros, cos[:, 0], -sin[:, 0], zeros, sin[:, 0], cos[:, 0]],
        dim=1,
    )
    matrices_2 = torch.stack(
        [cos[:, 1], zeros, sin[:, 1], zeros, ones, zeros, -sin[:, 1], zeros, cos[:, 1]],
        dim=1,
    )
    matrices_1 = matrices_1.view(-1, 3, 3)
    matrices_2 = matrices_2.view(-1, 3, 3)
    matrices = torch.matmul(matrices_2, matrices_1)
    if inverse:
        matrices = torch.transpose(matrices, 1, 2)
    return matrices


def rotate(embedding, pseudo_label, inverse=False):

    rotated_embeddings = []
    for i in range(embedding.shape[0]):

        rotation_matrix = rotation_matrix_2d(
            pseudo_label[i : i + 1, :], inverse=inverse
        )

        rotated_embeddings.append(
            torch.matmul(torch.transpose(torch.transpose(embedding[i : i + 1, :], 2, 3), 3, 4), rotation_matrix)
        )
    rotated_embeddings_ = torch.cat([j for j in rotated_embeddings], dim=0)

    return rotated_embeddings_


def soft_load_model(net, pre_state_dict):

    if isinstance(pre_state_dict, str):
        pre_state_dict = torch.load(pre_state_dict, map_location="cpu")

    state_dict = net.state_dict()
    increment_state_dict = OrderedDict()

    for k, v in pre_state_dict.items():
        if k in state_dict:
            if v.size() == state_dict[k].size():
                increment_state_dict[k] = v

    state_dict.update(increment_state_dict)
    net.load_state_dict(state_dict)
    return net


def draw_res_img(
    rendered_imgs_0, ori_imgs, batch_mask, proj_lm2ds=None, gt_lm2ds=None, num_per_row=1
):
    num = rendered_imgs_0.size(0)
    res_list = []

    rendered_imgs = rendered_imgs_0.clone()
    rendered_imgs *= 255.0
    observed_imgs = ori_imgs.clone()
    observed_imgs *= 255.0

    if isinstance(rendered_imgs, torch.Tensor):
        rendered_imgs = (
            rendered_imgs.detach().cpu().numpy()
            if rendered_imgs.is_cuda
            else rendered_imgs.numpy()
        )

    if isinstance(observed_imgs, torch.Tensor):
        observed_imgs = (
            observed_imgs.detach().cpu().numpy()
            if observed_imgs.is_cuda
            else observed_imgs.numpy()
        )

    if isinstance(batch_mask, torch.Tensor):
        batch_mask = (
            batch_mask.detach().cpu().numpy()
            if batch_mask.is_cuda
            else batch_mask.numpy()
        )

    if gt_lm2ds is not None:
        if isinstance(gt_lm2ds, torch.Tensor):
            gt_lm2ds = gt_lm2ds.detach().cpu().numpy() if gt_lm2ds.is_cuda else gt_lm2ds

    if proj_lm2ds is not None:
        if isinstance(proj_lm2ds, torch.Tensor):
            proj_lm2ds = (
                proj_lm2ds.detach().cpu().numpy() if proj_lm2ds.is_cuda else proj_lm2ds
            )

    for cnt in range(num):
        re_img = rendered_imgs[cnt]
        mask = batch_mask[cnt]
        ori_img = observed_imgs[cnt]

        temp_img_1 = ori_img.copy()
        temp_img_2 = ori_img.copy()
        ori_img[mask] = re_img[mask]

        if gt_lm2ds is not None:
            lm2ds = gt_lm2ds[cnt]
            for lm2d in lm2ds:
                temp_img_2 = cv2.circle(
                    temp_img_2,
                    center=(int(lm2d[0]), int(lm2d[1])),
                    radius=2,
                    color=(255, 0, 0),
                    thickness=1,
                )

        if proj_lm2ds is not None:
            lm2ds = proj_lm2ds[cnt]
            for lm2d in lm2ds:
                temp_img_2 = cv2.circle(
                    temp_img_2,
                    center=(int(lm2d[0]), int(lm2d[1])),
                    radius=2,
                    color=(0, 0, 255),
                    thickness=1,
                )

        img = np.concatenate([temp_img_1, temp_img_2, ori_img], axis=1)
        res_list.append(img)

    if num == 1:
        res = np.concatenate(res_list, axis=0)
    elif num_per_row != 1:
        n_rows = num // num_per_row

        last_res_imgs = []
        for cnt in range(n_rows):
            temp_res = np.concatenate(
                res_list[cnt * num_per_row : cnt * num_per_row + num_per_row], axis=1
            )
            last_res_imgs.append(temp_res)

        if num % num_per_row > 0:
            temp_img = np.ones_like(last_res_imgs[-1])
            temp_res = np.concatenate(res_list[n_rows * num_per_row :], axis=1)
            _, w, _ = temp_res.shape
            temp_img[:, :w, :] = temp_res
            last_res_imgs.append(temp_img)

        res = np.concatenate(last_res_imgs, axis=0)
    else:
        res = np.concatenate(res_list, axis=0)
    return res


def convert_loss_dict_2_str(loss_dict):
    res = ""
    for k, v in loss_dict.items():
        res = res + "{}:{:.04f}, ".format(k, v)
    res = res[:-2]
    return res


def put_text_alignmentcenter(img, img_size, text_str, color, offset_x):

    font = cv2.FONT_HERSHEY_COMPLEX
    textsize = cv2.getTextSize(text_str, font, 1, 2)[0]

    textX = (img_size - textsize[0]) // 2 + offset_x
    textY = img_size - textsize[1]

    img = cv2.putText(img, text_str, (textX, textY), font, 1, color, 2)

    return img


def eulurangle2Rmat(angles):
    """
    angles: (3, 1) or (1, 3)
    """
    angles = angles.reshape(-1)
    sinx = np.sin(angles[0])
    siny = np.sin(angles[1])
    sinz = np.sin(angles[2])
    cosx = np.cos(angles[0])
    cosy = np.cos(angles[1])
    cosz = np.cos(angles[2])

    mat_x = np.eye(3, dtype=np.float32)
    mat_y = np.eye(3, dtype=np.float32)
    mat_z = np.eye(3, dtype=np.float32)

    mat_x[1, 1] = cosx
    mat_x[1, 2] = -sinx
    mat_x[2, 1] = sinx
    mat_x[2, 2] = cosx

    mat_y[0, 0] = cosy
    mat_y[0, 2] = siny
    mat_y[2, 0] = -siny
    mat_y[2, 2] = cosy

    mat_z[0, 0] = cosz
    mat_z[0, 1] = -sinz
    mat_z[1, 0] = sinz
    mat_z[1, 1] = cosz

    res = mat_z.dot(mat_y.dot(mat_x))

    return res


def Rmat2EulurAng(Rmat):
    sy = math.sqrt(Rmat[0, 0] * Rmat[0, 0] + Rmat[1, 0] * Rmat[1, 0])

    if sy > 1e-6:
        x = math.atan2(Rmat[2, 1], Rmat[2, 2])
        y = math.atan2(-Rmat[2, 0], sy)
        z = math.atan2(Rmat[1, 0], Rmat[0, 0])
    else:
        x = math.atan2(-Rmat[1, 2], Rmat[1, 1])
        y = math.atan2(-Rmat[2, 0], sy)
        z = 0

    return np.array([x, y, z])


class Embedder(nn.Module):
    def __init__(self, N_freqs, include_input, input_dims=3) -> None:
        super().__init__()

        self.log_sampling = True
        self.periodic_fns = [torch.sin, torch.cos]
        self.max_freq = N_freqs - 1
        self.N_freqs = N_freqs
        self.include_input = include_input
        self.input_dims = input_dims

        self._Pre_process()

    def _Pre_process(self):

        embed_fns = []

        if self.include_input:
            embed_fns.append(lambda x: x)

        if self.log_sampling:
            freq_bands = 2.0 ** torch.linspace(0.0, self.max_freq, steps=self.N_freqs)
        else:
            freq_bands = torch.linspace(
                2.0**0.0, 2.0**self.max_freq, steps=self.N_freqs
            )

        for freq in freq_bands:
            for p_fn in self.periodic_fns:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
        self.embed_fns = embed_fns

    def forward(self, x):
        """
        x: [B, 3, N_1, N_2]
        """

        res = [fn(x) for fn in self.embed_fns]
        res = torch.cat(res, dim=1)

        return res


class GenSamplePoints(nn.Module):
    def __init__(self, opt):
        super().__init__()

        self.world_z1 = opt.world_z1
        self.world_z2 = opt.world_z2
        self.n_sample_fg = opt.num_sample_coarse

    @staticmethod
    def _calc_sample_points_by_zvals(
        zvals, batch_ray_o, batch_ray_d, batch_ray_l, disturb
    ):
        """
        zvals      :[B, N_r, N_p + 1]
        batch_ray_o:[B, 3,   N_r    ,   1]
        batch_ray_d:[B, 3,   N_r    ,   1]
        batch_ray_l:[B, 1,   N_r    ,   1]
        """

        if disturb:
            mids = 0.5 * (zvals[:, :, 1:] + zvals[:, :, :-1])
            upper = torch.cat([mids, zvals[:, :, -1:]], dim=-1)
            lower = torch.cat([zvals[:, :, :1], mids], dim=-1)
            t_rand = torch.rand_like(zvals)
            zvals = lower + (upper - lower) * t_rand

        z_dists = zvals[:, :, 1:] - zvals[:, :, :-1]  # [B, N_r, N_p]
        z_dists = (z_dists.unsqueeze(1)) * batch_ray_l  # [B, 1, N_r, N_p]

        zvals = zvals[:, :, :-1]  # [B, N_r, N_p]
        zvals = zvals.unsqueeze(1)  # [B, 1, N_r, N_p]

        sample_pts = batch_ray_o + batch_ray_d * batch_ray_l * zvals  # [B, 3, N_r, N_p]

        n_sample = zvals.size(-1)
        sample_dirs = batch_ray_d.expand(-1, -1, -1, n_sample)

        res = {
            "pts": sample_pts,  # [B, 3, N_r, N_p]
            "dirs": sample_dirs,  # [B, 3, N_r, N_p]
            "zvals": zvals,  # [B, 1, N_r, N_p]
            "z_dists": z_dists,  # [B, 1, N_r, N_p]
            "batch_ray_o": batch_ray_o,  # [B, 3, N_r, 1]
            "batch_ray_d": batch_ray_d,  # [B, 3, N_r, 1]
            "batch_ray_l": batch_ray_l,  # [B, 1, N_r, 1]
        }

        return res

    def _calc_sample_points(self, batch_ray_o, batch_ray_d, batch_ray_l, disturb):
        """
        batch_ray_o:[B, 3, N_r]
        batch_ray_d:[B, 3, N_r]
        batch_ray_l:[B, 1, N_r]
        """

        rela_z1 = batch_ray_o[:, -1, :] - self.world_z1  # [B, N_r]
        rela_z2 = batch_ray_o[:, -1, :] - self.world_z2  # [B, N_r]

        rela_z1 = rela_z1.unsqueeze(-1)  # [B, N_r, 1]
        rela_z2 = rela_z2.unsqueeze(-1)  # [B, N_r, 1]

        data_type = batch_ray_o.dtype
        data_device = batch_ray_o.device

        batch_ray_o = batch_ray_o.unsqueeze(-1)
        batch_ray_d = batch_ray_d.unsqueeze(-1)
        batch_ray_l = batch_ray_l.unsqueeze(-1)

        t_vals_fg = torch.linspace(
            0.0, 1.0, steps=self.n_sample_fg + 1, dtype=data_type, device=data_device
        ).view(1, 1, self.n_sample_fg + 1)
        sample_zvals_fg = (
            rela_z1 * (1.0 - t_vals_fg) + rela_z2 * t_vals_fg
        )  # [B, N_r, N_p + 1]
        sample_dict_fg = self._calc_sample_points_by_zvals(
            sample_zvals_fg, batch_ray_o, batch_ray_d, batch_ray_l, disturb
        )

        return sample_dict_fg

    def forward(self, batch_xy, batch_Rmat, batch_Tvec, batch_inv_inmat, disturb):
        temp_xyz = F.pad(batch_xy, [0, 0, 0, 1, 0, 0], mode="constant", value=1.0)
        ray_d = batch_Rmat.bmm(batch_inv_inmat.bmm(temp_xyz))
        ray_l = torch.norm(ray_d, dim=1, keepdim=True)
        ray_d = ray_d / ray_l
        ray_l = -1.0 / ray_d[:, -1:, :]

        batch_size, _, num_ray = batch_xy.size()
        ray_o = batch_Tvec.expand(batch_size, 3, num_ray)

        fg_sample_dict = self._calc_sample_points(ray_o, ray_d, ray_l, disturb)
        return fg_sample_dict


class FineSample(nn.Module):
    def __init__(self, opt) -> None:
        super().__init__()
        self.n_sample = opt.num_sample_fine + 1
        # self.world_z2 = opt.world_z2

    @staticmethod
    def _calc_sample_points_by_zvals(zvals, batch_ray_o, batch_ray_d, batch_ray_l):
        """
        zvals      :[B, N_r, N_p + 1]
        batch_ray_o:[B, 3,   N_r    ,   1]
        batch_ray_d:[B, 3,   N_r    ,   1]
        batch_ray_l:[B, 1,   N_r    ,   1]
        """

        z_dists = zvals[:, :, 1:] - zvals[:, :, :-1]  # [B, N_r, N_p]
        z_dists = (z_dists.unsqueeze(1)) * batch_ray_l  # [B, 1, N_r, N_p]

        zvals = zvals[:, :, :-1]  # [B, N_r, N_p]
        zvals = zvals.unsqueeze(1)  # [B, 1, N_r, N_p]

        sample_pts = batch_ray_o + batch_ray_d * batch_ray_l * zvals  # [B, 3, N_r, N_p]

        n_sample = zvals.size(-1)
        sample_dirs = batch_ray_d.expand(-1, -1, -1, n_sample)

        res = {
            "pts": sample_pts,  # [B, 3, N_r, N_p]
            "dirs": sample_dirs,  # [B, 3, N_r, N_p]
            "zvals": zvals,  # [B, 1, N_r, N_p]
            "z_dists": z_dists,  # [B, 1, N_r, N_p]
        }

        return res

    def forward(self, batch_weight, coarse_sample_dict, disturb):

        NFsample = self.n_sample
        coarse_zvals = coarse_sample_dict["zvals"]  # [B, 1, N_r, N_c]

        temp_weight = batch_weight[:, :, :, 1:-1].detach()
        (
            batch_size,
            _,
            num_ray,
            temp_NCsample,
        ) = temp_weight.size()  # temp_Csample = N_c - 2
        temp_weight = temp_weight.view(-1, temp_NCsample)  # [N_t, N_c - 2]

        x = temp_weight + 1e-5
        pdf = temp_weight / torch.sum(x, dim=-1, keepdim=True)
        cdf = torch.cumsum(pdf, dim=-1)
        cdf = F.pad(cdf, pad=[1, 0, 0, 0], mode="constant", value=0.0)  # [N_t, N_c - 1]
        cdf = cdf.contiguous()

        num_temp = cdf.size(0)  # B*N_r
        if disturb:
            uniform_sample = torch.rand(
                num_temp, NFsample, device=batch_weight.device, dtype=batch_weight.dtype
            )  # [N_t, N_f]
        else:
            uniform_sample = (
                torch.linspace(
                    0.0,
                    1.0,
                    steps=NFsample,
                    device=batch_weight.device,
                    dtype=batch_weight.dtype,
                )
                .view(1, NFsample)
                .expand(num_temp, NFsample)
            )
        uniform_sample = uniform_sample.contiguous()  # [N_t, N_f]

        inds = torch.searchsorted(cdf, uniform_sample, right=True)  # [N_t, N_f]
        below = torch.max(torch.zeros_like(inds), inds - 1)
        above = torch.min(temp_NCsample * torch.ones_like(inds), inds)
        inds_g = torch.stack([below, above], dim=-1)  # [N_t, N_f, 2]

        temp_coarse_vpz = coarse_zvals.view(num_temp, temp_NCsample + 2)
        bins = 0.5 * (
            temp_coarse_vpz[:, 1:] + temp_coarse_vpz[:, :-1]
        )  # [N_t, N_c - 1]

        cdf_g = torch.gather(
            cdf.unsqueeze(1).expand(num_temp, NFsample, temp_NCsample + 1), 2, inds_g
        )  # [N_t, n_f, 2]
        bins_g = torch.gather(
            bins.unsqueeze(1).expand(num_temp, NFsample, temp_NCsample + 1), 2, inds_g
        )  # [N_t, n_f, 2]

        denom = cdf_g[:, :, 1] - cdf_g[:, :, 0]  # [N_t, N_f]
        denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
        t = (uniform_sample - cdf_g[:, :, 0]) / denom
        fine_sample_vz = bins_g[:, :, 0] + t * (
            bins_g[:, :, 1] - bins_g[:, :, 0]
        )  # [N_t, N_f]

        fine_sample_vz, _ = torch.sort(
            torch.cat([temp_coarse_vpz, fine_sample_vz], dim=-1), dim=-1
        )  # [N_t, N_f + N_c]
        fine_sample_vz = fine_sample_vz.view(
            batch_size, num_ray, NFsample + temp_NCsample + 2
        )

        res = self._calc_sample_points_by_zvals(
            fine_sample_vz,
            coarse_sample_dict["batch_ray_o"],
            coarse_sample_dict["batch_ray_d"],
            coarse_sample_dict["batch_ray_l"],
        )

        return res


class CalcRayColor(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def _calc_alpha(batch_density, batch_dists):

        res = 1.0 - torch.exp(-batch_density * batch_dists)
        return res

    @staticmethod
    def _calc_weight(batch_alpha):
        """
        batch_alpha:[B, 1, N_r, N_p]
        """
        x = 1.0 - batch_alpha + 1e-10
        x = F.pad(x, [1, 0, 0, 0, 0, 0, 0, 0], mode="constant", value=1.0)
        x = torch.cumprod(x, dim=-1)

        res = batch_alpha * x[:, :, :, :-1]

        return res

    def forward(self, fg_vps, batch_rgb, batch_density, batch_dists, batch_z_vals):

        """
        batch_rgb: [B, 3, N_r, N_p]
        batch_density: [B, 1, N_r, N_p]
        batch_dists: [B, 1, N_r, N_p]
        batch_z_vals:[B, N_r, N_p]
        """

        batch_alpha = self._calc_alpha(batch_density, batch_dists)
        batch_weight = self._calc_weight(batch_alpha)

        rgb_res = torch.sum(batch_weight * batch_rgb, dim=-1)  # [B, 3, N_r]
        depth_res = torch.sum(batch_weight * batch_z_vals, dim=-1)  # [B, 1, N_r]

        acc_weight = torch.sum(batch_weight, dim=-1)  # [B, 1, N_r]
        bg_alpha = 1.0 - acc_weight

        return rgb_res, bg_alpha, depth_res, batch_weight


if __name__ == "__main__":
    angle = np.array([0.00872665, 0.337, 0.113])
    rmat = eulurangle2Rmat(angle)
    res = Rmat2EulurAng(rmat)

    print(angle)
    print(res)
