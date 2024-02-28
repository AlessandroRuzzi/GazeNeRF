import os
import sys

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
import copy
from glob import glob

import cv2
import numpy as np
import torch
from bisenet import BiSeNet
from correct_head_mask import correct_hair_mask
from torchvision.transforms import transforms
from tqdm import tqdm
from pre_processing.unet import unet
from utils.logging import log_mask


class GenMask(object):
    def __init__(self, gpu_id=0, log=False) -> None:
        super().__init__()

        if torch.cuda.is_available():
            self.device = "cuda:%s" % gpu_id
        else:
            self.device = "cpu"
        self.model_path = "configs/config_models/faceparsing_model.pth"
        self.log = log

        self.second_model_path = "configs/config_models/model.pth"

        self.init_model()
        self.lut = np.zeros((256,), dtype=np.uint8)
        self.lut[1:14] = 1
        self.lut[17] = 2

        self.de_lut = np.zeros((256,), dtype=np.uint8)
        self.de_lut[4:6] = 1

    def init_model(self):
        n_classes = 19  # background + the rest above
        net = BiSeNet(n_classes=n_classes).to(self.device)
        net.load_state_dict(torch.load(self.model_path, map_location=self.device))
        self.net = net
        self.net.eval()

        second_net = unet().to(self.device)
        second_net.load_state_dict(
            torch.load(self.second_model_path, map_location=self.device)
        )
        self.second_net = second_net
        self.second_net.eval()

        self.to_tensor = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )

        self.second_to_tensor = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

    def main_process(self, img_dir):
        img_path_list = [x for x in glob("%s/*.png" % img_dir) if "_mask" and "_big" not in x]
        if len(img_path_list) == 0:
            print("Dir: %s does include any .png images." % img_dir)
            exit(0)
        img_path_list.sort()
        loop_bar = tqdm(img_path_list)
        loop_bar.set_description("Generate masks")
        face_list = []
        left_eye_list = []
        right_eye_list = []
        for img_path in loop_bar:
            bgr_img = cv2.imread(img_path)  # input images resolution: (512, 512)
            img_to_log = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
            img = self.to_tensor(img_to_log)
            img = img.unsqueeze(0)
            img = img.to(self.device)
            
            if 'cam11' not in img_path and 'cam12' not in img_path and 'cam14' not in img_path and 'cam15' not in img_path:
            
                with torch.set_grad_enabled(False):
                    pred_res = self.net(img)  # pred_res(tuple) length: 3
                    out = pred_res[0]  # out(torch) size:[1,19,512,512]
    
                res_face = (
                    out.squeeze(0).cpu().numpy().argmax(0)
                )  # res(numpy.array) shape: (512, 512)
                res_face = res_face.astype(np.uint8)  # res(numpy.array) shape: (512, 512)
                cv2.LUT(res_face, self.lut, res_face)
                res_face = correct_hair_mask(res_face)
                res_face[res_face != 0] = 255
                face_list.append(res_face)
    
                res = out.squeeze(0).cpu().numpy().argmax(0)
                res = res.astype(np.uint8)
                cv2.LUT(res, self.de_lut, res)
                res[res != 0] = 255
    
                res_le = copy.deepcopy(res)
                res_le[:, int(res_le.shape[1] / 2) :] = 0
                if len(np.nonzero(res_le)[0]) != 0:
                    left_eye_list.append(res_le)
    
                res_re = copy.deepcopy(res)
                res_re[:, : int(res_le.shape[1] / 2)] = 0
                if len(np.nonzero(res_re)[0]) != 0:
                    right_eye_list.append(res_re)
    
                if len(np.nonzero(res_le)[0]) == 0 or len(np.nonzero(res_re)[0]) == 0:
                    img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
                    img = self.second_to_tensor(img)
                    img = img.unsqueeze(0)
                    img = img.to(self.device)
                    with torch.set_grad_enabled(False):
                        pred_res = self.second_net(img)
                        out = pred_res[0]
                    res = out.squeeze(0).cpu().numpy().argmax(0)
                    res = res.astype(np.uint8)
                    cv2.LUT(res, self.de_lut, res)
                    res[res != 0] = 255
    
                    if len(np.nonzero(res_le)[0]) == 0:
                        res_le = copy.deepcopy(res)
                        res_le[:, int(res_le.shape[1] / 2) :] = 0
                        if len(np.nonzero(res_le)[0]) != 0:
                            left_eye_list.append(res_le)
    
                    if len(np.nonzero(res_re)[0]) == 0:
                        res_re = copy.deepcopy(res)
                        res_re[:, : int(res_le.shape[1] / 2)] = 0
                        if len(np.nonzero(res_re)[0]) != 0:
                            right_eye_list.append(res_re)
    
                if len(np.nonzero(res_le)[0]) == 0 or len(np.nonzero(res_re)[0]) == 0:
                    landmark_path = img_path[:-4] + "_lm2d.txt"
                    landmark = np.loadtxt(landmark_path)
                    lm2d = landmark.reshape(68, 2)
                    le = np.int0(lm2d[36:42, :])
                    re = np.int0(lm2d[42:48, :])
                    if len(np.nonzero(res_le)[0]) == 0:
                        mask_img_le = np.zeros([512, 512, 3], np.uint8)
                        _ = cv2.fillPoly(mask_img_le, pts=[le], color=(255, 255, 255))
                        mask_img_le = cv2.cvtColor(mask_img_le, cv2.COLOR_BGR2GRAY)
                        left_eye_list.append(mask_img_le)
    
                    if len(np.nonzero(res_re)[0]) == 0:
                        mask_img_re = np.zeros([512, 512, 3], np.uint8)
                        _ = cv2.fillPoly(mask_img_re, pts=[re], color=(255, 255, 255))
                        mask_img_re = cv2.cvtColor(mask_img_re, cv2.COLOR_BGR2GRAY)
                        right_eye_list.append(mask_img_re)
            
            elif 'cam11' in img_path or 'cam12' in img_path:
                
                with torch.set_grad_enabled(False):
                    pred_res = self.net(img)  # pred_res(tuple) length: 3
                    out = pred_res[0]  # out(torch) size:[1,19,512,512]
    
                res_face = (
                    out.squeeze(0).cpu().numpy().argmax(0)
                )  # res(numpy.array) shape: (512, 512)
                res_face = res_face.astype(np.uint8)  # res(numpy.array) shape: (512, 512)
                cv2.LUT(res_face, self.lut, res_face)
                res_face = correct_hair_mask(res_face)
                res_face[res_face != 0] = 255
                face_list.append(res_face)
    
                res = out.squeeze(0).cpu().numpy().argmax(0)
                res = res.astype(np.uint8)
                cv2.LUT(res, self.de_lut, res)
                res[res != 0] = 255
    
                res_le = copy.deepcopy(res)
                res_le[:, int(res_le.shape[1] / 2) :] = 0
                if len(np.nonzero(res_le)[0]) != 0:
                    left_eye_list.append(res_le)
                    
                if len(np.nonzero(res_le)[0]) == 0:
                    img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
                    img = self.second_to_tensor(img)
                    img = img.unsqueeze(0)
                    img = img.to(self.device)
                    with torch.set_grad_enabled(False):
                        pred_res = self.second_net(img)
                        out = pred_res[0]
                    res = out.squeeze(0).cpu().numpy().argmax(0)
                    res = res.astype(np.uint8)
                    cv2.LUT(res, self.de_lut, res)
                    res[res != 0] = 255
                    
                    if len(np.nonzero(res_le)[0]) == 0:
                        res_le = copy.deepcopy(res)
                        res_le[:, int(res_le.shape[1] / 2) :] = 0
                        if len(np.nonzero(res_le)[0]) != 0:
                            left_eye_list.append(res_le)
                
                if len(np.nonzero(res_le)[0]) == 0:
                    landmark_path = img_path[:-4] + "_lm2d.txt"
                    landmark = np.loadtxt(landmark_path)
                    lm2d = landmark.reshape(68, 2)
                    le = np.int0(lm2d[36:42, :])
                    if len(np.nonzero(res_le)[0]) == 0:
                        mask_img_le = np.zeros([512, 512, 3], np.uint8)
                        _ = cv2.fillPoly(mask_img_le, pts=[le], color=(255, 255, 255))
                        mask_img_le = cv2.cvtColor(mask_img_le, cv2.COLOR_BGR2GRAY)
                        left_eye_list.append(mask_img_le)
                
                right_eye_list.append(np.zeros_like(res_le))
                
            elif 'cam14' in img_path or 'cam15' in img_path:
                
                with torch.set_grad_enabled(False):
                    pred_res = self.net(img)  # pred_res(tuple) length: 3
                    out = pred_res[0]  # out(torch) size:[1,19,512,512]
    
                res_face = (
                    out.squeeze(0).cpu().numpy().argmax(0)
                )  # res(numpy.array) shape: (512, 512)
                res_face = res_face.astype(np.uint8)  # res(numpy.array) shape: (512, 512)
                cv2.LUT(res_face, self.lut, res_face)
                res_face = correct_hair_mask(res_face)
                res_face[res_face != 0] = 255
                face_list.append(res_face)
    
                res = out.squeeze(0).cpu().numpy().argmax(0)
                res = res.astype(np.uint8)
                cv2.LUT(res, self.de_lut, res)
                res[res != 0] = 255
                    
                res_re = copy.deepcopy(res)
                res_re[:, : int(res_re.shape[1] / 2)] = 0
                if len(np.nonzero(res_re)[0]) != 0:
                    right_eye_list.append(res_re)
                    
                if len(np.nonzero(res_re)[0]) == 0:
                    img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
                    img = self.second_to_tensor(img)
                    img = img.unsqueeze(0)
                    img = img.to(self.device)
                    with torch.set_grad_enabled(False):
                        pred_res = self.second_net(img)
                        out = pred_res[0]
                    res = out.squeeze(0).cpu().numpy().argmax(0)
                    res = res.astype(np.uint8)
                    cv2.LUT(res, self.de_lut, res)
                    res[res != 0] = 255
                    
                    if len(np.nonzero(res_re)[0]) == 0:
                        res_re = copy.deepcopy(res)
                        res_re[:, : int(res_re.shape[1] / 2)] = 0
                        if len(np.nonzero(res_re)[0]) != 0:
                            right_eye_list.append(res_re)
                    
                if len(np.nonzero(res_re)[0]) == 0:
                    landmark_path = img_path[:-4] + "_lm2d.txt"
                    landmark = np.loadtxt(landmark_path)
                    lm2d = landmark.reshape(68, 2)
                    re = np.int0(lm2d[42:48, :])
                    if len(np.nonzero(res_re)[0]) == 0:
                        mask_img_re = np.zeros([512, 512, 3], np.uint8)
                        _ = cv2.fillPoly(mask_img_re, pts=[re], color=(255, 255, 255))
                        mask_img_re = cv2.cvtColor(mask_img_re, cv2.COLOR_BGR2GRAY)
                        right_eye_list.append(mask_img_re)
                
                left_eye_list.append(np.zeros_like(res_re))
                    
            

            if self.log:
                class_labels = {0: "background", 255: "face"}
                log_mask(img_to_log, face_list[-1], "Face Segmentation", class_labels)

                class_labels = {0: "background", 255: "left eye"}
                log_mask(img_to_log, left_eye_list[-1], "Left Eye Segmentation", class_labels)

                class_labels = {0: "background", 255: "right eye"}
                log_mask(img_to_log, right_eye_list[-1], "Right Eye Segmentation", class_labels)

        return face_list, left_eye_list, right_eye_list


def generate_masks(source_path):

    tt = GenMask()
    face_list, left_eye_list, right_eye_list = tt.main_process(img_dir=source_path)

    return face_list, left_eye_list, right_eye_list


if __name__ == "__main__":

    source_path = "/home/nfs/xshi2/test_merge_codes/images/subject0040/frame0000"
    face_list, left_eye_list, right_eye_list = generate_masks(source_path)
